#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
GRPO + DistiLLM-v2 Training Script

This script trains a student model using:
1. GRPO (Group Relative Policy Optimization) for RL-based training
2. DistiLLM-v2 for knowledge distillation from a teacher model

Key features:
- Samples G responses per prompt for group-relative advantages
- Uses adaptive mixture KL divergence for distillation
- Teacher model guides the student, NOT used as GRPO baseline
- π_θ_old (old policy) is the student from previous timestep
"""

import logging
import random
import sys
import os
from typing import List, Dict, Any

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
from datasets import load_dataset, DatasetDict

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from alignment.configs import GRPOConfig
from peft import PeftConfig, PeftModel
from grpo_distillm_trainer import GRPODistiLLMTrainer

logger = logging.getLogger(__name__)


def length_reward(
    prompts: List[str],
    responses: List[str],
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Simple length-based reward function (example).
    Rewards responses that are not too short or too long.
    
    Replace this with your actual reward function!
    """
    rewards = []
    for response in responses:
        words = len(response.split())
        # Reward peaks around 50-100 words
        if words < 10:
            r = 0.1
        elif words < 50:
            r = 0.5 + 0.5 * (words / 50)
        elif words <= 150:
            r = 1.0
        else:
            r = max(0.5, 1.0 - (words - 150) / 200)
        rewards.append(r)
    return torch.tensor(rewards, dtype=torch.float32, device=response_ids.device)


def format_reward(
    prompts: List[str],
    responses: List[str],
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Reward for good formatting (example).
    Rewards responses with proper structure.
    
    Replace this with your actual reward function!
    """
    rewards = []
    for response in responses:
        r = 0.5  # Base reward
        
        # Reward for having sentences
        if '.' in response or '!' in response or '?' in response:
            r += 0.2
        
        # Reward for not having repetition
        words = response.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            r += 0.3 * unique_ratio
        
        rewards.append(r)
    return torch.tensor(rewards, dtype=torch.float32, device=response_ids.device)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, GRPOConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "prompt", "chosen", "rejected", "completion", "label"],
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose important context
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Prepare prompts
    #####################
    def extract_prompt(example):
        """Extract prompt from various dataset formats."""
        if "prompt" in example and example["prompt"]:
            # Direct prompt field
            if isinstance(example["prompt"], list):
                # Chat format - use as is
                return {"prompt": example["prompt"]}
            else:
                return {"prompt": example["prompt"]}
        elif "messages" in example and example["messages"]:
            # Extract prompt from messages (all but last if it's assistant)
            messages = example["messages"]
            if messages[-1]["role"] == "assistant":
                prompt = messages[:-1]
            else:
                prompt = messages
            return {"prompt": prompt}
        elif "chosen" in example:
            # DPO format - use the prompt part from chosen
            if isinstance(example["chosen"], list):
                # Chat format
                messages = example["chosen"]
                # Find the last user message as prompt
                prompt = []
                for msg in messages:
                    if msg["role"] != "assistant":
                        prompt.append(msg)
                    else:
                        break
                return {"prompt": prompt}
            else:
                # Text format - this is tricky, might need custom handling
                return {"prompt": example.get("text_prompt", example["chosen"].split("\n")[0])}
        else:
            raise ValueError(f"Cannot extract prompt from example: {example.keys()}")

    raw_datasets = raw_datasets.map(
        extract_prompt,
        num_proc=data_args.preprocessing_num_workers,
        desc="Extracting prompts",
    )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), min(3, len(raw_datasets["train"]))):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")

    #######################
    # Load models
    #######################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Load student model
    logger.info(f"Loading student model from {model_args.model_name_or_path}")
    if is_adapter_model(model_args.model_name_or_path, model_args.model_revision):
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            revision=model_args.base_model_revision,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )

    # Load teacher model
    if model_args.ref_model_name_or_path is not None:
        logger.info(f"Loading teacher model from {model_args.ref_model_name_or_path}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            model_args.ref_model_name_or_path,
            **model_kwargs,
        )
    else:
        logger.warning("No teacher model specified! Using student model as teacher (not recommended).")
        teacher_model = None

    #########################
    # Setup reward functions
    #########################
    # You can customize these reward functions based on your task
    reward_funcs = [
        length_reward,
        format_reward,
    ]
    reward_weights = [0.5, 0.5]  # Equal weights
    
    logger.info(f"Using {len(reward_funcs)} reward functions with weights {reward_weights}")

    #########################
    # Instantiate GRPO trainer
    #########################
    trainer = GRPODistiLLMTrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets.get("test"),
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
        # GRPO specific
        num_samples_per_prompt=training_args.num_samples_per_prompt,
        clip_epsilon=training_args.clip_epsilon,
        beta=training_args.beta,
        # DistiLLM specific
        base_alpha_1=training_args.base_alpha_1,
        base_alpha_2=training_args.base_alpha_2,
        # Generation
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        max_new_tokens=training_args.max_new_tokens,
        temperature=training_args.temperature,
        top_p=training_args.top_p,
        # Reward
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Starting GRPO + DistiLLM-v2 Training ***")
    
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()) if data_args.dataset_mixer else [],
        "dataset_tags": list(data_args.dataset_mixer.keys()) if data_args.dataset_mixer else [],
        "tags": ["grpo", "distillm", "alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets.get("test", []))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"  # Disable wandb by default
    main()