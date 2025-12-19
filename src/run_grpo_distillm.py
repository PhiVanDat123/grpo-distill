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
GRPO + DistiLLM-v2 Training Script for Math Tasks

This script trains a student model using:
1. GRPO (Group Relative Policy Optimization) for RL-based training
2. DistiLLM-v2 for knowledge distillation from a teacher model
3. Math reward functions from math_rewards.py

Key features:
- Samples G responses per prompt for group-relative advantages
- Uses adaptive mixture KL divergence for distillation
- Supports math datasets with solution verification
"""

import logging
import random
import re
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
from alignment.grpo_configs import GRPOConfig
from peft import PeftConfig, PeftModel
from grpo_distillm_trainer import GRPODistiLLMTrainer

# =============================================================================
# IMPORT REWARD FUNCTIONS FROM math_rewards.py
# =============================================================================
from math_reward import accuracy_reward


logger = logging.getLogger(__name__)


# =============================================================================
# Solution Extraction for MetaMathQA
# =============================================================================

def extract_solution_from_response(response: str) -> str:
    """
    Extract the final answer from MetaMathQA response format.
    
    MetaMathQA responses typically end with:
    - "The answer is X"
    - "#### X" 
    - "\\boxed{X}"
    """
    if not response:
        return ""
    
    # Try "The answer is X" pattern
    match = re.search(r'[Tt]he answer is[:\s]*([^\.\n]+)', response)
    if match:
        return match.group(1).strip()
    
    # Try "#### X" pattern (GSM8K style)
    match = re.search(r'####\s*(.+?)(?:\n|$)', response)
    if match:
        return match.group(1).strip()
    
    # Try \boxed{X} pattern (MATH style)
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        return match.group(1).strip()
    
    # Fallback: try to find last number
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        return numbers[-1]
    
    return response.strip()


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
        columns_to_keep=["query", "original_question", "response", "answer", "solution"],
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    logger.info(f"Dataset columns: {column_names}")

    #####################################
    # Load tokenizer
    #####################################
    data_args.truncation_side = "left"
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Prepare prompts and solutions
    #####################
    def extract_prompt_and_solution(example):
        """
        Extract prompt and solution from various dataset formats.
        
        Supports:
        - MetaMathQA: query, response
        - GSM8K: question, answer
        - Pre-processed: prompt, solution
        """
        result = {}
        
        # === Extract Prompt ===
        if "prompt" in example and example["prompt"]:
            if isinstance(example["prompt"], list):
                result["prompt"] = example["prompt"]
            else:
                result["prompt"] = [{"role": "user", "content": example["prompt"]}]
        
        elif "messages" in example and example["messages"]:
            messages = example["messages"]
            if messages[-1]["role"] == "assistant":
                result["prompt"] = messages[:-1]
            else:
                result["prompt"] = messages
        
        elif "query" in example and example["query"]:
            # MetaMathQA format
            result["prompt"] = [{"role": "user", "content": example["query"]}]
        
        elif "question" in example and example["question"]:
            # GSM8K format
            result["prompt"] = [{"role": "user", "content": example["question"]}]
        
        elif "chosen" in example:
            if isinstance(example["chosen"], list):
                messages = example["chosen"]
                prompt = []
                for msg in messages:
                    if msg["role"] != "assistant":
                        prompt.append(msg)
                    else:
                        break
                result["prompt"] = prompt
            else:
                result["prompt"] = [{"role": "user", "content": example["chosen"].split("\n")[0]}]
        else:
            raise ValueError(f"Cannot extract prompt from example: {example.keys()}")
        
        # === Extract Solution ===
        if "solution" in example and example["solution"]:
            # Already has solution column
            result["solution"] = str(example["solution"])
        
        elif "response" in example and example["response"]:
            # MetaMathQA: extract answer from response
            result["solution"] = extract_solution_from_response(example["response"])
        
        elif "answer" in example and example["answer"]:
            # GSM8K format: "... #### X"
            answer = example["answer"]
            if "####" in answer:
                result["solution"] = answer.split("####")[1].strip()
            else:
                result["solution"] = extract_solution_from_response(answer)
        
        else:
            # No solution available
            result["solution"] = ""
        
        return result

    raw_datasets = raw_datasets.map(
        extract_prompt_and_solution,
        num_proc=data_args.preprocessing_num_workers,
        desc="Extracting prompts and solutions",
    )

    # Log a few random samples
    logger.info("Sample examples after processing:")
    for index in random.sample(range(len(raw_datasets["train"])), min(3, len(raw_datasets["train"]))):
        example = raw_datasets["train"][index]
        logger.info(f"  Example {index}:")
        logger.info(f"    Prompt: {example['prompt']}")
        logger.info(f"    Solution: {example.get('solution', 'N/A')}")

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
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    
    # Add attention implementation if specified
    if model_args.attn_implementation:
        model_kwargs["attn_implementation"] = model_args.attn_implementation

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
    teacher_model = None
    if model_args.ref_model_name_or_path is not None:
        logger.info(f"Loading teacher model from {model_args.ref_model_name_or_path}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            model_args.ref_model_name_or_path,
            **model_kwargs,
        )
    else:
        logger.warning("No teacher model specified! Distillation will be disabled (beta=0).")

    #########################
    # Setup reward functions
    #########################
    # Kiểm tra xem dataset có solution không
    has_solutions = "solution" in raw_datasets["train"].column_names and \
                   raw_datasets["train"][0].get("solution", "") != ""
    
    if has_solutions:
        logger.info("Dataset has solutions - using accuracy-based rewards from math_rewards.py")
        
        # Sử dụng reward functions từ math_rewards.py
        reward_funcs = [
            accuracy_reward,      # Main: đánh giá đúng/sai
        ]
        reward_kwargs = {}  # accuracy_reward sẽ nhận solutions từ trainer
    
    logger.info(f"Using {len(reward_funcs)} reward functions:")
    for func in reward_funcs:
        logger.info(f"  - {func.__name__}")

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
        beta=training_args.beta if teacher_model else 0.0,  # Disable distillation if no teacher
        
        # DistiLLM specific
        base_alpha_1=training_args.base_alpha_1,
        base_alpha_2=training_args.base_alpha_2,
        
        # Generation
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        max_new_tokens=training_args.max_new_tokens,
        temperature=training_args.temperature,
        top_p=training_args.top_p,
        
        # Reward - SỬ DỤNG REWARD TỪ math_rewards.py
        reward_funcs=reward_funcs,
        reward_kwargs=reward_kwargs,
        
        # Dataset column for solutions
        solution_column="solution",
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Starting GRPO + DistiLLM-v2 Training ***")
    logger.info(f"  Num samples per prompt (G): {training_args.num_samples_per_prompt}")
    logger.info(f"  Clip epsilon: {training_args.clip_epsilon}")
    logger.info(f"  Beta (distillation weight): {training_args.beta if teacher_model else 0.0}")
    
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

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()) if data_args.dataset_mixer else [],
        "dataset_tags": list(data_args.dataset_mixer.keys()) if data_args.dataset_mixer else [],
        "tags": ["grpo", "distillm", "math", "alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
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
    os.environ["WANDB_DISABLED"] = "true"
    main()