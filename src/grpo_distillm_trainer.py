# GRPO + DistiLLM-v2 Trainer
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
GRPO (Group Relative Policy Optimization) combined with DistiLLM-v2 distillation loss.

Key differences from standard GRPO:
- KL regularization term is replaced with DistiLLM-v2's adaptive mixture KL divergence
- Teacher model is used for distillation (NOT as the baseline for GRPO)
- π_θ_old (old policy) is the student model from the previous timestep
- Each prompt samples G=4 responses for group-based advantage estimation
"""
import math
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    GenerationConfig,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from trl.import_utils import is_peft_available, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


class GRPODataCollatorWithPadding:
    """
    Data collator for GRPO that handles prompt-only data.
    Different from DPO - we only need prompts, responses are generated online.
    """
    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
    ):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Get max length for padding
        max_prompt_length = max(len(f["prompt_input_ids"]) for f in features)
        
        batch = {
            "prompt_input_ids": [],
            "prompt_attention_mask": [],
            "prompt": [],
        }
        
        for feature in features:
            prompt_ids = feature["prompt_input_ids"]
            prompt_mask = feature["prompt_attention_mask"]
            
            # Pad from left for prompt (decoder-only models)
            padding_length = max_prompt_length - len(prompt_ids)
            prompt_ids = [self.pad_token_id] * padding_length + prompt_ids
            prompt_mask = [0] * padding_length + prompt_mask
            
            batch["prompt_input_ids"].append(prompt_ids)
            batch["prompt_attention_mask"].append(prompt_mask)
            batch["prompt"].append(feature.get("prompt", ""))
        
        batch["prompt_input_ids"] = torch.tensor(batch["prompt_input_ids"], dtype=torch.long)
        batch["prompt_attention_mask"] = torch.tensor(batch["prompt_attention_mask"], dtype=torch.long)
        
        return batch


class GRPODistiLLMTrainer(Trainer):
    """
    GRPO Trainer with DistiLLM-v2 distillation loss.
    
    Loss function:
    J_GRPO(θ) = E[q~P(Q), {o_i}~π_θ_old(O|q)] * (1/G) * Σ (1/|o_i|) * Σ_t [
        min(
            r_t * A_hat_i,t,
            clip(r_t, 1-ε, 1+ε) * A_hat_i,t
        )
    ] - β * DistiLLM_v2_loss(π_θ || π_teacher)
    
    where:
    - r_t = π_θ(o_i,t|q,o_i,<t) / π_θ_old(o_i,t|q,o_i,<t)
    - A_hat_i,t is the advantage (normalized reward within the group)
    - π_θ_old is the OLD student policy (NOT the teacher)
    - π_teacher is the teacher model for distillation
    - G is the number of samples per prompt (default=4)
    
    Args:
        model: The student model to train
        teacher_model: The teacher model for distillation (frozen)
        args: Training arguments (GRPOConfig)
        num_samples_per_prompt: Number of responses to sample per prompt (G)
        clip_epsilon: PPO clipping parameter (ε)
        beta: Weight for DistiLLM-v2 loss
        reward_model: Optional reward model for computing rewards
        tokenizer: Tokenizer for encoding/decoding
    """

    _tag_names = ["trl", "grpo", "distillm"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        teacher_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        # GRPO specific
        num_samples_per_prompt: int = 4,
        clip_epsilon: float = 0.2,
        beta: float = 0.1,  # Weight for DistiLLM-v2 loss
        # DistiLLM specific
        base_alpha_1: float = 0.1,
        base_alpha_2: float = 0.1,
        # Generation
        max_length: int = 512,
        max_prompt_length: int = 128,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        # PEFT
        peft_config: Optional[Dict] = None,
        model_init_kwargs: Optional[Dict] = None,
        teacher_model_init_kwargs: Optional[Dict] = None,
        # Reward (can be rule-based or model-based)
        reward_funcs: Optional[List[Callable]] = None,
        reward_weights: Optional[List[float]] = None,
        # Other
        disable_dropout: bool = True,
        label_pad_token_id: int = -100,
    ):
        # Handle model initialization
        if model_init_kwargs is None:
            model_init_kwargs = {}
        if teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
            
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        if isinstance(teacher_model, str):
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)

        self._peft_has_been_casted_to_bf16 = False

        # Handle PEFT
        if is_peft_available() and peft_config is not None:
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )
                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}
                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs
                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                self._peft_has_been_casted_to_bf16 = True

        elif getattr(args, "gradient_checkpointing", False):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # Store models
        self.teacher_model = teacher_model
        
        # Create old policy model (copy of student for π_θ_old)
        # This will be updated periodically or use EMA
        self.old_policy_model = create_reference_model(model)
        
        # Store tokenizer
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # GRPO hyperparameters
        self.num_samples_per_prompt = num_samples_per_prompt  # G
        self.clip_epsilon = clip_epsilon  # ε
        self.beta = beta  # Weight for distillation loss
        
        # DistiLLM hyperparameters
        self.base_alpha_1 = base_alpha_1
        self.base_alpha_2 = base_alpha_2
        self.logp_logq = None
        self.logq_logp = None
        
        # Generation parameters
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Other parameters
        self.label_pad_token_id = label_pad_token_id
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        
        # Reward functions (can be multiple)
        self.reward_funcs = reward_funcs if reward_funcs is not None else []
        self.reward_weights = reward_weights if reward_weights is not None else [1.0] * len(self.reward_funcs)
        
        # Data collator
        if data_collator is None:
            data_collator = GRPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=label_pad_token_id,
            )
        self.use_grpo_data_collator = True
        
        # Disable dropout
        if disable_dropout:
            disable_dropout_in_model(model)
            if self.teacher_model is not None:
                disable_dropout_in_model(self.teacher_model)
            if self.old_policy_model is not None:
                disable_dropout_in_model(self.old_policy_model)
        
        # Metrics storage
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        
        # Tokenize dataset (only prompts needed for GRPO)
        with PartialState().local_main_process_first():
            train_dataset = train_dataset.map(
                self.tokenize_prompt, 
                num_proc=getattr(args, 'dataset_num_proc', None),
                writer_batch_size=10,
                desc="Tokenizing prompts"
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    self.tokenize_prompt,
                    num_proc=getattr(args, 'dataset_num_proc', None),
                    writer_batch_size=10,
                    desc="Tokenizing eval prompts"
                )
        
        # Initialize parent Trainer
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        
        # Prepare models with accelerator
        if not hasattr(self, "accelerator"):
            raise AttributeError("Trainer does not have an accelerator object.")
        
        if self.teacher_model is not None:
            if self.is_deepspeed_enabled:
                self.teacher_model = self._prepare_deepspeed(self.teacher_model)
            else:
                self.teacher_model = self.accelerator.prepare_model(self.teacher_model, evaluation_mode=True)
        
        if self.old_policy_model is not None:
            if self.is_deepspeed_enabled:
                self.old_policy_model = self._prepare_deepspeed(self.old_policy_model)
            else:
                self.old_policy_model = self.accelerator.prepare_model(self.old_policy_model, evaluation_mode=True)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        """Prepare model for DeepSpeed."""
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None and hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                config_kwargs.update({
                    "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                    "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                    "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                })

        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def tokenize_prompt(self, example: Dict) -> Dict:
        """Tokenize a single prompt."""
        prompt = example["prompt"]
        
        # Handle different prompt formats
        if isinstance(prompt, list):
            # Chat format
            prompt_text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = prompt
        
        tokens = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=True,
        )
        
        return {
            "prompt_input_ids": tokens["input_ids"],
            "prompt_attention_mask": tokens["attention_mask"],
            "prompt": prompt_text,
        }

    @torch.no_grad()
    def generate_responses(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        num_samples: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate multiple responses per prompt using the current policy.
        
        Args:
            prompt_input_ids: [batch_size, seq_len]
            prompt_attention_mask: [batch_size, seq_len]
            num_samples: Number of samples per prompt (G)
            
        Returns:
            response_ids: [batch_size * num_samples, max_len]
            response_attention_mask: [batch_size * num_samples, max_len]
        """
        if num_samples is None:
            num_samples = self.num_samples_per_prompt
        
        batch_size = prompt_input_ids.shape[0]
        
        # Repeat prompts for multiple samples
        # [batch_size, seq_len] -> [batch_size * num_samples, seq_len]
        expanded_input_ids = prompt_input_ids.repeat_interleave(num_samples, dim=0)
        expanded_attention_mask = prompt_attention_mask.repeat_interleave(num_samples, dim=0)
        
        # Generate
        generate_context = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
        
        with generate_context():
            outputs = self.model.generate(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
        
        response_ids = outputs.sequences
        response_attention_mask = (response_ids != self.tokenizer.pad_token_id).long()
        
        return response_ids, response_attention_mask

    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute rewards for generated responses.
        Can use rule-based rewards, reward models, or a combination.
        
        Args:
            prompts: List of prompt strings
            responses: List of response strings
            prompt_ids: Prompt token ids
            response_ids: Full sequence (prompt + response) token ids
            
        Returns:
            rewards: [batch_size * num_samples] tensor of rewards
        """
        batch_size = len(prompts)
        device = response_ids.device
        
        if len(self.reward_funcs) == 0:
            # Default: use response length as a simple reward (placeholder)
            # In practice, you should implement proper reward functions
            rewards = torch.tensor(
                [len(r.split()) / 100.0 for r in responses],  # Normalized length
                device=device,
                dtype=torch.float32
            )
        else:
            # Combine multiple reward functions
            all_rewards = []
            for reward_func, weight in zip(self.reward_funcs, self.reward_weights):
                r = reward_func(prompts, responses, prompt_ids, response_ids)
                if not isinstance(r, torch.Tensor):
                    r = torch.tensor(r, device=device, dtype=torch.float32)
                all_rewards.append(weight * r)
            rewards = sum(all_rewards)
        
        return rewards

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Compute group-relative advantages (GRPO style).
        Normalize rewards within each group (same prompt).
        
        Args:
            rewards: [batch_size * num_samples] rewards
            num_samples: G (number of samples per prompt)
            
        Returns:
            advantages: [batch_size * num_samples] normalized advantages
        """
        # Reshape to [batch_size, num_samples]
        batch_size = rewards.shape[0] // num_samples
        rewards_grouped = rewards.view(batch_size, num_samples)
        
        # Normalize within group (subtract mean, divide by std)
        mean = rewards_grouped.mean(dim=1, keepdim=True)
        std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        advantages_grouped = (rewards_grouped - mean) / std
        
        # Flatten back
        advantages = advantages_grouped.view(-1)
        
        return advantages

    def get_per_token_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities.
        
        Args:
            model: The model to use
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] with -100 for tokens to ignore
            
        Returns:
            per_token_logps: [batch_size, seq_len-1]
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits
        
        # Shift for next-token prediction
        logits = logits[:, :-1, :]  # [batch, seq-1, vocab]
        labels = labels[:, 1:]  # [batch, seq-1]
        
        # Compute log probs
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for actual tokens
        per_token_logps = torch.gather(
            log_probs, 
            dim=-1, 
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask out padding
        mask = (labels != self.label_pad_token_id).float()
        per_token_logps = per_token_logps * mask
        
        return per_token_logps, mask

    def compute_distillm_v2_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DistiLLM-v2 loss with adaptive alpha mixture KL divergence.
        
        Args:
            student_logits: [batch_size, seq_len, vocab_size]
            teacher_logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]
            mask: [batch_size, seq_len] loss mask
            
        Returns:
            distillm_loss: Scalar loss
        """
        # Shift for next-token prediction
        student_logits = student_logits[:, :-1, :]
        teacher_logits = teacher_logits[:, :-1, :]
        labels = labels[:, 1:]
        mask = mask[:, 1:] if mask.shape[1] > 1 else mask
        
        # Log softmax
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        
        # Get per-token log probs for labels
        student_token_logps = torch.gather(student_log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        teacher_token_logps = torch.gather(teacher_log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        
        # Adaptive alpha_1 for forward KL (teacher -> mixture)
        alpha_1 = self.base_alpha_1
        try:
            if self.logp_logq is not None:
                anchor = (1 - self.base_alpha_1) * self.logp_logq
                logps_logqs = ((teacher_token_logps * mask).sum(-1) / (mask.sum(-1) + 1e-8)).exp() - \
                              ((student_token_logps * mask).sum(-1) / (mask.sum(-1) + 1e-8)).exp()
                alpha_1 = torch.clip(
                    1 - anchor / (logps_logqs + 1e-5), 
                    min=1e-2, 
                    max=self.base_alpha_1
                ).unsqueeze(-1).unsqueeze(-1)
        except:
            alpha_1 = self.base_alpha_1
        
        # Compute forward KL: KL(teacher || mixture)
        if isinstance(alpha_1, torch.Tensor):
            log_alpha_1 = torch.log(alpha_1)
            log_one_minus_alpha_1 = torch.log(1 - alpha_1)
        else:
            log_alpha_1 = math.log(alpha_1)
            log_one_minus_alpha_1 = math.log(1 - alpha_1)
        
        mix_log_probs = torch.logsumexp(
            torch.stack([
                log_alpha_1 + teacher_log_probs,
                log_one_minus_alpha_1 + student_log_probs
            ], dim=0), dim=0
        )
        forward_kl = (teacher_log_probs.exp() * (teacher_log_probs - mix_log_probs)).sum(-1)
        
        # Adaptive alpha_2 for reverse KL (student -> mixture)
        alpha_2 = self.base_alpha_2
        try:
            if self.logq_logp is not None:
                anchor = (1 - self.base_alpha_2) * self.logq_logp
                logqs_logps = ((student_token_logps * mask).sum(-1) / (mask.sum(-1) + 1e-8)).exp() - \
                              ((teacher_token_logps * mask).sum(-1) / (mask.sum(-1) + 1e-8)).exp()
                alpha_2 = torch.clip(
                    1 - anchor / (logqs_logps + 1e-5),
                    min=1e-2,
                    max=self.base_alpha_2
                ).unsqueeze(-1).unsqueeze(-1)
        except:
            alpha_2 = self.base_alpha_2
        
        # Compute reverse KL: KL(student || mixture)
        if isinstance(alpha_2, torch.Tensor):
            log_alpha_2 = torch.log(alpha_2)
            log_one_minus_alpha_2 = torch.log(1 - alpha_2)
        else:
            log_alpha_2 = math.log(alpha_2)
            log_one_minus_alpha_2 = math.log(1 - alpha_2)
        
        mix_log_probs_2 = torch.logsumexp(
            torch.stack([
                log_one_minus_alpha_2 + teacher_log_probs,
                log_alpha_2 + student_log_probs.detach()
            ], dim=0), dim=0
        )
        reverse_kl = (student_log_probs.exp() * (student_log_probs - mix_log_probs_2)).sum(-1)
        
        # Combine: forward KL for chosen (teacher-guided), reverse KL for diversity
        # Following DistiLLM-v2 weighting
        distillm_loss = (forward_kl * mask).sum(-1) / (mask.sum(-1) + 1e-8) + \
                        (reverse_kl * mask).sum(-1) / (mask.sum(-1) + 1e-8)
        
        return distillm_loss.mean()

    def compute_grpo_loss(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the full GRPO + DistiLLM-v2 loss.
        
        Steps:
        1. Generate G responses per prompt using current policy
        2. Compute rewards and group-relative advantages
        3. Compute policy ratio π_θ / π_θ_old
        4. Compute clipped PPO objective
        5. Compute DistiLLM-v2 distillation loss
        6. Combine losses
        """
        metrics = {}
        
        prompt_input_ids = batch["prompt_input_ids"]
        prompt_attention_mask = batch["prompt_attention_mask"]
        prompts = batch["prompt"]
        
        batch_size = prompt_input_ids.shape[0]
        num_samples = self.num_samples_per_prompt
        
        # Step 1: Generate responses
        with torch.no_grad():
            response_ids, response_attention_mask = self.generate_responses(
                prompt_input_ids, prompt_attention_mask, num_samples
            )
        
        # Decode responses
        prompt_lengths = prompt_attention_mask.sum(dim=1).repeat_interleave(num_samples)
        responses = []
        for i, (resp_ids, prompt_len) in enumerate(zip(response_ids, prompt_lengths)):
            response_text = self.tokenizer.decode(
                resp_ids[prompt_len:], 
                skip_special_tokens=True
            )
            responses.append(response_text)
        
        # Expand prompts for reward computation
        expanded_prompts = []
        for p in prompts:
            expanded_prompts.extend([p] * num_samples)
        
        # Step 2: Compute rewards and advantages
        rewards = self.compute_rewards(
            expanded_prompts, 
            responses,
            prompt_input_ids.repeat_interleave(num_samples, dim=0),
            response_ids
        )
        advantages = self.compute_advantages(rewards, num_samples)
        
        # Create labels (mask prompt tokens)
        labels = response_ids.clone()
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = self.label_pad_token_id
        
        # Step 3: Compute log probs for current policy
        current_logps, mask = self.get_per_token_logps(
            self.model, 
            response_ids, 
            response_attention_mask,
            labels
        )
        
        # Compute log probs for old policy (π_θ_old)
        with torch.no_grad():
            old_logps, _ = self.get_per_token_logps(
                self.old_policy_model,
                response_ids,
                response_attention_mask,
                labels
            )
        
        # Step 4: Compute policy ratio and clipped objective
        # Sum log probs per sequence (or average)
        current_seq_logps = (current_logps * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        old_seq_logps = (old_logps * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # Ratio: exp(log π_θ - log π_θ_old) = π_θ / π_θ_old
        log_ratio = current_seq_logps - old_seq_logps
        ratio = torch.exp(log_ratio)
        
        # Clipped ratio
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        # PPO objective (maximize, so negate for loss)
        # obj = min(ratio * A, clip(ratio) * A)
        obj1 = ratio * advantages
        obj2 = clipped_ratio * advantages
        ppo_objective = torch.min(obj1, obj2)
        ppo_loss = -ppo_objective.mean()
        
        # Step 5: Compute DistiLLM-v2 loss
        # Get logits from current model and teacher
        student_outputs = self.model(
            input_ids=response_ids,
            attention_mask=response_attention_mask,
            use_cache=False,
        )
        student_logits = student_outputs.logits
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=response_ids,
                attention_mask=response_attention_mask,
                use_cache=False,
            )
            teacher_logits = teacher_outputs.logits
        
        distillm_loss = self.compute_distillm_v2_loss(
            student_logits, teacher_logits, labels, mask
        )
        
        # Step 6: Combine losses
        # J_GRPO = PPO_obj - β * DistiLLM_loss
        # As loss: L = -PPO_obj + β * DistiLLM_loss = ppo_loss + β * distillm_loss
        total_loss = ppo_loss + self.beta * distillm_loss
        
        # Log metrics
        metrics["grpo/ppo_loss"] = ppo_loss.detach().cpu().item()
        metrics["grpo/distillm_loss"] = distillm_loss.detach().cpu().item()
        metrics["grpo/total_loss"] = total_loss.detach().cpu().item()
        metrics["grpo/mean_reward"] = rewards.mean().cpu().item()
        metrics["grpo/mean_advantage"] = advantages.mean().cpu().item()
        metrics["grpo/mean_ratio"] = ratio.mean().cpu().item()
        metrics["grpo/clip_fraction"] = ((ratio - 1).abs() > self.clip_epsilon).float().mean().cpu().item()
        
        return total_loss, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Main loss computation called by Trainer."""
        
        compute_loss_context = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
        
        with compute_loss_context():
            loss, metrics = self.compute_grpo_loss(model, inputs)
        
        loss = loss.to(self.args.device)
        self.store_metrics(metrics, train_eval="train")
        
        if return_outputs:
            return (loss, metrics)
        return loss

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        """Store metrics for logging."""
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        """Log metrics including stored ones."""
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    def update_old_policy(self):
        """
        Update the old policy model to match current policy.
        Should be called periodically (e.g., every N steps or epochs).
        """
        # Copy parameters from current model to old policy
        self.old_policy_model.load_state_dict(
            self.accelerator.unwrap_model(self.model).state_dict()
        )

    def training_step(self, model, inputs):
        """Override to update old policy periodically."""
        loss = super().training_step(model, inputs)
        
        # Update old policy every N steps (configurable)
        update_frequency = getattr(self.args, 'old_policy_update_frequency', 100)
        if self.state.global_step % update_frequency == 0 and self.state.global_step > 0:
            self.update_old_policy()
        
        return loss

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)
        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)