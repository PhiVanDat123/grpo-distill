# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import dataclasses
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser, TrainingArguments

import trl


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DataClassType = NewType("DataClassType", Any)


class H4ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys

                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type is bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> DataClassType | Tuple[DataClassType]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    base_model_revision: Optional[str] = field(
        default=None,
        metadata={"help": ("The base model checkpoint for weights initialization with PEFT adapters.")},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    ref_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The reference model checkpoint for weights initialization. Don't set if you want to train a model "
                "from scratch."
            )
        }
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    model_code_revision: str = field(default=None, metadata={"help": "The branch of the IFT model"})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The path to the tokenizer. Useful if you want to use a different tokenizer to the one stored in `model_name_or_path`."
            )
        },
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code when loading a model."})
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use; you can use --attn_implementation=flash_attention_2, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "use 8 bit precision"})
    load_in_4bit: bool = field(default=False, metadata={"help": "use 4 bit precision"})

    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "use nested quantization"})
    bnb_4bit_quant_storage: Optional[str] = field(
        default="uint8",
        metadata={"help": "storage type to pack the quanitzed 4-bit prarams."},
    )

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    dataset_mixer: Optional[Dict[str, float]] = field(
        default=None,
        metadata={"help": ("Datasets and their proportions to be used for training ift/rl.")},
    )
    text_column: Optional[str] = field(
        default="text",
        metadata={"help": "The column name to use for the text in the dataset (only used for continued pretraining)."},
    )
    dataset_splits: Optional[List[str]] = field(
        default_factory=lambda: ["train", "test"],
        metadata={"help": ("List of train test splits to use in the dataset")},
    )
    dataset_configs: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of dataset config names. If given must be the same length as 'dataset_mixer' keys."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )
    auto_insert_empty_system_msg: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to automatically insert an empty system message as the first message if `system` is mentioned in the chat template."
            )
        },
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    Arguments related to the training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/trainer#transformers.TrainingArguments
    Also used for the continued pretraining task.
    """

    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )


@dataclass
class DPOConfig(trl.DPOConfig):
    """
    Arguments related to the DPO training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/trainer#transformers.TrainingArguments
    """

    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )
    optim: Optional[str] = field(default="rmsprop")
    remove_unused_columns: bool = field(default=False)


@dataclass
class GRPOConfig(TrainingArguments):
    """
    Configuration for GRPO (Group Relative Policy Optimization) + DistiLLM-v2 training.
    
    This config combines:
    - GRPO: PPO-style policy optimization with group-relative advantages
    - DistiLLM-v2: Adaptive mixture KL divergence for knowledge distillation
    
    Key differences from standard GRPO:
    - KL term is replaced with DistiLLM-v2 loss
    - Teacher model is for distillation, NOT for GRPO baseline
    - π_θ_old is the student model from previous timestep
    """
    
    # GRPO hyperparameters
    num_samples_per_prompt: int = field(
        default=4,
        metadata={
            "help": "Number of responses to sample per prompt (G in GRPO). "
                    "Higher values give better advantage estimates but increase computation."
        },
    )
    clip_epsilon: float = field(
        default=0.2,
        metadata={
            "help": "PPO clipping parameter (ε). Clips the policy ratio to [1-ε, 1+ε]. "
                    "Typical values: 0.1-0.3"
        },
    )
    beta: float = field(
        default=0.1,
        metadata={
            "help": "Weight for DistiLLM-v2 distillation loss. "
                    "Higher values put more emphasis on matching the teacher."
        },
    )
    old_policy_update_frequency: int = field(
        default=100,
        metadata={
            "help": "How often to update the old policy model (in training steps). "
                    "Set to 1 for on-policy, higher for more stability."
        },
    )
    
    # DistiLLM-v2 hyperparameters
    base_alpha_1: float = field(
        default=0.1,
        metadata={
            "help": "Base alpha for forward KL in DistiLLM-v2. "
                    "Controls mixture weight: α*teacher + (1-α)*student"
        },
    )
    base_alpha_2: float = field(
        default=0.1,
        metadata={
            "help": "Base alpha for reverse KL in DistiLLM-v2. "
                    "Controls mixture weight for student-to-mixture divergence."
        },
    )
    
    # Generation parameters
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum total sequence length (prompt + response)."},
    )
    max_prompt_length: int = field(
        default=128,
        metadata={"help": "Maximum prompt length. Prompts longer than this will be truncated."},
    )
    max_new_tokens: int = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to generate per response."},
    )
    temperature: float = field(
        default=0.7,
        metadata={
            "help": "Sampling temperature for response generation. "
                    "Higher values = more random, lower = more deterministic."
        },
    )
    top_p: float = field(
        default=0.9,
        metadata={
            "help": "Nucleus sampling probability. "
                    "Only tokens with cumulative probability < top_p are considered."
        },
    )
    
    # Dataset processing
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes for dataset preprocessing."},
    )
    
    # Training settings
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in all models during training."},
    )
    label_pad_token_id: int = field(
        default=-100,
        metadata={"help": "Token ID used for padding labels (ignored in loss computation)."},
    )
    
    # Hub settings
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    
    # Logging
    logging_first_step: bool = field(
        default=True,
        metadata={"help": "Whether to log and evaluate the first global_step or not."},
    )
    
    # Optimizer (default to AdamW for RL)
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "Optimizer to use. AdamW is recommended for RL."},
    )
    
    # Remove unused columns (must be False for custom data collator)
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Must be False for GRPO trainer."},
    )
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate GRPO parameters
        if self.num_samples_per_prompt < 2:
            raise ValueError(
                f"num_samples_per_prompt must be >= 2 for group-relative advantages, "
                f"got {self.num_samples_per_prompt}"
            )
        
        if not 0 < self.clip_epsilon < 1:
            raise ValueError(
                f"clip_epsilon should be in (0, 1), got {self.clip_epsilon}"
            )
        
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        
        if not 0 < self.base_alpha_1 < 1:
            raise ValueError(
                f"base_alpha_1 should be in (0, 1), got {self.base_alpha_1}"
            )
        
        if not 0 < self.base_alpha_2 < 1:
            raise ValueError(
                f"base_alpha_2 should be in (0, 1), got {self.base_alpha_2}"
            )
        
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p should be in (0, 1], got {self.top_p}")