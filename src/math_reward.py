"""
Math Reward Functions for GRPO + DistiLLM Training.

Compatible with GRPODistiLLMTrainer's reward function signature:
    reward_func(prompts, responses, prompt_ids, response_ids) -> List[float]

Focus: Only evaluate the final answer, not the reasoning process.
"""

import re
import math
from typing import List, Optional
import torch

# Optional imports for math verification
try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("Warning: math_verify not available. Using simple number matching.")


# =============================================================================
# Answer Extraction Utilities
# =============================================================================

def extract_answer_from_model_output(text: str) -> Optional[str]:
    """
    Extract the value from the last <answer> tag in the text.
    
    Args:
        text: Model-generated text containing <answer>...</answer> tags.
        
    Returns:
        Content inside the <answer> tags, or None if not found.
    """
    parts = text.split("<answer>")
    if len(parts) < 2:
        return None
    
    last_part = parts[-1]
    if "</answer>" not in last_part:
        return None
    
    answer = last_part.split("</answer>")[0].strip().replace(',', '')
    return None if answer == "..." or answer == "" else answer


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \\boxed{...} format.
    
    Args:
        text: Text containing \\boxed{answer}.
        
    Returns:
        Content inside \\boxed{}, or None if not found.
    """
    # Handle nested braces
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def extract_last_number(text: str) -> Optional[float]:
    """
    Extract the last number appearing in the text.
    
    Args:
        text: Text to extract number from.
        
    Returns:
        The last number as float, or None if not found.
    """
    if text is None:
        return None
    
    # Clean up common symbols
    text = text.replace('$', '').replace('%', '').replace(',', '')
    
    # Find all numbers (including negative and decimals)
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    
    return float(matches[-1]) if matches else None


def extract_answer_flexible(text: str) -> Optional[str]:
    """
    Try multiple methods to extract the final answer.
    
    Priority:
    1. <answer>...</answer> tags
    2. \\boxed{...}
    3. Last number in text
    
    Args:
        text: Model output text.
        
    Returns:
        Extracted answer string, or None if not found.
    """
    # Try <answer> tags first
    answer = extract_answer_from_model_output(text)
    if answer is not None:
        return answer
    
    # Try \\boxed{}
    answer = extract_boxed_answer(text)
    if answer is not None:
        return answer
    
    # Fallback: extract last number
    num = extract_last_number(text)
    if num is not None:
        return str(num)
    
    return None


def extract_ground_truth(solution: str) -> Optional[str]:
    """
    Extract ground truth from dataset solution format.
    
    Handles:
    - GSM8K format: "... #### answer"
    - Direct number
    - LaTeX expressions
    
    Args:
        solution: Ground truth solution string.
        
    Returns:
        Extracted answer string.
    """
    # GSM8K format
    if "####" in solution:
        return solution.split("####")[1].strip().replace(',', '')
    
    # Already a clean answer
    return solution.strip()


# =============================================================================
# Core Reward Functions (Compatible with GRPODistiLLMTrainer)
# =============================================================================

def accuracy_reward(
    prompts: List[str],
    responses: List[str],
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    solutions: List[str] = None,
    **kwargs
) -> List[float]:
    """
    Binary accuracy reward based on final answer correctness.
    
    Args:
        prompts: List of prompt strings
        responses: List of response strings
        prompt_ids: Prompt token ids (unused)
        response_ids: Response token ids (unused)
        solutions: List of ground truth solutions
        
    Returns:
        List of rewards: 1.0 for correct, 0.0 for incorrect
    """
    if solutions is None:
        raise ValueError("accuracy_reward requires 'solutions' in kwargs")
    
    rewards = []
    
    for response, solution in zip(responses, solutions):
        # Extract model's answer
        model_answer = extract_answer_flexible(response)
        if model_answer is None:
            rewards.append(0.0)
            continue
        
        # Extract ground truth
        gold_answer = extract_ground_truth(solution)
        
        if MATH_VERIFY_AVAILABLE:
            # Use math_verify for robust comparison
            try:
                gold_parsed = parse(
                    gold_answer,
                    extraction_mode="first_match",
                    extraction_config=[LatexExtractionConfig()],
                )
                
                if len(gold_parsed) == 0:
                    # Fallback to string comparison
                    reward = 1.0 if model_answer.strip() == gold_answer.strip() else 0.0
                else:
                    answer_parsed = parse(
                        model_answer,
                        extraction_mode="first_match",
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    equations=True,
                                    boxed=True,
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                    )
                    reward = 1.0 if verify(answer_parsed, gold_parsed) else 0.0
            except Exception:
                reward = 0.0
        else:
            # Simple number comparison fallback
            try:
                model_num = extract_last_number(model_answer)
                gold_num = extract_last_number(gold_answer)
                
                if model_num is not None and gold_num is not None:
                    # Allow small floating point tolerance
                    reward = 1.0 if abs(model_num - gold_num) < 1e-6 else 0.0
                else:
                    # String comparison
                    reward = 1.0 if model_answer.strip() == gold_answer.strip() else 0.0
            except Exception:
                reward = 0.0
        
        rewards.append(reward)
    
    return rewards


def scaled_accuracy_reward(
    prompts: List[str],
    responses: List[str],
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    solutions: List[str] = None,
    correct_reward: float = 2.0,
    incorrect_reward: float = 0.0,
    **kwargs
) -> List[float]:
    """
    Scaled accuracy reward with configurable reward values.
    
    Args:
        prompts: List of prompt strings
        responses: List of response strings
        prompt_ids: Prompt token ids (unused)
        response_ids: Response token ids (unused)
        solutions: List of ground truth solutions
        correct_reward: Reward for correct answer (default: 2.0)
        incorrect_reward: Reward for incorrect answer (default: 0.0)
        
    Returns:
        List of scaled rewards
    """
    base_rewards = accuracy_reward(
        prompts, responses, prompt_ids, response_ids, 
        solutions=solutions, **kwargs
    )
    
    return [
        correct_reward if r > 0.5 else incorrect_reward 
        for r in base_rewards
    ]


def format_reward(
    prompts: List[str],
    responses: List[str],
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    **kwargs
) -> List[float]:
    """
    Reward for correct <think>...</think><answer>...</answer> format.
    
    Args:
        prompts: List of prompt strings
        responses: List of response strings
        prompt_ids: Prompt token ids (unused)
        response_ids: Response token ids (unused)
        
    Returns:
        List of rewards: 1.0 for correct format, 0.0 otherwise
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    
    rewards = []
    for response in responses:
        match = re.match(pattern, response, re.DOTALL | re.MULTILINE)
        rewards.append(1.0 if match else 0.0)
    
    return rewards


def tag_count_reward(
    prompts: List[str],
    responses: List[str],
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    **kwargs
) -> List[float]:
    """
    Partial reward for having correct number of think/answer tags.
    
    Args:
        prompts: List of prompt strings
        responses: List of response strings
        prompt_ids: Prompt token ids (unused)
        response_ids: Response token ids (unused)
        
    Returns:
        List of rewards: 0.0 to 1.0 based on tag correctness
    """
    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>") == 1:
            count += 0.25
        if text.count("</think>") == 1:
            count += 0.25
        if text.count("<answer>") == 1:
            count += 0.25
        if text.count("</answer>") == 1:
            count += 0.25
        return count
    
    return [count_tags(r) for r in responses]


def length_penalty_reward(
    prompts: List[str],
    responses: List[str],
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    solutions: List[str] = None,
    max_len: int = 2048,
    **kwargs
) -> List[float]:
    """
    Length-based reward following Kimi 1.5 paper.
    Rewards shorter correct answers, penalizes longer incorrect ones.
    
    Args:
        prompts: List of prompt strings
        responses: List of response strings
        prompt_ids: Prompt token ids (unused)
        response_ids: Response token ids (unused)
        solutions: List of ground truth solutions
        max_len: Maximum expected length for scaling
        
    Returns:
        List of length-adjusted rewards
    """
    if solutions is None:
        # Without solutions, just use length penalty
        lengths = [len(r) for r in responses]
        min_len = min(lengths) if lengths else 1
        max_actual = max(lengths) if lengths else 1
        
        if max_actual == min_len:
            return [0.0] * len(responses)
        
        return [
            0.5 - (length - min_len) / (max_actual - min_len)
            for length in lengths
        ]
    
    # Get base accuracy
    correctness = accuracy_reward(
        prompts, responses, prompt_ids, response_ids,
        solutions=solutions, **kwargs
    )
    
    lengths = [len(r) for r in responses]
    min_len = min(lengths) if lengths else 1
    max_actual = max(lengths) if lengths else 1
    
    if max_actual == min_len:
        return [0.0] * len(responses)
    
    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_actual - min_len)
        
        if is_correct > 0.5:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)
        
        rewards.append(reward)
    
    return rewards


def cosine_scaled_reward(
    prompts: List[str],
    responses: List[str],
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    solutions: List[str] = None,
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 2048,
    **kwargs
) -> List[float]:
    """
    Cosine-scaled reward based on length and correctness.
    Shorter correct solutions get higher rewards.
    
    Args:
        prompts: List of prompt strings
        responses: List of response strings
        prompt_ids: Prompt token ids (unused)
        response_ids: Response token ids (unused)
        solutions: List of ground truth solutions
        min_value_wrong: Min reward for wrong answers
        max_value_wrong: Max reward for wrong answers
        min_value_correct: Min reward for correct answers
        max_value_correct: Max reward for correct answers
        max_len: Maximum length for scaling
        
    Returns:
        List of cosine-scaled rewards
    """
    if solutions is None:
        raise ValueError("cosine_scaled_reward requires 'solutions' in kwargs")
    
    correctness = accuracy_reward(
        prompts, responses, prompt_ids, response_ids,
        solutions=solutions, **kwargs
    )
    
    rewards = []
    for response, is_correct in zip(responses, correctness):
        gen_len = len(response)
        progress = min(gen_len / max_len, 1.0)
        cosine = math.cos(progress * math.pi)
        
        if is_correct > 0.5:
            min_val, max_val = min_value_correct, max_value_correct
        else:
            min_val, max_val = max_value_wrong, min_value_wrong
        
        reward = min_val + 0.5 * (max_val - min_val) * (1.0 + cosine)
        rewards.append(reward)
    
    return rewards


def repetition_penalty_reward(
    prompts: List[str],
    responses: List[str],
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    ngram_size: int = 4,
    max_penalty: float = -0.5,
    **kwargs
) -> List[float]:
    """
    Penalize repetitive n-grams in the response.
    
    Args:
        prompts: List of prompt strings
        responses: List of response strings
        prompt_ids: Prompt token ids (unused)
        response_ids: Response token ids (unused)
        ngram_size: Size of n-grams to check
        max_penalty: Maximum penalty (should be negative)
        
    Returns:
        List of penalty rewards (0 to max_penalty)
    """
    def get_ngrams(text: str, n: int):
        words = text.lower().split()
        return list(zip(*[words[i:] for i in range(n)]))
    
    rewards = []
    for response in responses:
        if not response or len(response.split()) < ngram_size:
            rewards.append(0.0)
            continue
        
        ngrams = get_ngrams(response, ngram_size)
        if not ngrams:
            rewards.append(0.0)
            continue
        
        unique_ngrams = set(ngrams)
        repetition_ratio = 1 - len(unique_ngrams) / len(ngrams)
        
        reward = repetition_ratio * max_penalty
        rewards.append(reward)
    
    return rewards


# =============================================================================
# Combined Reward Function
# =============================================================================

def combined_math_reward(
    prompts: List[str],
    responses: List[str],
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    solutions: List[str] = None,
    accuracy_weight: float = 1.0,
    format_weight: float = 0.1,
    length_weight: float = 0.1,
    repetition_weight: float = 0.1,
    **kwargs
) -> List[float]:
    """
    Combined reward function for math tasks.
    
    Components:
    - Accuracy (main): Correct answer gets high reward
    - Format (auxiliary): Proper <think>/<answer> tags
    - Length (auxiliary): Prefer concise solutions
    - Repetition (auxiliary): Penalize repetitive text
    
    Args:
        prompts: List of prompt strings
        responses: List of response strings
        prompt_ids: Prompt token ids
        response_ids: Response token ids
        solutions: List of ground truth solutions
        accuracy_weight: Weight for accuracy reward
        format_weight: Weight for format reward
        length_weight: Weight for length reward
        repetition_weight: Weight for repetition penalty
        
    Returns:
        List of combined rewards
    """
    # Accuracy (main reward)
    acc_rewards = scaled_accuracy_reward(
        prompts, responses, prompt_ids, response_ids,
        solutions=solutions, correct_reward=1.0, incorrect_reward=0.0,
        **kwargs
    )
    
    # Format reward
    fmt_rewards = tag_count_reward(
        prompts, responses, prompt_ids, response_ids, **kwargs
    )
    
    # Length penalty (only if solutions provided)
    if solutions is not None:
        len_rewards = length_penalty_reward(
            prompts, responses, prompt_ids, response_ids,
            solutions=solutions, **kwargs
        )
    else:
        len_rewards = [0.0] * len(responses)
    
    # Repetition penalty
    rep_rewards = repetition_penalty_reward(
        prompts, responses, prompt_ids, response_ids, **kwargs
    )
    
    # Combine
    combined = []
    for acc, fmt, lng, rep in zip(acc_rewards, fmt_rewards, len_rewards, rep_rewards):
        total = (
            accuracy_weight * acc +
            format_weight * fmt +
            length_weight * lng +
            repetition_weight * rep
        )
        combined.append(total)
    
    return combined


# =============================================================================
# Factory Functions for Creating Reward Functions with Solutions
# =============================================================================

def create_accuracy_reward_with_solutions(solutions: List[str]):
    """
    Create an accuracy reward function with pre-bound solutions.
    
    Usage:
        solutions = dataset["solution"]
        reward_func = create_accuracy_reward_with_solutions(solutions)
        trainer = GRPODistiLLMTrainer(reward_funcs=[reward_func], ...)
    """
    def reward_fn(prompts, responses, prompt_ids, response_ids):
        # Map prompts to solutions (assumes same order in batch)
        batch_solutions = solutions[:len(responses)]
        return accuracy_reward(
            prompts, responses, prompt_ids, response_ids,
            solutions=batch_solutions
        )
    return reward_fn


def create_combined_reward_with_solutions(
    solutions: List[str],
    accuracy_weight: float = 1.0,
    format_weight: float = 0.1,
    length_weight: float = 0.1,
    repetition_weight: float = 0.1,
):
    """
    Create a combined reward function with pre-bound solutions and weights.
    """
    def reward_fn(prompts, responses, prompt_ids, response_ids):
        batch_solutions = solutions[:len(responses)]
        return combined_math_reward(
            prompts, responses, prompt_ids, response_ids,
            solutions=batch_solutions,
            accuracy_weight=accuracy_weight,
            format_weight=format_weight,
            length_weight=length_weight,
            repetition_weight=repetition_weight,
        )
    return reward_fn


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Test the reward functions
    test_prompts = [
        "What is 2 + 2?",
        "What is 3 * 4?",
    ]
    
    test_responses = [
        "<think>Let me calculate: 2 + 2 = 4</think>\n<answer>4</answer>",
        "<think>I need to multiply 3 by 4, which gives 12</think>\n<answer>12</answer>",
    ]
    
    test_solutions = ["4", "12"]
    
    # Create dummy tensors
    dummy_prompt_ids = torch.zeros(2, 10)
    dummy_response_ids = torch.zeros(2, 50)
    
    # Test accuracy reward
    acc = accuracy_reward(
        test_prompts, test_responses, 
        dummy_prompt_ids, dummy_response_ids,
        solutions=test_solutions
    )
    print(f"Accuracy rewards: {acc}")
    
    # Test format reward
    fmt = format_reward(
        test_prompts, test_responses,
        dummy_prompt_ids, dummy_response_ids
    )
    print(f"Format rewards: {fmt}")
    
    # Test combined reward
    combined = combined_math_reward(
        test_prompts, test_responses,
        dummy_prompt_ids, dummy_response_ids,
        solutions=test_solutions
    )
    print(f"Combined rewards: {combined}")