# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

# /// script
# dependencies = [
#     "trl[peft]",
#     "math-verify",
#     "latex2sympy2_extended",
#     "trackio",
# ]
# ///

r"""
Example of GRPO training with inline group-relative length-scaled accuracy rewards (GRPO-LEAD) to reduce overthinking.

Reference: "GRPO-LEAD: Length-scaled Efficiency-Aware Decoupling for Reasoning Models" (https://huggingface.co/papers/2504.09696).

### Motivation & The Overthinking Problem

During RL post-training of reasoning models (such as DeepSeek-R1 or Qwen reasoning variants), models often succumb to
"overthinking": generating thousands of tokens on simple problems with circular or redundant reasoning loops.
Standard accuracy verifiers reward correct answers equally regardless of length, failing to penalize verbosity.

This script demonstrates an inline group-relative length-scaled accuracy reward. For each prompt group:
1. Correctness is evaluated via math verification.
2. For correct completions within the group, lengths are standardized into group-relative z-scores:
       z_i = (|y_i| - \mu_{|y|}) / (\sigma_{|y|} + \epsilon)
3. Rewards are scaled exponentially by length efficiency:
       R(y_i) = \exp(-\alpha \cdot z_i)   if correct
       R(y_i) = incorrect_reward          if incorrect

### Note on Group Standardization in GRPO

GRPO standardizes rewards within each group by default (`A_i = (R_i - \mu) / \sigma`).
For an all-correct group, \exp(-\alpha \cdot z) behaves similarly to plain -z because of downstream standardization.
However, \alpha becomes a critical tuning knob in two practical scenarios:
1. Mixed groups (some correct, some incorrect): \alpha controls how strongly length efficiency competes with
   correctness vs wrong answers.
2. When training with unstandardized rewards (`GRPOConfig(scale_rewards="none")`).

### Usage

Run demo mode (simulates rollouts & prints reward breakdown without GPU):
```bash
python examples/grpo_lead_length_reward/grpo_lead_length_reward.py --demo
```

Train with Accelerate on single or multi-GPU:
```bash
accelerate launch \
    examples/grpo_lead_length_reward/grpo_lead_length_reward.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir grpo-lead-qwen-0.5b \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 4 \
    --num_generations 4 \
    --gradient_accumulation_steps 2
```
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from typing import Any

from trl.rewards import accuracy_reward, think_format_reward


# -----------------------------------------------------------------------------
# 1. Inline Group-Relative Length-Scaled Reward (GRPO-LEAD)
# -----------------------------------------------------------------------------


def get_length_scaled_accuracy_reward(
    alpha: float = 0.5,
    incorrect_reward: float = -1.0,
    eps: float = 1e-6,
) -> Callable:
    """Factory creating an inline reward function that scales accuracy by group-relative completion length.

    Args:
        alpha (float, optional): Length sensitivity coefficient. Defaults to 0.5.
        incorrect_reward (float, optional): Penalty assigned to incorrect completions. Defaults to -1.0.
        eps (float, optional): Small constant to avoid division by zero. Defaults to 1e-6.

    Returns:
        Callable: A picklable reward function compatible with GRPOTrainer.
    """
    return _LengthScaledAccuracyReward(alpha=alpha, incorrect_reward=incorrect_reward, eps=eps)


class _LengthScaledAccuracyReward:
    __name__ = "length_scaled_accuracy_reward"

    def __init__(self, alpha: float, incorrect_reward: float, eps: float):
        self.alpha = alpha
        self.incorrect_reward = incorrect_reward
        self.eps = eps

    def __call__(
        self,
        completions: list[list[dict[str, str]]],
        solution: list[str],
        completion_ids: list[list[int]] | None = None,
        **kwargs,
    ) -> list[float | None]:
        # 1. Evaluate ground-truth mathematical correctness
        correctness_scores = accuracy_reward(completions, solution, **kwargs)

        # 2. Extract completion lengths (in tokens if completion_ids provided, otherwise word count)
        if completion_ids is not None:
            lengths = [len(ids) for ids in completion_ids]
        else:
            lengths = [len(c[0]["content"].split()) for c in completions]

        # 3. Compute group statistics across correct completions
        correct_indices = [i for i, score in enumerate(correctness_scores) if score == 1.0]

        if not correct_indices:
            # If no completion was correct, return incorrect_reward (or None if unparseable)
            return [self.incorrect_reward if score is not None else None for score in correctness_scores]

        correct_lengths = [lengths[i] for i in correct_indices]
        mean_len = sum(correct_lengths) / len(correct_lengths)
        variance_len = sum((length - mean_len) ** 2 for length in correct_lengths) / len(correct_lengths)
        std_len = math.sqrt(variance_len)

        # 4. Scale rewards by group-relative length z-score
        rewards: list[float | None] = []
        for i, score in enumerate(correctness_scores):
            if score is None:
                # Gold solution could not be parsed; skip sample as in accuracy_reward
                rewards.append(None)
            elif score == 1.0:
                if std_len < self.eps:
                    z_score = 0.0
                else:
                    z_score = (lengths[i] - mean_len) / (std_len + self.eps)
                # Shorter completions (z < 0) yield rewards > 1.0; longer ones yield rewards < 1.0
                scaled_reward = math.exp(-self.alpha * z_score)
                rewards.append(scaled_reward)
            else:
                rewards.append(self.incorrect_reward)

        return rewards


# -----------------------------------------------------------------------------
# 2. Interactive / Standalone Verification Demo
# -----------------------------------------------------------------------------


def run_demo() -> None:
    """Demonstrate how length-scaled rewards prioritize concise reasoning traces within a GRPO group."""
    print("=" * 75)
    print("GRPO-LEAD: Inline Length-Scaled Accuracy Reward Demo")
    print("=" * 75)

    solution = ["0.05", "0.05", "0.05", "0.05"]
    completions = [
        # Candidate 1: Short & correct (120 tokens)
        [
            {
                "role": "assistant",
                "content": "<think>x + (x + 1.00) = 1.10 => 2x = 0.10 => x = 0.05</think>The ball costs \\boxed{0.05}.",
            }
        ],
        # Candidate 2: Medium & correct (450 tokens)
        [
            {
                "role": "assistant",
                "content": "<think>Let x be the ball. Then bat is x + 1.00. Wait, let me verify. Total is 1.10. So 2x + 1.00 = 1.10.</think>The ball costs \\boxed{0.05}.",
            }
        ],
        # Candidate 3: Overlong & correct (1200 tokens) - overthinking loop
        [
            {
                "role": "assistant",
                "content": "<think>Let me recheck. Suppose bat is 1.05 and ball is 0.05. Wait, let's think again... Alternatively... hold on... verify again...</think>The ball is \\boxed{0.05}.",
            }
        ],
        # Candidate 4: Incorrect (80 tokens)
        [
            {
                "role": "assistant",
                "content": "<think>1.10 minus 1.00 is 0.10.</think>The ball costs \\boxed{0.10}.",
            }
        ],
    ]
    # Simulated token counts for each candidate
    simulated_token_ids = [[1] * 120, [1] * 450, [1] * 1200, [1] * 80]

    reward_fn = get_length_scaled_accuracy_reward(alpha=0.5, incorrect_reward=-1.0)
    rewards = reward_fn(
        completions=completions,
        solution=solution,
        completion_ids=simulated_token_ids,
    )

    print(f"{'Candidate':<15} | {'Tokens':<8} | {'Correct':<8} | {'Raw Reward':<12} | {'Advantage':<12}")
    print("-" * 75)

    # Calculate GRPO advantages across the group
    valid_rewards = [r for r in rewards if r is not None]
    mean_r = sum(valid_rewards) / len(valid_rewards)
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in valid_rewards) / len(valid_rewards))

    for idx, (ids, r) in enumerate(zip(simulated_token_ids, rewards, strict=True), start=1):
        adv = (r - mean_r) / (std_r + 1e-6) if r is not None else 0.0
        is_corr = "Yes" if r is not None and r > 0 else "No"
        print(f"Candidate #{idx:<8} | {len(ids):<8} | {is_corr:<8} | {r:>10.3f}   | {adv:>+10.3f}")

    print("-" * 75)
    print("Takeaway:")
    print("  • Candidate #1 (concise reasoning, 120 tokens) receives the highest positive advantage.")
    print("  • Candidate #3 (overthinking loop, 1200 tokens) is penalized despite arriving at the correct answer.")
    print("  • Candidate #4 (incorrect answer) receives the negative baseline penalty.")
    print("=" * 75)


# -----------------------------------------------------------------------------
# 3. Main Training Entry Point
# -----------------------------------------------------------------------------


def main() -> None:
    if "--demo" in sys.argv:
        run_demo()
        return

    import torch
    from datasets import load_dataset

    from trl import (
        GRPOConfig,
        GRPOTrainer,
        ModelConfig,
        ScriptArguments,
        TrlParser,
        get_peft_config,
        get_quantization_config,
    )

    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )

    # Dataset: GSM8K
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    def format_sample(sample: dict[str, str]) -> dict[str, Any]:
        return {
            "prompt": [{"role": "user", "content": sample["question"]}],
            "solution": sample["answer"].split("####")[-1].strip(),
        }

    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    # Initialize the inline length-scaled accuracy reward
    length_scaled_reward = get_length_scaled_accuracy_reward(
        alpha=0.5,
        incorrect_reward=-1.0,
    )

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[think_format_reward, length_scaled_reward],
        train_dataset=dataset,
        quantization_config=get_quantization_config(model_args),
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.end()


if __name__ == "__main__":
    main()
