# GRPO with Length-Scaled Reasoning Rewards (GRPO-LEAD)

This example demonstrates how to train reasoning models with **Group Relative Policy Optimization (GRPO)** using an inline **length-scaled accuracy reward** to mitigate reasoning overthinking.

Reference: *"GRPO-LEAD: Length-scaled Efficiency-Aware Decoupling for Reasoning Models"* ([arXiv:2504.09696](https://huggingface.co/papers/2504.09696)).

---

## The Overthinking Problem

When post-training reasoning models (such as DeepSeek-R1 or Qwen-2.5-Math) with standard rule-based verifiers, models frequently develop **overthinking pathologies**:
- They generate circular, verbose internal reasoning traces for trivial questions.
- Standard accuracy verifiers award full credit ($1.0$) as long as the final boxed answer is correct, offering no gradient pressure against verbosity.
- Fixed length thresholds (e.g., hard token caps) can abruptly truncate valid long chains of thought required for genuinely hard problems.

---

## The GRPO-LEAD Solution

Rather than setting an arbitrary global token cap, **GRPO-LEAD** computes completion lengths **relative to the other correct completions of the same prompt** (a group-relative z-score):

1. **Correctness Verification**: Check if the final boxed answer matches ground truth using math verification.
2. **Group Length Standardization**: For all correct completions in group $G$, compute the mean length $\mu_{|y|}$ and standard deviation $\sigma_{|y|}$:
   $$z_i = \frac{|y_i| - \mu_{|y|}}{\sigma_{|y|} + \epsilon}$$
3. **Efficiency Reward Shaping**:
   $$R(y_i) = \begin{cases} \exp(-\alpha \cdot z_i) & \text{if correct} \\ R_{\text{incorrect}} & \text{if incorrect} \end{cases}$$

- Shorter correct completions ($z_i < 0$) receive rewards $> 1.0$.
- Verbose correct completions ($z_i > 0$) receive rewards $< 1.0$.
- Incorrect completions receive the baseline penalty $R_{\text{incorrect}} = -1.0$.

### Note on Group Standardization in GRPO

GRPO standardizes rewards within each group by default ($A_i = \frac{R_i - \mu_R}{\sigma_R}$).
For groups where all sampled completions are correct, $\exp(-\alpha \cdot z)$ behaves nearly identically to $-z$ because of downstream normalization.
However, $\alpha$ serves as an essential hyperparameter in two key scenarios:
1. **Mixed Groups** (some correct, some incorrect): $\alpha$ sets the relative magnitude of the length incentive compared to the correct-vs-incorrect reward gap.
2. **Unstandardized Training**: When training with `GRPOConfig(scale_rewards="none")`.

---

## Quickstart

### 1. Interactive Demo (No GPU Required)
Run the script in demo mode to inspect how length-scaled rewards score candidates and calculate advantages on simulated rollouts:

```bash
python examples/grpo_lead_length_reward/grpo_lead_length_reward.py --demo
```

### 2. Training with Accelerate
Run GRPO training on GSM8K using a lightweight model (`Qwen/Qwen2.5-0.5B-Instruct`):

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
