# Sudoku GRPO Training

Training a 3B LLM (Qwen2.5-3B-Instruct) to solve 9×9 Sudoku puzzles using GRPO (Group Relative Policy Optimization).

## Project Overview

This project implements a complete training pipeline from SFT cold-start to GRPO reinforcement learning, enabling a 3B model to develop multi-step reasoning capability for 9×9 Sudoku puzzles under 24GB single-GPU memory constraints.

Key technical contributions:
- DFS-style trial-and-error Chain-of-Thought data synthesis
- Multi-dimensional reward design (correctness, format, row/column/block logic, clue preservation, brevity, backtracking)
- Memory-efficient training with 4-bit quantization + LoRA + paged optimizer

## Directory Structure

```
├── scripts/
│   ├── sudoku_grpo.py              # Main single-GPU GRPO training script
│   ├── create_sudoku_dataset.py    # Sudoku dataset generation with DFS-style CoT
│   ├── eval_9x9.py                 # Single-GPU evaluation
│   └── debug_generations_callback.py # Debug callback for generation inspection
├── dataset/
│   └── sudoku_cluewise.csv         # Training dataset
├── configs/
│   ├── grpo_config.yaml            #  GRPO configuration
├── results/
│   ├── grpo_exam_results.json      # GRPO model evaluation results
│   └── sudoku_reasoning_dataset.json # Generated reasoning dataset
├── reward_images/                  # Reward function visualization plots
├── test_grpo_10steps.sh            # Quick 10-step test script

```

## Quick Start

### Single GPU Training

```bash
python sudoku_grpo.py
```

### Multi-GPU Training

```bash
# Linux
bash run_grpo_multicard.sh

# Windows
.\run_grpo_multicard.ps1
```

### Quick Test (10 steps)

```bash
bash test_grpo_10steps.sh
```

## Key Parameters

| Parameter | Single-GPU | Multi-GPU |
|-----------|-----------|-----------|
| Model | Qwen2.5-3B-Instruct (4-bit) | Same |
| num_generations | 12 | 12 |
| max_completion_length | 4096 | 1024 |
| beta (KL penalty) | 0.001 | 0.003 |
| per_device_train_batch_size | 12 (auto-adjusted) | 1 |
| gradient_accumulation_steps | 1 | 4 |

## Reward Functions

1. **exact_answer_reward_func** (weight: 10.0) - Final answer correctness
2. **clue_preservation_reward_func** (weight: 5.0) - Prevents modifying original clues
3. **row_logic_reward_func** (weight: 1.0) - Row constraint satisfaction
4. **col_logic_reward_func** (weight: 1.0) - Column constraint satisfaction
5. **block_logic_reward_func** (weight: 1.0) - 3×3 block constraint satisfaction
6. **soft_format_reward_func** (weight: 1.0) - Output format compliance
7. **brevity_reward_func** (weight: 1.0) - Conciseness penalty
8. **backtracking_reward_func** (weight: 1.0) - Encourages DFS-style backtracking

## Requirements

- Python 3.10+
- PyTorch 2.5+
- TRL 0.14+
- Unsloth
- bitsandbytes
- accelerate

## Model

Base model: `unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit`

LoRA config:
- rank (r): 16
- alpha: 32 (2×r)
- target_modules: all_linear (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)

## Training Details

- Total steps: 1000
- Per step: 1 unique prompt × 12 rollouts
- Total prompts covered: ~1000
- Average completion length: ~454 tokens
- Peak GPU memory: ~22GB (RTX 4090D, 24GB)

## License

MIT
