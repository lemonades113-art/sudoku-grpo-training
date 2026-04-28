import os
import sys
import warnings
import torch
from types import ModuleType
import importlib.util
import importlib.metadata

# === 1. 环境补丁 (针对 PyTorch 2.3.0) ===
for attr in ["int1", "int2", "int4", "int3", "int5", "int6", "int7"]:
    if not hasattr(torch, attr): setattr(torch, attr, torch.uint8) 

_old_argsort = torch.argsort
def _patched_argsort(input, *args, **kwargs):
    if torch.is_tensor(input) and input.dtype == torch.bool and input.is_cuda:
        return _old_argsort(input.to(torch.uint8), *args, **kwargs)
    return _old_argsort(input, *args, **kwargs)
torch.argsort = _patched_argsort

os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1" 
os.environ["UNSLOTH_SKIP_TORCH_CHECK"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
import swanlab
swanlab.login(api_key="G188wyu1OGXmemI1EqF4O")

# === 2. 导入与组件 ===
import json
import random
import pandas as pd
import numpy as np
import re
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer, GRPOConfig, GRPOTrainer
import swanlab

# === [核心新增] 可观测性回调函数：让模型边练边“说话”并记录熵 ===
class LogGenerationCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 5 == 0 and "loss" in logs:
            # 1. 计算伪熵
            kl_value = logs.get("train/kl", logs.get("kl", logs.get("train_kl", 0.0)))
            pseudo_entropy = max(0, 1.0 - kl_value) 
            
            # 2. 从全局缓存上传 Table
            if swanlab.get_run() is not None:
                global LAST_COMPLETIONS
                if LAST_COMPLETIONS:
                    # 仅上传前 3 个样本，避免数据量过大
                    table_data = [[state.global_step, i, text[:3000]] for i, text in enumerate(LAST_COMPLETIONS[:3])]
                    my_table = swanlab.Table(columns=["Step", "Idx", "Response"], data=table_data)
                    swanlab.log({"samples/responses": my_table, "custom/pseudo_entropy": pseudo_entropy})
                else:
                    swanlab.log({"custom/pseudo_entropy": pseudo_entropy})

            print(f"\n" + "="*50)
            print(f"[Step {state.global_step}] 实时性能监控：")
            print(f">>> Loss: {logs.get('loss', 'N/A')}")
            print(f">>> Entropy (Pseudo): {pseudo_entropy:.4f}")
            
            logic_reward = logs.get('rewards/simple_robust_partial_reward_function/mean', 
                                   logs.get('train/rewards/simple_robust_partial_reward_function/mean', 'N/A'))
            format_reward = logs.get('rewards/soft_format_reward_func/mean', 
                                    logs.get('train/rewards/soft_format_reward_func/mean', 'N/A'))
            
            print(f">>> Logic Reward: {logic_reward}")
            print(f">>> Format Reward: {format_reward}")
            print(f">>> Clipped Ratio: {logs.get('completions/clipped_ratio', 'N/A')}")
            print("="*50)
            print("💡 监控提示：若 Logic 为 -1.0 且 Format 为 0，说明模型格式崩了，正在盲目探索中。")
            print("="*50 + "\n")

# === 3. 核心逻辑函数 ===

def load_and_prepare_data(data_path, train_ratio=0.9):
    actual_path = data_path
    if not os.path.exists(actual_path):
        for p in [os.path.basename(data_path), os.path.join("reasoning_sft_dataset", os.path.basename(data_path))]:
            if os.path.exists(p): actual_path = p; break
    with open(actual_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"✅ 加载了 {len(dataset)} 条高质量冷启动数据用于评估")
    formatted = []
    for ex in dataset:
        prompt = f"<|im_start|>system\n用以下格式回答问题:\n<think>推理过程</think>\n<answer>答案</answer><|im_end|>\n<|im_start|>user\n{ex['question']}<|im_end|>\n<|im_start|>assistant\n"
        formatted.append({
            "text": prompt + ex["answer"] + "<|im_end|>",
            "original_puzzle": ex.get("original_puzzle", ""),      # 加上补丁字段
            "original_solution": ex.get("original_solution", "")   # 加上补丁字段
        })
    random.shuffle(formatted)
    return Dataset.from_list(formatted)

def evaluate_model_fortified(model, tokenizer, dataset, num_samples=20):
    """
    根据 [Version Answer] 方案加固的评估逻辑：
    1. 格式完备率 (Format Pass Rate)
    2. 坐标定位准确度 (Coordinate Accuracy)
    3. 极简题解对率 (Pass@1)
    """
    print(f"\n🔍 正在进行 SFT 深度评估 (样本数: {num_samples})...")
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    metrics = {
        "format_ok": 0,
        "coord_acc": 0,
        "pass_at_1": 0,
        "coord_total": 0
    }
    
    FastLanguageModel.for_inference(model)
    
    for idx in indices:
        text = dataset[idx]["text"]
        # 提取原题字符串
        raw_puzzle = dataset[idx].get("original_puzzle", "")
        raw_solution = dataset[idx].get("original_solution", "")
        question = text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
        
        inputs = tokenizer([f"<|im_start|>system\n你是一个数独助手。请通过思考逻辑逐步解决数独，并按格式回答：\n<think>\n推理过程\n</think>\n<answer>\n81位纯数字答案\n</answer><|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=2048, use_cache=True, do_sample=False)
        resp = tokenizer.batch_decode(outputs)[0].split("<|im_start|>assistant\n")[-1]
        
        # 1. 检查格式
        has_tags = "<think>" in resp and "</think>" in resp and "<answer>" in resp and "</answer>" in resp
        if has_tags: metrics["format_ok"] += 1
        
        # 2. 检查坐标准确度 (正则提取模型提到的坐标)
        coords = re.findall(r"坐标 \((\d), (\d)\)", resp)
        for r_str, c_str in coords:
            r, c = int(r_str), int(c_str)
            metrics["coord_total"] += 1
            idx_in_str = (r - 1) * 9 + (c - 1)
            if idx_in_str < 81 and raw_puzzle and raw_puzzle[idx_in_str] == '0':
                metrics["coord_acc"] += 1
        
        # 3. 检查 Pass@1 (提取 answer 并对比)
        extracted = extract_xml_answer(resp)
        if extracted == raw_solution:
            metrics["pass_at_1"] += 1
            
    # 计算比例
    format_rate = metrics["format_ok"] / num_samples
    coord_rate = metrics["coord_acc"] / max(1, metrics["coord_total"])
    pass_rate = metrics["pass_at_1"] / num_samples
    
    print(f"  [评估结果]")
    print(f"  - 格式完备率: {format_rate*100:.1f}% (目标: 100%)")
    print(f"  - 坐标准确度: {coord_rate*100:.1f}% (目标: >90%)")
    print(f"  - 极简解对率: {pass_rate*100:.1f}% (目标: >70%)")
    
    return pass_rate

# === 全局缓存：用于可视化监控 ===
LAST_COMPLETIONS = []

# --- [核心修复] 强化版答案提取 ---
def extract_xml_answer(text: str) -> str:
    if not isinstance(text, str): return ""
    # 增加健壮性：寻找最后一个 <answer> 标签
    parts = text.split("<answer>")
    if len(parts) < 2: return ""
    ans_content = parts[-1].split("</answer>")[0].strip()
    # 过滤非数字字符，只取前 81 位
    cleaned = re.sub(r'[^0-9]', '', ans_content)
    return cleaned[:81]

# === 1. 完美解答奖励 (全对重赏，权重 10.0) ===
def exact_answer_reward_func(completions, target_solution, **kwargs):
    responses = [c[0]['content'] for c in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    rewards = []
    for r, a in zip(extracted, target_solution):
        if r == a:
            rewards.append(10.0) # 终极目标：满分
        elif len(r) == 81:
            # 位次匹配奖励：保持梯度，防止奖励沙漠
            match_count = sum(1 for i in range(81) if r[i] == a[i])
            rewards.append(match_count / 81.0) # 0.012 - 1.0 之间
        else:
            rewards.append(0.0)
    return rewards

# === 2. 线索匹配奖励 (防双重Hacking：篡改线索 + 复制谜题不解题，权重 5.0) ===
def clue_preservation_reward_func(completions, quiz_str, **kwargs):
    responses = [c[0]['content'] for c in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    rewards = []
    for r, q in zip(extracted, quiz_str):
        if len(r) != 81:
            rewards.append(0.0)
            continue
        
        clue_indices = [i for i, char in enumerate(q) if char != '0']
        non_clue_indices = [i for i, char in enumerate(q) if char == '0']  # 空白位索引
        if not clue_indices:
            rewards.append(1.0)
            continue
        
        # 核心检查1：线索是否被篡改
        match_clue = sum(1 for i in clue_indices if r[i] == q[i]) / len(clue_indices)
        # 核心检查2：空白位是否被填充（防复制谜题不解题）
        non_zero_fill = sum(1 for i in non_clue_indices if r[i] != '0') / max(1, len(non_clue_indices)) if non_clue_indices else 1.0
        
        # 综合奖励：两个条件都满足才给分
        if match_clue < 1.0:
            # 篡改线索的惩罚保持不变
            rewards.append(match_clue - 2.0)
        else:
            # 完美保住线索后，再检查是否真的解了题（填充了空白位）
            rewards.append(1.0 * non_zero_fill)
    return rewards

# === 3. 拆解式逻辑奖励 - 行 (防伪唯一值Hacking：分离线索位与填充位，权重 1.0) ===
def row_logic_reward_func(completions, quiz_str, **kwargs):
    responses = [c[0]['content'] for c in completions]
    rewards = []
    for r, q in zip(responses, quiz_str):
        ans = extract_xml_answer(r)
        if len(ans) != 81: 
            rewards.append(0.0)
            continue
        try:
            grid = np.array([int(d) for d in ans]).reshape(9, 9)
            quiz_grid = np.array([int(d) for d in q]).reshape(9, 9)
            score = 0.0
            
            for i in range(9):
                row = grid[i, :]
                quiz_row = quiz_grid[i, :]
                fill_indices = quiz_row == 0  # 空白位
                fill_vals = row[fill_indices]  # 模型填充的数值
                
                if len(fill_vals) > 0:
                    # 核心检查：填充位是否有效
                    is_non_zero = (fill_vals != 0).all()  # 不能全是0
                    row_clues = quiz_row[quiz_row != 0]
                    no_clue_dup = not any(v in row_clues for v in fill_vals)  # 不与线索重复
                    
                    if is_non_zero and no_clue_dup:
                        # 有效填充：奖励
                        fill_unique = len(np.unique(fill_vals)) / len(fill_vals)
                        row_unique = len(np.unique(row)) / 9
                        row_score = (row_unique + fill_unique) / 2
                        score += row_score
                    else:
                        # 无效填充：扣分
                        score += 0
                else:
                    # 无空白位（都是线索）：按唯一值打分
                    score += len(np.unique(row)) / 9
            
            rewards.append(score / 9)
        except: 
            rewards.append(0.0)
    return rewards

def col_logic_reward_func(completions, quiz_str, **kwargs):
    responses = [c[0]['content'] for c in completions]
    rewards = []
    for r, q in zip(responses, quiz_str):
        ans = extract_xml_answer(r)
        if len(ans) != 81: 
            rewards.append(0.0)
            continue
        try:
            grid = np.array([int(d) for d in ans]).reshape(9, 9)
            quiz_grid = np.array([int(d) for d in q]).reshape(9, 9)
            score = 0.0
            
            for j in range(9):
                col = grid[:, j]
                quiz_col = quiz_grid[:, j]
                fill_indices = quiz_col == 0
                fill_vals = col[fill_indices]
                
                if len(fill_vals) > 0:
                    is_non_zero = (fill_vals != 0).all()
                    col_clues = quiz_col[quiz_col != 0]
                    no_clue_dup = not any(v in col_clues for v in fill_vals)
                    
                    if is_non_zero and no_clue_dup:
                        fill_unique = len(np.unique(fill_vals)) / len(fill_vals)
                        col_unique = len(np.unique(col)) / 9
                        col_score = (col_unique + fill_unique) / 2
                        score += col_score
                    else:
                        score += 0
                else:
                    score += len(np.unique(col)) / 9
            
            rewards.append(score / 9)
        except: 
            rewards.append(0.0)
    return rewards

def block_logic_reward_func(completions, quiz_str, **kwargs):
    responses = [c[0]['content'] for c in completions]
    rewards = []
    for r, q in zip(responses, quiz_str):
        ans = extract_xml_answer(r)
        if len(ans) != 81: 
            rewards.append(0.0)
            continue
        try:
            grid = np.array([int(d) for d in ans]).reshape(9, 9)
            quiz_grid = np.array([int(d) for d in q]).reshape(9, 9)
            score = 0.0
            
            for bi in range(0, 9, 3):
                for bj in range(0, 9, 3):
                    block = grid[bi:bi+3, bj:bj+3].flatten()
                    quiz_block = quiz_grid[bi:bi+3, bj:bj+3].flatten()
                    fill_indices = quiz_block == 0
                    fill_vals = block[fill_indices]
                    
                    if len(fill_vals) > 0:
                        is_non_zero = (fill_vals != 0).all()
                        block_clues = quiz_block[quiz_block != 0]
                        no_clue_dup = not any(v in block_clues for v in fill_vals)
                        
                        if is_non_zero and no_clue_dup:
                            fill_unique = len(np.unique(fill_vals)) / len(fill_vals)
                            block_unique = len(np.unique(block)) / 9
                            block_score = (block_unique + fill_unique) / 2
                            score += block_score
                        else:
                            score += 0
                    else:
                        score += len(np.unique(block)) / 9
            
            rewards.append(score / 9)
        except: 
            rewards.append(0.0)
    return rewards

# === 4. 格式奖励（分层梯度给分：引导模型闭合标签）===
def soft_format_reward_func(completions, **kwargs):
    """格式奖励层级化，大幅奖励闭合行为，最高 0.6"""
    global LAST_COMPLETIONS
    responses = [c[0]["content"] for c in completions]
    # 拦截生成内容，存入全局缓存供可视化使用
    LAST_COMPLETIONS = responses
    
    rewards = []
    for r in responses:
        score = 0.0
        if "<think>" in r: score += 0.1
        if "</think>" in r: score += 0.2 # 重点奖励：推理结束
        if "<answer>" in r: score += 0.1
        if "</answer>" in r: score += 0.2 # 重点奖励：答案完整
        rewards.append(score)
    return rewards

# === 5. 精简奖励（防止话痨撞墙）===
def brevity_reward_func(completions, **kwargs):
    """
    避免模型因为怕扣分而不敢写推理逻辑。
    DFS回溯推理会导致更长的输出，放宽惩罚阈值至6000。
    """
    responses = [c[0]['content'] for c in completions]
    rewards = []
    for r in responses:
        # 放宽惩罚：只有超过 6000 字符才开始线性扣分（原3500，DFS推理更长）
        if len(r) > 6000:
            penalty = min(0.5, (len(r) - 6000) / 1000)
            rewards.append(-penalty)
        else:
            rewards.append(0.0)
    return rewards

# === 6. 回溯推理奖励（鼓励假设-冲突-回溯的DFS策略，权重 2.0）===
def backtracking_reward_func(completions, **kwargs):
    """
    奖励包含 [假设]→[冲突]→回溯 逻辑的DFS推理链。
    借鉴 NeurIPS 2025 depth-1 guessing 策略：
    当规则推理无法唯一确定时，模型应主动假设并验证。
    """
    responses = [c[0]['content'] for c in completions]
    rewards = []
    for r in responses:
        has_hypothesis = bool(re.search(r'\[假设\]', r))
        has_conflict = bool(re.search(r'\[冲突\]', r))
        has_backtrack = bool(re.search(r'回溯', r))
        has_confirm = bool(re.search(r'\[确认\]', r))

        if has_hypothesis and has_conflict and has_backtrack:
            # 完整的DFS推理链：假设→冲突→回溯
            if has_confirm:
                rewards.append(2.0)  # 假设→冲突→回溯→确认（最完整）
            else:
                rewards.append(1.0)  # 假设→冲突→回溯（部分完整）
        elif has_hypothesis and not has_conflict and not has_backtrack:
            # 有假设但没有冲突检测和回溯（盲目猜测）
            rewards.append(-1.0)
        else:
            # 无假设（直接推理，不奖不罚）
            rewards.append(0.0)
    return rewards

# === 4. 主程序 ===
print(">>> 步骤 1: 环境初始化...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="output/sft_model", 
    max_seq_length=5120, # 听取建议：提升至 5120，为 4096 生成留出 Prompt 空间
    load_in_4bit=True
)

sft_data = load_and_prepare_data("sudoku_reasoning_dataset.json")
success_val = evaluate_model_fortified(model, tokenizer, sft_data, num_samples=5)

if success_val < 3:
    print("\n⚠️ 警报：SFT 模型格式掌握极差！建议先跑 1 轮格式加固训练...")
    # 这里可以添加一个极速 SFT 逻辑，但我们先尝试用 GRPO 强制拉回
else:
    print(f"\n✨ SFT 基础尚可 ({success_val}/5)，直接进入 GRPO...")

print("\n>>> 步骤 2: 准备 GRPO 数据 (混合难度策略)...")
grpo_csv = "sudoku_cluewise.csv" if os.path.exists("sudoku_cluewise.csv") else "dataset/sudoku_cluewise.csv"
df = pd.read_csv(grpo_csv, dtype={'quizzes': str, 'solutions': str}).fillna('0'*81) 

# 抽取 5000 条数据，涵盖 1空 到 11空，保证采样多样性
df_filtered = df[df['clue_numbers'] >= 70] 
if len(df_filtered) > 5000:
    df_filtered = df_filtered.sample(n=5000, random_state=42)
else:
    df_filtered = df_filtered.sample(frac=1.0, random_state=42)

print(f"📊 已抽取 {len(df_filtered)} 条混合难度题目作为强化学习题库")
grpo_data = []

# 修改点 A：借鉴 Script C 的专家 persona 和规则复述，增强 3B 模型稳定性
system_prompt = """你是一个数独专家。请通过思考逻辑逐步解决 9x9 数独。
'0' 表示待填写的空白位置。

## 数独规则
1. 每行：数字 1-9 各出现一次。
2. 每列：数字 1-9 各出现一次。
3. 每个 3x3 宫格：数字 1-9 各出现一次。

## 推理策略
当规则推理无法唯一确定某个格子的值时，采用假设验证法：
1. 列出候选数，选择一个进行假设，标记 [假设]
2. 继续推导验证假设是否成立
3. 若发现冲突（矛盾），标记 [冲突] 并回溯，尝试其他候选
4. 若无冲突，标记 [确认] 并继续推理

## 输出格式
请按以下格式回答：
<think>
详细的推理过程，按坐标锁定数字。
</think>
<answer>
81位纯数字答案
</answer>"""

for _, row in df_filtered.iterrows():
    grpo_data.append({
        'prompt': [
            {'role': 'system', 'content': system_prompt}, 
            {'role': 'user', 'content': f"请填满以下数独：\n{row['quizzes']}\n直接在 <answer> 标签中输出 81 位纯数字答案。"}
        ],
        'target_solution': row['solutions'],
        'quiz_str': row['quizzes']
    })

# === [关键验证] 检查 SFT 与 GRPO 的 Prompt 一致性 ===
print("\n🔍 验证 Prompt 格式一致性...")
sample_prompt = grpo_data[0]['prompt']
formatted_prompt = tokenizer.apply_chat_template(sample_prompt, tokenize=False, add_generation_prompt=True)
print("[GRPO 阶段生成的 Prompt]:\n", formatted_prompt[:200], "...")
print("\n[预期的 SFT 格式应包含]:\n<|im_start|>system\n用以下格式回答问题...\n<|im_end|>\n<|im_start|>user...")
if "<|im_start|>" not in formatted_prompt:
    print("\n⚠️ 警告：GRPO 的 Prompt 格式与 SFT 不一致！模型可能无法正常工作。")
    print("建议：手动拼接 Prompt 字符串，而不是使用 apply_chat_template。")
else:
    print("\n✅ Prompt 格式一致，可以安全训练。")
print("=" * 60)

training_args = GRPOConfig(
    learning_rate=2e-5, 
    lr_scheduler_type="cosine", optim="paged_adamw_8bit",
    logging_steps=1, max_steps=1000, # 提升至 1000 步，冲击顿悟点
    num_generations=12,    
    max_completion_length=4096,
    beta=0.001,            
    output_dir="output/sudoku_grpo_fortified",
    report_to="swanlab"
)

trainer = GRPOTrainer(
    model=model, processing_class=tokenizer,
    reward_funcs=[
        exact_answer_reward_func,        # 权重 10.0 ( 冲击全对 )
        clue_preservation_reward_func,   # 权重 5.0 ( 防两重作弊）
        row_logic_reward_func,           # 权重 1.0 ( 行功课 )
        col_logic_reward_func,           # 权重 1.0 ( 列功课 )
        block_logic_reward_func,         # 权重 1.0 ( 宫功课 )
        soft_format_reward_func,         # 权重 0.1
        brevity_reward_func,             # 权重 0.5
        backtracking_reward_func         # 权重 2.0 ( DFS回溯，借鉴NeurIPS 2025 )
    ],
    args=training_args, train_dataset=Dataset.from_list(grpo_data),
    callbacks=[LogGenerationCallback()] # 注入实时监控
)

print("\n>>> 步骤 3: 启动最终强化学习...")
trainer.train()
swanlab.finish()
