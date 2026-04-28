import os
import sys
import torch

# =========================================================
# 1. 优先级最高：环境补丁与离线设置 (必须在所有 import 之前)
# =========================================================
# 彻底修复 AttributeError: module 'torch' has no attribute 'int1'
for attr in ["int1", "int2", "int3", "int4", "int5", "int6", "int7"]:
    if not hasattr(torch, attr):
        setattr(torch, attr, torch.int8) 

# 修复网络连接问题与强制离线 (AutoDL 专用)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"  
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"
os.environ["UNSLOTH_SKIP_TORCH_CHECK"] = "1"

import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# === 1. 配置区 ===
# GRPO 模型路径（LoRA adapter）- 指向最新的 checkpoint
GRPO_MODEL_PATH = "./output/sudoku_grpo_fortified/checkpoint-1000"
# SFT 基础模型
BASE_MODEL_PATH = "./output/sft_model"
TEST_DATA_PATH = "./dataset/sudoku_cluewise.csv"
OUTPUT_RESULT_PATH = "./output/grpo_exam_results.json"
NUM_TEST_SAMPLES = 100
MAX_SEQ_LENGTH = 5120

# === 2. 工具函数 (与训练保持一致) ===
def extract_xml_answer(text: str) -> str:
    if not isinstance(text, str): return ""
    parts = text.split("<answer>")
    if len(parts) < 2: return ""
    ans_content = parts[-1].split("</answer>")[0].strip()
    cleaned = re.sub(r'[^0-9]', '', ans_content)
    return cleaned[:81]

def check_sudoku_logic(answer_str: str):
    """审计 9x9 数独的逻辑正确性"""
    if len(answer_str) != 81: return 0.0
    try:
        grid = np.array([int(d) for d in answer_str]).reshape(9, 9)
        valid_units = 0
        # 检查行/列
        for i in range(9):
            if len(set(grid[i, :])) == 9 and 0 not in grid[i, :]: valid_units += 1
            if len(set(grid[:, i])) == 9 and 0 not in grid[:, i]: valid_units += 1
        # 检查 3x3 宫
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                box = grid[r:r+3, c:c+3].flatten()
                if len(set(box)) == 9 and 0 not in box: valid_units += 1
        return valid_units / 27.0
    except:
        return 0.0

# === 3. 核心评估流程 ===
def run_final_exam():
    print(f">>> \u6b63\u5728\u52a0\u8f7d GRPO \u6a21\u578b...")
    print(f"   - LoRA adapter: {GRPO_MODEL_PATH}")
    print(f"   - \u57fa\u7840\u6a21\u578b: {BASE_MODEL_PATH}")
    
    # 从 SFT \u6a21\u578b\u52a0\u8f7d tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    # \u52a0\u8f7d SFT \u57fa\u7840\u6a21\u578b
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # \u52a0\u8f7d GRPO LoRA \u6743\u91cd【\u4e0d\u4ece HF \u4e0b\u8f7d\uff0c\u4f7f\u7528 local_files_only\u3011
    model = PeftModel.from_pretrained(
        base_model, 
        GRPO_MODEL_PATH,
        is_trainable=False,
        local_files_only=True  # \u2190 \u5f3a\u5236\u4f7f\u7528\u672c\u5730\u8def\u5f84
    )
    model.eval()
    
    # 设备初始化
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f">>> 使用设备: {device}")
    model = model.to(device)
    
    print(f">>> 准备测试数据: {TEST_DATA_PATH}")
    df = pd.read_csv(TEST_DATA_PATH, dtype={'quizzes': str, 'solutions': str}).fillna('0'*81)
    test_samples = df[df['clue_numbers'] >= 78].sample(n=min(NUM_TEST_SAMPLES, len(df)), random_state=99)

    results = []
    summary = {
        "total": len(test_samples),
        "perfect_match": 0,      # 81位完全一致
        "logic_pass_rate": 0.0,  # 27单元通过率均值
        "format_correct": 0,     # 标签闭合率
        "clue_preserved": 0      # 原题数字未改动
    }

    print(f">>> 开始闭卷考试... (评估 GRPO 模型)")
    for idx, row in tqdm(test_samples.iterrows(), total=len(test_samples)):
        puzzle = str(row['quizzes']).strip()
        solution = str(row['solutions']).strip()
        
        # 保证 81 位
        if len(puzzle) < 81:
            puzzle = puzzle.zfill(81)
        if len(solution) < 81:
            solution = solution.zfill(81)
        
        system_prompt = "你是一个数独助手。请通过思考逻辑逐步解决数独，并按格式回答：\n<think>\n推理过程\n</think>\n<answer>\n81位纯数字答案\n</answer>"
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n请填满以下数独：\n{puzzle}\n0代表空格。请输出81位纯数字。<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        
        # 贪婷解码
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                use_cache=True,
                do_sample=False
            )
        
        full_resp = tokenizer.batch_decode(outputs)[0]
        assistant_resp = full_resp.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
        
        extracted_ans = extract_xml_answer(assistant_resp)
        
        # --- 多维审计 ---
        # 1. 完美匹配
        is_perfect = (extracted_ans == solution)
        if is_perfect: summary["perfect_match"] += 1
        
        # 2. 逻辑通过率
        logic_score = check_sudoku_logic(extracted_ans)
        summary["logic_pass_rate"] += logic_score
        
        # 3. 格式校验
        is_format_ok = "<think>" in assistant_resp and "</think>" in assistant_resp and "</answer>" in assistant_resp
        if is_format_ok: summary["format_correct"] += 1
        
        # 4. 线索保真
        is_clue_ok = True
        for i in range(81):
            if puzzle[i] != '0' and (i >= len(extracted_ans) or extracted_ans[i] != puzzle[i]):
                is_clue_ok = False
                break
        if is_clue_ok: summary["clue_preserved"] += 1

        results.append({
            "puzzle": puzzle,
            "solution": solution,
            "model_ans": extracted_ans,
            "is_perfect": is_perfect,
            "logic_score": logic_score,
            "is_format_ok": is_format_ok,
            "is_clue_ok": is_clue_ok
        })

    # 计算最终统计
    if len(test_samples) > 0:
        summary["logic_pass_rate"] /= len(test_samples)
    summary["perfect_match_pct"] = (summary["perfect_match"] / len(test_samples)) * 100
    
    print("\n" + "="*30)
    print("🎓 GRPO 模型考试成绩单")
    print(f"全对通过率 (Exact Match): {summary['perfect_match_pct']:.2f}%")
    print(f"逻辑单元均分 (Logic Score): {summary['logic_pass_rate']*100:.2f}%")
    print(f"格式合规率: {summary['format_correct']}/{len(test_samples)}")
    print(f"线索保真率: {summary['clue_preserved']}/{len(test_samples)}")
    print("="*30)

    with open(OUTPUT_RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump({"summary": summary, "details": results}, f, indent=2, ensure_ascii=False)
    print(f"详细报告已保存至: {OUTPUT_RESULT_PATH}")

if __name__ == "__main__":
    run_final_exam()
