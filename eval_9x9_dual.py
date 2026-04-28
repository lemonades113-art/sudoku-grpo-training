"""
数独模型评估脚本 - 支持双卡环境
用法：
    单卡: python eval_9x9_双卡版.py
    双卡: CUDA_VISIBLE_DEVICES=0 python eval_9x9_双卡版.py  (只用单卡评估)
"""

# ========================
# PyTorch 2.5.1 + torchao 0.15.0 兼容性补丁（必须在所有导入之前）
# ========================
import torch
for attr in ["int1", "int2", "int3", "int4", "int5", "int6", "int7"]:
    if not hasattr(torch, attr):
        setattr(torch, attr, torch.uint8)

import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ========================
# GPU 配置（自动适配单/双卡）
# ========================
# HuggingFace 镜像加速
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"

if torch.cuda.device_count() > 1:
    # 双卡环境：只用第一张卡评估（避免模型重复加载）
    device = torch.device("cuda:0")
    print(f"🚗 检测到 {torch.cuda.device_count()} 张 GPU，使用 cuda:0 进行评估")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚗 使用设备: {device}")

# 模型路径 - GRPO 训练保存的是 LoRA adapter
MODEL_PATH = "./output/sudoku_grpo_trl_edition"
BASE_MODEL_NAME = "./output/sft_model"  # 改为本地 SFT 模型路径，不再从 HF 下载
NUM_TEST_SAMPLES = 100  # 测试样本数
OUTPUT_RESULT_PATH = "eval_results.json"

# ========================
# 固定随机种子
# ========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ========================
# 数独逻辑检查
# ========================
def check_sudoku_logic(ans):
    """检查数独答案的合法性，返回通过率"""
    if len(ans) != 81 or not ans.isdigit():
        return 0.0
    
    try:
        grid = np.array([int(d) for d in ans]).reshape(9, 9)
        
        # 行约束
        row_score = sum(len(np.unique(row)) == 9 for row in grid) / 9
        # 列约束
        col_score = sum(len(np.unique(col)) == 9 for col in grid.T) / 9
        # 宫格约束
        block_score = 0
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                block = grid[i:i+3, j:j+3].flatten()
                block_score += len(np.unique(block)) == 9
        block_score /= 9
        
        return (row_score + col_score + block_score) / 3
    except:
        return 0.0

def extract_xml_answer(text):
    """从模型输出中提取81位纯数字答案"""
    if not isinstance(text, str):
        return ""
    if "<answer>" in text:
        ans = text.split("<answer>")[-1].split("</answer>")[0].strip()
    else:
        ans = text.strip()
    cleaned = re.sub(r'[^0-9]', '', ans)
    return cleaned[:81]

# ========================
# 核心评估流程
# ========================
def run_final_exam():
    set_seed(42)
    
    # 加载数据
    CSV_PATH = "dataset/sudoku_cluewise.csv"
    if not os.path.exists(CSV_PATH):
        CSV_PATH = "sudoku_cluewise.csv"  # 备选路径
    
    print(f"📊 正在加载数据: {CSV_PATH}")
    # 强制以字符串读取，防止 0 开头的题目被截断
    df = pd.read_csv(CSV_PATH, dtype={'quizzes': str, 'solutions': str})
    test_samples = df.sample(n=min(NUM_TEST_SAMPLES, len(df)), random_state=42)
    
    print(f"📊 已加载 {len(test_samples)} 条测试数据")
    
    # 加载模型
    print(f"📦 正在加载模型: {MODEL_PATH} (LoRA adapter)")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # 加载基础模型 + LoRA 适配器
    print(f"🔗 加载基础模型: {BASE_MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("🔗 加载 LoRA 权重...")
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    
    model.eval()
    print("✅ 模型加载完成！")
    print(f"🔢 模型参数量: {model.num_parameters() / 1e6:.1f}M")
    
    # 系统提示词
    system_prompt = """你是一个数独助手。请通过思考逻辑逐步解决数独，并按格式回答：
<think>
推理过程
</think>
<answer>
81位纯数字答案
</answer>"""
    
    # 评估统计
    summary = {
        "perfect_match": 0,
        "logic_pass_rate": 0,
        "format_correct": 0,
        "clue_preserved": 0
    }
    
    # 难度分级统计
    difficulty_stats = {
        "very_easy": {"clue_range": (80, 81), "perfect": 0, "logic": 0, "count": 0},      # >= 80 线索
        "easy": {"clue_range": (75, 80), "perfect": 0, "logic": 0, "count": 0},         # 75-79 线索
        "medium": {"clue_range": (70, 75), "perfect": 0, "logic": 0, "count": 0}       # 70-74 线索
    }
    
    results = []
    
    print(f"\n🚀 开始评估 {len(test_samples)} 个数独谜题...")
    
    for idx, (_, row) in enumerate(test_samples.iterrows()):
        puzzle = str(row['quizzes']).strip()
        solution = str(row['solutions']).strip()
        clue_count = int(row['clue_numbers']) if 'clue_numbers' in row.index else puzzle.count('0') - 81  # 反推线索数
        
        # 确保题目是 81 位（不足补 0，防止报错）
        if len(puzzle) < 81:
            puzzle = puzzle.zfill(81)
        if len(solution) < 81:
            solution = solution.zfill(81)
        
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n请填满以下数独：\n{puzzle}\n0代表空格。请输出81位纯数字。<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        
        # 贪婪解码（确定性）
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                use_cache=True,
                do_sample=False,
            )
        
        full_resp = tokenizer.batch_decode(outputs)[0]
        assistant_resp = full_resp.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
        
        extracted_ans = extract_xml_answer(assistant_resp)
        
        # 多维审计
        is_perfect = (extracted_ans == solution)
        if is_perfect:
            summary["perfect_match"] += 1
        
        logic_score = check_sudoku_logic(extracted_ans)
        summary["logic_pass_rate"] += logic_score
        
        is_format_ok = "<answer>" in assistant_resp
        if is_format_ok:
            summary["format_correct"] += 1
        
        # 线索保真
        is_clue_ok = True
        for i in range(81):
            if puzzle[i] != '0' and (i >= len(extracted_ans) or extracted_ans[i] != puzzle[i]):
                is_clue_ok = False
                break
        if is_clue_ok:
            summary["clue_preserved"] += 1
        
        # 按难度分类统计
        if clue_count >= 80:
            difficulty_stats["very_easy"]["count"] += 1
            difficulty_stats["very_easy"]["perfect"] += int(is_perfect)
            difficulty_stats["very_easy"]["logic"] += logic_score
        elif 75 <= clue_count < 80:
            difficulty_stats["easy"]["count"] += 1
            difficulty_stats["easy"]["perfect"] += int(is_perfect)
            difficulty_stats["easy"]["logic"] += logic_score
        elif 70 <= clue_count < 75:
            difficulty_stats["medium"]["count"] += 1
            difficulty_stats["medium"]["perfect"] += int(is_perfect)
            difficulty_stats["medium"]["logic"] += logic_score
        
        results.append({
            "puzzle": puzzle,
            "solution": solution,
            "model_ans": extracted_ans,
            "is_perfect": is_perfect,
            "logic_score": logic_score,
            "is_format_ok": is_format_ok,
            "is_clue_ok": is_clue_ok,
            "clue_count": clue_count
        })
        
        if (idx + 1) % max(1, len(test_samples) // 10) == 0:
            print(f"  进度: {idx+1}/{len(test_samples)}")
    
    # 最终统计
    if len(test_samples) > 0:
        summary["logic_pass_rate"] /= len(test_samples)
    summary["perfect_match_pct"] = (summary["perfect_match"] / len(test_samples)) * 100
    
    # 难度分级计算
    for level in difficulty_stats:
        if difficulty_stats[level]["count"] > 0:
            difficulty_stats[level]["perfect_pct"] = (difficulty_stats[level]["perfect"] / difficulty_stats[level]["count"]) * 100
            difficulty_stats[level]["logic_avg"] = (difficulty_stats[level]["logic"] / difficulty_stats[level]["count"]) * 100
        else:
            difficulty_stats[level]["perfect_pct"] = 0
            difficulty_stats[level]["logic_avg"] = 0
    
    print("\n" + "="*60)
    print("🎓 数独模型考试成绩单")
    print("="*60)
    print(f"📊 测试数量: {len(test_samples)}")
    print(f"✅ 全对通过率: {summary['perfect_match_pct']:.2f}% ({summary['perfect_match']}/{len(test_samples)})")
    print(f"🔢 逻辑单元均分: {summary['logic_pass_rate']*100:.2f}%")
    print(f"📝 格式合规率: {summary['format_correct']}/{len(test_samples)}")
    print(f"🔒 线索保真率: {summary['clue_preserved']}/{len(test_samples)}")
    print("="*60)
    
    # 难度分级报告
    print("\n📈 难度分级评估报告：")
    print("-" * 60)
    
    difficulty_names = {
        "very_easy": "超简 (≥80线索)",
        "easy": "简单 (75-79线索)",
        "medium": "中等 (70-74线索)"
    }
    
    for level_key in ["very_easy", "easy", "medium"]:
        stats = difficulty_stats[level_key]
        if stats["count"] > 0:
            print(f"\n🎯 {difficulty_names[level_key]}")
            print(f"   样本数: {stats['count']}")
            print(f"   全对率: {stats['perfect_pct']:.2f}% ({stats['perfect']}/{stats['count']})")
            print(f"   逻辑均分: {stats['logic_avg']:.2f}%")
        else:
            print(f"\n🎯 {difficulty_names[level_key]}: 无样本")
    
    print("-" * 60)
    
    # 保存结果
    output_data = {
        "summary": summary,
        "difficulty_stats": difficulty_stats,
        "details": results
    }
    with open(OUTPUT_RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n📁 详细报告已保存至: {OUTPUT_RESULT_PATH}")

if __name__ == "__main__":
    run_final_exam()
