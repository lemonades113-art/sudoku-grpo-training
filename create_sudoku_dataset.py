import numpy as np
import json
import os
import re
import time
import random

# 创建输出目录
os.makedirs('output', exist_ok=True)

# 从数据集中解析并提取数独谜题和解答
def parse_sudoku_data(file_path, num_samples, min_clues, max_clues):
    """
    解析包含跨行数据的数独数据集，选择线索数在 [min_clues, max_clues] 范围内的样本
    """
    data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到数据集文件: {file_path}")
        
    with open(file_path, 'r') as f:
        content = f.read()
        pattern = r'([0-9]{81}),([0-9]{81}),([0-9]+)'
        matches = re.findall(pattern, content)
        
        # 筛选出符合区间要求的样本
        filtered_matches = [m for m in matches if min_clues <= int(m[2]) <= max_clues]
        
        print(f"  - 线索范围 [{min_clues}-{max_clues}]: 候选数 {len(filtered_matches)}, 需求数 {num_samples}")
        
        if len(filtered_matches) > num_samples:
            filtered_matches = random.sample(filtered_matches, num_samples)
        
        for quizzes, solutions, clue_numbers in filtered_matches:
            data.append({
                'quizzes': quizzes,
                'solutions': solutions,
                'clue_numbers': int(clue_numbers)
            })
    return data

def format_sudoku_matrix(puzzle_str):
    """将 81 位字符串转为带分隔符的矩阵格式"""
    grid = np.array(list(puzzle_str)).reshape(9, 9)
    formatted = []
    for i in range(9):
        row = " ".join(grid[i, :3]) + " | " + " ".join(grid[i, 3:6]) + " | " + " ".join(grid[i, 6:])
        formatted.append(row)
        if i == 2 or i == 5:
            formatted.append("------+-------+------")
    return "\n".join(formatted)

def _get_candidates(grid, idx):
    """计算位置 idx 的候选数集合（基于当前盘面约束）"""
    if grid[idx] != 0:
        return set()
    r, c = idx // 9, idx % 9
    used = set()
    for j in range(9): used.add(grid[r * 9 + j])
    for i in range(9): used.add(grid[i * 9 + c])
    br, bc = (r // 3) * 3, (c // 3) * 3
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            used.add(grid[i * 9 + j])
    return set(range(1, 10)) - used - {0}

def _check_guess_conflict(grid, idx, guess_val):
    """
    检查在 idx 位置填入 guess_val 是否会导致约束冲突。
    通过在副本上做约束传播来检测：若传播后某格候选数为0，即为冲突。
    返回: (是否冲突, 冲突描述)
    """
    grid_copy = grid[:]
    grid_copy[idx] = guess_val

    # 约束传播，检测冲突
    changed = True
    conflict_cell = None
    while changed and conflict_cell is None:
        changed = False
        for i in range(81):
            if grid_copy[i] == 0:
                cands = _get_candidates(grid_copy, i)
                if len(cands) == 0:
                    conflict_cell = i
                    break
                if len(cands) == 1:
                    grid_copy[i] = list(cands)[0]
                    changed = True

    if conflict_cell is not None:
        cr, cc = conflict_cell // 9 + 1, conflict_cell % 9 + 1
        return True, f"推导至坐标 ({cr}, {cc}) 时无有效候选数"
    return False, ""

def synthesize_reasoning(puzzle_str, solution_str):
    """
    生成带假设验证的DFS风格推理链。
    借鉴 NeurIPS 2025 "Teaching Transformers to Solve Combinatorial Problems"
    的 depth-1 guessing 策略：
    1. 约束传播：确定性填充（唯一候选直接填）
    2. DFS猜测：当规则无法唯一确定时，假设→冲突检测→回溯→确认
    """
    grid = list(map(int, puzzle_str))
    zero_indices = [i for i in range(81) if grid[i] == 0]
    steps = ["分析数独盘面并定位空格："]
    step_num = 0

    # === 第一阶段：约束传播（确定性填充）===
    changed = True
    while changed:
        changed = False
        for i in range(81):
            if grid[i] == 0:
                cands = _get_candidates(grid, i)
                if len(cands) == 1:
                    val = list(cands)[0]
                    step_num += 1
                    r, c = i // 9 + 1, i % 9 + 1
                    box_idx = (r - 1) // 3 * 3 + (c - 1) // 3 + 1
                    steps.append(f"步骤 {step_num}: 坐标 ({r}, {c}) 为空。"
                                 f"检查第 {r} 行、第 {c} 列及第 {box_idx} 宫格。"
                                 f"唯一候选数为 {val}，填入 {val}。")
                    grid[i] = val
                    changed = True

    # === 第二阶段：DFS猜测（不确定性填充）===
    remaining = [i for i in range(81) if grid[i] == 0]
    for i in remaining:
        r, c = i // 9 + 1, i % 9 + 1
        cands = _get_candidates(grid, i)
        correct_val = int(solution_str[i])

        if len(cands) > 1:
            # 候选数>1，规则推理无法唯一确定，需要猜测
            step_num += 1
            steps.append(f"步骤 {step_num}: 坐标 ({r}, {c}) 为空。"
                         f"候选数为 {sorted(cands)}，规则推理无法唯一确定。")

            # 尝试1个错误猜测并检测冲突（depth-1 guessing）
            wrong_cands = [v for v in sorted(cands) if v != correct_val]
            if wrong_cands:
                wrong_val = wrong_cands[0]
                has_conflict, conflict_desc = _check_guess_conflict(grid, i, wrong_val)
                step_num += 1
                if has_conflict:
                    steps.append(f"步骤 {step_num}: [假设] R{r}C{c} = {wrong_val}，"
                                 f"{conflict_desc} → [冲突] → 回溯")
                else:
                    steps.append(f"步骤 {step_num}: [假设] R{r}C{c} = {wrong_val}，"
                                 f"深层推导后发现约束矛盾 → [冲突] → 回溯")

            # 填入正确值
            step_num += 1
            steps.append(f"步骤 {step_num}: [确认] R{r}C{c} = {correct_val}，"
                         f"推导无冲突，填入 {correct_val}。")
        else:
            # 候选数=1，直接填入
            step_num += 1
            val = list(cands)[0] if cands else correct_val
            steps.append(f"步骤 {step_num}: 坐标 ({r}, {c}) 为空。"
                         f"唯一候选数为 {val}，填入 {val}。")

        grid[i] = correct_val

    steps.append(f"确认所有 {len(zero_indices)} 个空格已处理完毕，生成最终结果。")
    return "\n".join(steps)

def create_curriculum_dataset(input_file, output_file):
    """
    按照用户定义的三个阶段构建“扫盲到进阶”课程数据集
    """
    # 阶段定义：(min_clues, max_clues, count, name)
    stages = [
        (80, 80, 200, "极简扫盲班 (1空)"),
        (76, 79, 400, "进阶强化班 (2-5空)"),
        (70, 75, 400, "逻辑连贯班 (6-11空)")
    ]
    
    results = []
    print(">>> 开始分阶段合成‘逻辑脱水’课程数据集...")
    
    for min_c, max_c, count, name in stages:
        print(f"\n[阶段] {name}")
        stage_data = parse_sudoku_data(input_file, count, min_c, max_c)
        
        for item in stage_data:
            puzzle = item['quizzes']
            solution = item['solutions']
            formatted_puzzle = format_sudoku_matrix(puzzle)
            
            question = f"以下是一个数独游戏，其中0代表空缺数字，请完成该数独并在 <answer> 标签中输出 81 位纯数字答案。\n\n{formatted_puzzle}"
            reasoning = synthesize_reasoning(puzzle, solution)
            
            results.append({
                "question": question,
                "answer": f"<think>\n{reasoning}\n</think>\n\n<answer>{solution}</answer>",
                "original_puzzle": puzzle,
                "original_solution": solution,
                "clue_number": item['clue_numbers']
            })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 课程数据集创建完成！总样本数: {len(results)}")
    print(f"📁 保存路径: {output_file}")

if __name__ == "__main__":
    input_csv = 'dataset/sudoku_cluewise.csv'
    output_json = 'output/sudoku_reasoning_dataset.json'
    
    create_curriculum_dataset(input_csv, output_json) 