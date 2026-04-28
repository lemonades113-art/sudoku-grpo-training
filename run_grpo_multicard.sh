#!/bin/bash

# ============================================================
# sudoku_grpo.py 多卡启动脚本（方案 B：保守方案）
# ============================================================

echo "🚀 启动 sudoku GRPO 多卡训练..."
echo ""

# 检查 GPU 数量
num_gpus=$(nvidia-smi --list-gpus | wc -l)
echo "📊 检测到 $num_gpus 张 GPU"
echo ""

# 检查 accelerate 和 deepspeed
pip list | grep -i accelerate > /dev/null || pip install accelerate
pip list | grep -i deepspeed > /dev/null || pip install deepspeed

echo "✅ 依赖检查完成"
echo ""

# 确定启动方式
if [ $num_gpus -ge 2 ]; then
    echo "🔧 [多卡模式] 启用 DeepSpeed ZeRO-2（$num_gpus 卡）"
    echo "⏱️  预期耗时: 28-32 分钟（vs 单卡 50-60 分钟）"
    echo ""
    
    # 双卡启动
    accelerate launch --config_file accelerate_config_zero2.yaml sudoku_grpo.py
else
    echo "⚠️  [单卡模式] 仅检测到 1 张 GPU"
    echo "💡 建议：若有多卡可用，建议升级到多卡配置"
    echo ""
    python sudoku_grpo.py
fi

echo ""
echo "✨ 训练完成！"
