#!/bin/bash

# 快速测试脚本：10步GRPO验证流程通畅

echo "🚀 开始10步GRPO测试..."
echo "📝 参数配置已改为:"
echo "   - max_steps: 10"
echo "   - logging_steps: 1"
echo "   - report_to: [swanlab, tensorboard]"
echo ""

cd /root/autodl-tmp/grpo_for_soduku_train

# 使用双卡ZeRO-2配置运行
accelerate launch --config_file accelerate_config_zero2.yaml \
  改动双卡_最终版.py --config grpo_config.yaml

echo ""
echo "✅ 测试完成！"
echo "📊 输出位置: output/sudoku_grpo_trl_edition"
echo "📈 SwanLab日志: https://swanlab.cn (登录后查看)"
echo "📁 TensorBoard: tensorboard --logdir output/logs"
