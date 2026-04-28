# ============================================================
# sudoku_grpo.py 多卡启动脚本（方案 B：保守方案）- PowerShell 版本
# ============================================================

Write-Host "🚀 启动 sudoku GRPO 多卡训练..." -ForegroundColor Green
Write-Host ""

# 检查 GPU 数量
$gpu_output = & nvidia-smi --list-gpus 2>&1
$num_gpus = ($gpu_output | Measure-Object -Line).Lines
Write-Host "📊 检测到 $num_gpus 张 GPU" -ForegroundColor Cyan
Write-Host ""

# 检查 accelerate 和 deepspeed
Write-Host "🔍 检查依赖..." -ForegroundColor Yellow
$accelerate_check = pip list | Select-String "accelerate"
$deepspeed_check = pip list | Select-String "deepspeed"

if (-not $accelerate_check) {
    Write-Host "📦 安装 accelerate..." -ForegroundColor Yellow
    pip install accelerate
}

if (-not $deepspeed_check) {
    Write-Host "📦 安装 deepspeed..." -ForegroundColor Yellow
    pip install deepspeed
}

Write-Host "✅ 依赖检查完成" -ForegroundColor Green
Write-Host ""

# 确定启动方式
if ($num_gpus -ge 2) {
    Write-Host "🔧 [多卡模式] 启用 DeepSpeed ZeRO-2（$num_gpus 卡）" -ForegroundColor Green
    Write-Host "⏱️  预期耗时: 28-32 分钟（vs 单卡 50-60 分钟）" -ForegroundColor Cyan
    Write-Host ""
    
    # 双卡启动
    Write-Host "🚀 启动命令：accelerate launch --config_file accelerate_config_zero2.yaml sudoku_grpo.py" -ForegroundColor Yellow
    accelerate launch --config_file accelerate_config_zero2.yaml sudoku_grpo.py
} else {
    Write-Host "⚠️  [单卡模式] 仅检测到 1 张 GPU" -ForegroundColor Yellow
    Write-Host "💡 建议：若有多卡可用，建议升级到多卡配置" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "🚀 启动命令：python sudoku_grpo.py" -ForegroundColor Yellow
    python sudoku_grpo.py
}

Write-Host ""
Write-Host "✨ 训练完成！" -ForegroundColor Green
