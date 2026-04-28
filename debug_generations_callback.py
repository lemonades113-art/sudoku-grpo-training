"""
调试回调：保存GRPO训练过程中生成的12个答案
"""
import os
from transformers import TrainerCallback

LAST_COMPLETIONS = []

class DebugGenerationCallback(TrainerCallback):
    """保存每个step的生成答案到本地文件"""
    
    def __init__(self, output_dir="output/debug_generations"):
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.save_frequency = 1  # 每个step都保存
    
    def on_step_end(self, args, state, control, **kwargs):
        """在每个step结束时保存12个生成的答案"""
        if state.global_step % self.save_frequency == 0:
            self._save_completions(state.global_step)
        return control
    
    def _save_completions(self, step):
        """保存当前step的12个生成答案"""
        if not LAST_COMPLETIONS or len(LAST_COMPLETIONS) < 12:
            return
        
        step_dir = os.path.join(self.output_dir, f"step_{step:04d}")
        os.makedirs(step_dir, exist_ok=True)
        
        # 保存12个样本
        for i, completion in enumerate(LAST_COMPLETIONS[:12]):
            filename = os.path.join(step_dir, f"sample_{i+1:02d}.txt")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(completion)
        
        print(f"[DEBUG] Step {step}: 已保存12个生成样本到 {step_dir}")
        
        # 保存摘要信息
        summary_file = os.path.join(self.output_dir, "summary.txt")
        with open(summary_file, 'a', encoding='utf-8') as f:
            f.write(f"\nStep {step}:\n")
            for i, completion in enumerate(LAST_COMPLETIONS[:12]):
                # 检查是否包含CoT
                has_cot = "<think>" in completion
                has_answer = "<answer>" in completion
                f.write(f"  Sample {i+1}: {len(completion)} tokens | CoT: {has_cot} | Answer: {has_answer}\n")


def set_last_completions(completions):
    """更新全局的生成内容（从奖励函数调用）"""
    global LAST_COMPLETIONS
    LAST_COMPLETIONS = completions
