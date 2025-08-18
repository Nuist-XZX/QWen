#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
与微调后的 Qwen1.5-0.5B 模型进行交互对话
支持多轮对话和角色扮演
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, StoppingCriteria, StoppingCriteriaList

from termcolor import colored

# =============== 配置区域 ===============
MODEL_DIR = "./qwen_finetuned"  # 您保存微调模型的目录
MAX_NEW_TOKENS = 512  # 生成回复的最大长度
TEMPERATURE = 0.7     # 创造性（0.1-1.0，值越大越有创意）
TOP_P = 0.9           # 核采样（0.0-1.0）
REPETITION_PENALTY = 1.2  # 重复惩罚
SYSTEM_PROMPT = "你是一个名为沐雪的AI女孩子"  # 角色设定（可修改）
# ======================================

class QwenStoppingCriteria(StoppingCriteria):
    """自定义停止条件：检测 <|im_end|> 标记"""
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        # 检查最后生成的 token 是否是 <|im_end|>
        return input_ids[0][-1] == self.stop_token_id

def load_model(model_dir):
    """加载微调后的模型和 tokenizer"""
    print(colored("🔍 正在加载模型...", "yellow"))
    
    if not os.path.exists(model_dir):
        print(colored(f"❌ 错误：模型目录不存在: {model_dir}", "red"))
        print(colored("请检查 MODEL_DIR 配置是否正确", "red"))
        sys.exit(1)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 确保 pad_token 存在
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 获取 <|im_end|> 的 token ID
        im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        
        print(colored(f"✅ 模型加载成功! 设备: {next(model.parameters()).device}", "green"))
        return tokenizer, model, im_end_id
    
    except Exception as e:
        print(colored(f"❌ 模型加载失败: {str(e)}", "red"))
        print(colored("可能原因：", "red"))
        print(colored("1. 模型目录不包含有效的微调模型", "red"))
        print(colored("2. 缺少必要的依赖库", "red"))
        print(colored("3. 显存不足", "red"))
        sys.exit(1)

def format_chat_history(chat_history):
    """将对话历史格式化为 Qwen 所需格式"""
    formatted = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    
    for role, message in chat_history:
        formatted += f"<|im_start|>{role}\n{message}<|im_end|>\n"
    
    formatted += "<|im_start|>assistant\n"
    return formatted

def generate_response(pipe, chat_history, im_end_id):
    """生成模型回复"""
    # 构建输入
    prompt = format_chat_history(chat_history)
    
    # 生成配置
    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "do_sample": True,
        "repetition_penalty": REPETITION_PENALTY,
        "pad_token_id": pipe.tokenizer.eos_token_id,
        "stopping_criteria": StoppingCriteriaList([QwenStoppingCriteria(im_end_id)]),
    }
    
    # 生成回复
    start_time = time.time()
    outputs = pipe(
        prompt,
        **generation_kwargs
    )
    response_time = time.time() - start_time
    
    # 提取 assistant 回复（去除 prompt 和 <|im_end|>）
    full_response = outputs[0]['generated_text']
    assistant_response = full_response[len(prompt):].replace("<|im_end|>", "").strip()
    
    # 打印性能信息
    token_count = len(pipe.tokenizer.encode(assistant_response))
    print(colored(f"\n⏱️  生成 {token_count} 个 tokens | 耗时: {response_time:.2f} 秒", "cyan"))
    
    return assistant_response

def print_header():
    """打印欢迎头部"""
    print(colored("="*50, "blue"))
    print(colored("✨ 沐雪 AI 对话系统 ✨", "magenta", attrs=["bold"]))
    print(colored("="*50, "blue"))
    print(colored("角色设定:", "yellow"), SYSTEM_PROMPT)
    print(colored("输入 'exit' 退出, 'clear' 清空对话历史", "cyan"))
    print(colored("="*50, "blue") + "\n")

def main():
    # 检查依赖
    try:
        import termcolor
    except ImportError:
        print(colored("⚠️ 请先安装 termcolor: pip install termcolor", "yellow"))
        sys.exit(1)
    
    # 加载模型
    tokenizer, model, im_end_id = load_model(MODEL_DIR)
    
    # 创建 pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    
    # 初始化对话历史
    chat_history = []
    
    # 打印欢迎信息
    print_header()
    
    # 对话主循环
    while True:
        try:
            # 获取用户输入
            user_input = input(colored("👤 你: ", "green")).strip()
            
            # 处理特殊命令
            if user_input.lower() == 'exit':
                print(colored("\n👋 再见！希望下次还能和你聊天～", "magenta"))
                break
            elif user_input.lower() == 'clear':
                chat_history = []
                print(colored("\n🧹 对话历史已清空", "yellow"))
                continue
            elif not user_input:
                continue
            
            # 添加到对话历史
            chat_history.append(("user", user_input))
            
            # 生成回复
            print(colored("🤖 沐雪:", "blue"), end=" ", flush=True)
            response = generate_response(pipe, chat_history, im_end_id)
            
            # 显示回复（逐字打印效果）
            for char in response:
                print(char, end="", flush=True)
                time.sleep(0.01)
            print("\n")
            
            # 保存到对话历史
            chat_history.append(("assistant", response))
            
        except KeyboardInterrupt:
            print("\n" + colored("\n⚠️ 按下 Ctrl+C 两次可强制退出", "yellow"))
        except Exception as e:
            print(colored(f"\n❌ 对话出错: {str(e)}", "red"))
            print(colored("尝试清空对话历史或重启", "red"))

if __name__ == "__main__":
    main()