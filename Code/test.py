# 用测试集验证 

import json
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch

# 配置参数
MODEL_NAME = "Qwen/Qwen1.5-0.5B"
TRAIN_FILE = "train.jsonl"
TEST_FILE = "test.jsonl"
OUTPUT_DIR = "./qwen_finetuned"
MAX_LENGTH = 1024  # 根据GPU内存调整
BATCH_SIZE = 4     # 根据GPU内存调整
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 5e-5
EPOCHS = 3

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="right",
    use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充token

# 特殊标记
SYS_START = "<|im_start|>system"
USR_START = "<|im_start|>user"
ASST_START = "<|im_start|>assistant"
END_TOKEN = "<|im_end|>"

def format_conversation(entry):
    """将JSONL条目转换为ChatML格式"""
    system_msg = entry.get("system", "你是一个乐于助人的AI助手")
    conversation = entry["conversation"]
    
    # 添加系统消息
    lines = [f"{SYS_START}\n{system_msg}{END_TOKEN}"]
    
    # 添加对话轮次
    for turn in conversation:
        if "human" in turn:
            lines.append(f"{USR_START}\n{turn['human']}{END_TOKEN}")
        if "assistant" in turn:
            lines.append(f"{ASST_START}\n{turn['assistant']}{END_TOKEN}")
    
    return "\n".join(lines)

def process_dataset(file_path):
    """处理JSONL数据集"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            formatted_text = format_conversation(entry)
            data.append({"text": formatted_text})
    return Dataset.from_list(data)

# 加载并处理数据集
train_dataset = process_dataset(TRAIN_FILE)
test_dataset = process_dataset(TEST_FILE)

print(f"训练样本数: {len(train_dataset)}")
print(f"测试样本数: {len(test_dataset)}")
print("\n样例格式:")
print(train_dataset[0]["text"][:200] + "...")

# 数据编码函数
def tokenize_function(examples):
    """分词函数"""
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    
    # 创建标签 - 只计算assistant回复的损失
    labels = tokenized["input_ids"].clone()
    
    # 找到所有assistant开始标记的位置
    assistant_start_token_id = tokenizer.convert_tokens_to_ids(ASST_START)
    
    # 将非assistant部分的标签设为-100（忽略）
    for i in range(len(labels)):
        # 找到所有assistant开始位置
        start_indices = (tokenized["input_ids"][i] == assistant_start_token_id).nonzero(as_tuple=True)[0]
        
        # 对于每个assistant部分，只保留内容部分的损失
        for start_idx in start_indices:
            # 找到下一个END_TOKEN的位置
            end_idx = (tokenized["input_ids"][i][start_idx:] == tokenizer.eos_token_id).nonzero()
            if len(end_idx) > 0:
                end_idx = start_idx + end_idx[0]
                
                # 将assistant开始标记之后到结束标记之前的内容设为有效标签
                content_start = start_idx + 1  # 跳过assistant开始标记
                if content_start < end_idx:
                    labels[i][content_start:end_idx] = tokenized["input_ids"][i][content_start:end_idx]
            
            # 将assistant开始标记本身设为忽略
            labels[i][start_idx] = -100
    
    # 确保所有非标签部分设为-100
    labels[labels == tokenizer.pad_token_id] = -100
    tokenized["labels"] = labels
    
    return tokenized

# 应用分词
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    batch_size=100
)

tokenized_test = test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    batch_size=100
)

# 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    device_map="auto"
)
model.gradient_checkpointing_enable()  # 减少显存使用

# 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none",
    fp16=torch.cuda.is_available(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
)

# 开始训练
print("开始训练...")
train_result = trainer.train()

# 保存最终模型
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"训练完成! 模型已保存到 {OUTPUT_DIR}")

# 保存训练指标
metrics = train_result.metrics
metrics["train_samples"] = len(tokenized_train)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)