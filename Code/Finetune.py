import os
import json
import logging
import torch
import warnings
# 忽略特定的警告
warnings.filterwarnings("ignore", category=UserWarning, message="1Torch was not compiled with flash attention.")

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# =============== 配置区域 ===============
# 路径、参数
MODEL_NAME       = "./QWen1_5_0_5B"               # 模型名称/路径
RAW_DATA_PATH    = "./Muice_Dataset/train.jsonl"  # 原始数据集路径
RAW_TEST_PATH    = "./Muice_Dataset/test.jsonl"   # 原始测试集路径
OUTPUT_DIR       = "./qwen_finetuned/30"          # 微调后模型保存路径
MAX_SEQ_LENGTH   = 1024                           # 最大序列长度（Qwen1.5-0.5B 支持 32768，但微调建议 2048）
LORA_R           = 8                              # LoRA 秩
LORA_ALPHA       = 32                             # LoRA alpha
LORA_DROPOUT     = 0.1                            # LoRA dropout
BATCH_SIZE       = 2                              # 每设备批次大小
GRAD_ACCUM_STEPS = 8                              # 梯度累积步数
LEARNING_RATE    = 2e-5                           # 学习率
NUM_EPOCHS       = 30                             # 训练轮数
FP16             = True                           # 是否使用 FP16（如果 GPU 支持）
# ======================================

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def convert_dataset(raw_data_path: str, output_path: str = None) -> Dataset:
    """
    将原始数据集转换为 Qwen1.5 格式并加载为 Hugging Face Dataset
    :param raw_data_path: 原始 JSONL 文件路径
    :param output_path: (可选) 保存转换后的文本文件
    :return: Hugging Face Dataset 对象
    """
    logger.info(f"开始转换数据集: {raw_data_path}")
    
    texts = []
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # 构建 Qwen1.5 格式的对话模板
                text = f"<|im_start|>system\n{data['system']}<|im_end|>\n"
                
                for turn in data['conversation']:
                    text += f"<|im_start|>user\n{turn['human']}<|im_end|>\n"
                    text += f"<|im_start|>assistant\n{turn['assistant']}<|im_end|>\n"
                
                texts.append({"text": text.strip()})
            except Exception as e:
                logger.warning(f"跳过无效数据行: {line.strip()}, 错误: {str(e)}")
    
    # 保存转换后的文本（可选）
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for item in texts:
                f_out.write(item["text"] + "\n\n")
        logger.info(f"转换后的数据已保存至: {output_path}")
    
    # 创建 Hugging Face Dataset
    dataset = Dataset.from_list(texts)
    logger.info(f"数据集转换完成! 共 {len(dataset)} 条样本")
    return dataset

def load_and_prepare_model(model_name: str, max_seq_length: int) -> tuple:
    """
    加载模型并准备 LoRA 微调
    :param model_name: 模型名称或路径
    :param max_seq_length: 最大序列长度
    :return: (tokenizer, model)
    """
    logger.info(f"加载模型: {model_name}")
    
    # 配置量化（可选，8GB 显存可关闭） -- 需注意，bitsandbytes 官方不支持windows，故此处仅供参考
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit              = True,             # 启用4bit量化
    #     bnb_4bit_quant_type       = "nf4",            # 量化类型
    #     bnb_4bit_compute_dtype    = torch.bfloat16,   # 计算数据类型
    #     bnb_4bit_use_double_quant = True              # 使用双量化
    # )
    
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code = True,
        padding_side      = "right",  # 必须右填充
        use_fast          = False     # 是否使用快速实现的分词器
    )
    
    # 添加缺失的特殊 token
    tokenizer.pad_token = tokenizer.pad_token
    logger.info(f"Special tokens: {tokenizer.special_tokens_map}")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config,  # 4-bit 量化（可选）
        trust_remote_code   = True,
        device_map          = "auto",
        torch_dtype         = torch.bfloat16,  # 添加半精度支持
        attn_implementation = "sdpa"           # 使用更快的注意力实现
    )
    
    # 准备模型进行 k-bit 训练
    model = prepare_model_for_kbit_training(model)
    
    # 配置 LoRA
    lora_config = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        task_type      = "CAUSAL_LM"
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数量
    
    return tokenizer, model

def main():
    # 1. 数据预处理
    dataset = convert_dataset(
        raw_data_path = RAW_DATA_PATH,
        output_path   = None           # 保存转换后的文本（可选）
    )

    data_test = convert_dataset(
        raw_data_path = RAW_TEST_PATH,
        output_path   = None
    )
    
    # 2. 加载并准备模型
    tokenizer, model = load_and_prepare_model(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH
    )
    
    # 3. 数据处理函数
    def tokenize_function(examples):
        # 将文本转换为模型输入
        tokenized = tokenizer(
            examples["text"],                          # 转换词元ID
            truncation                = True,          # 截断，序列过长进行截断
            max_length                = MAX_SEQ_LENGTH,
            # padding                 = False,         # 动态填充在 DataCollator 中处理
            padding                   = 'max_length',  # 填充，动态填充在 DataCollator 中处理
            return_overflowing_tokens = False
        )
        
        # 为 labels 创建副本
        # tokenized["labels"] = tokenized["input_ids"].copy()
        # print("tokenized:", tokenized)

        # 新增 -- 掩码 assistant 部分
        # 创建标签 - 只计算assistant回复的损失
        labels = tokenized["input_ids"].copy()

        # 获取特殊标记的ID
        im_start_id  = tokenizer.convert_tokens_to_ids('<|im_start|>')   # 151644
        im_end_id    = tokenizer.convert_tokens_to_ids('<|im_end|>')     # 151645
        assistant_id = tokenizer.convert_tokens_to_ids('assistant')      # 77091
        
        # 将非assistant部分的标签设为-100（忽略）
        for i in range(len(labels)):
            input_ids      = tokenized["input_ids"][i]
            current_labels = [-100] * len(input_ids)   # 初始化为全部忽略
            
            # 找到所有assistant部分
            idx = 0
            while idx < len(input_ids):
                # 查找<|im_start|>assistant模式
                if (idx < len(input_ids) - 1 and 
                    input_ids[idx] == im_start_id and 
                    input_ids[idx + 1] == assistant_id):
                    
                    # 找到assistant开始位置
                    start_idx = idx
                    idx += 2  # 跳过<|im_start|>和assistant标记
                    
                    # 查找对应的结束标记<|im_end|>
                    end_idx = None
                    for j in range(idx, len(input_ids)):
                        if input_ids[j] == im_end_id:
                            end_idx = j
                            break
                    
                    # 如果找到了结束标记，保留assistant内容部分
                    if end_idx is not None:
                        # 保留从assistant内容开始到结束标记前的内容
                        content_start = idx
                        content_end   = end_idx
                        
                        # 将assistant内容部分复制到标签中
                        for k in range(content_start, content_end):
                            current_labels[k] = input_ids[k]
                        
                        # 跳过已处理的内容
                        idx = end_idx + 1  # 跳过结束标记
                    else:
                        # 没有找到结束标记，保留到序列末尾
                        for k in range(idx, len(input_ids)):
                            current_labels[k] = input_ids[k]
                        break
                else:
                    idx += 1
            
            labels[i] = current_labels
        
        # 确保填充部分也被掩码
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if tokenized["input_ids"][i][j] == tokenizer.pad_token_id:
                    labels[i][j] = -100
        
        tokenized["labels"] = labels

        return tokenized
    
    # 应用 tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched        = True,
        num_proc       = 4,
        remove_columns = dataset.column_names,
        desc           = "Train: Tokenizing and chunking..."
    )

    tokenized_data_test = data_test.map(
        tokenize_function,
        batched        = True,
        num_proc       = 4,
        remove_columns = data_test.column_names,
        desc           = "Test: Tokenizing and chunking..."
    )
    
    # 4. 配置训练参数
    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        overwrite_output_dir        = True,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        learning_rate               = LEARNING_RATE,
        weight_decay                = 0.01,
        logging_dir                 = f"{OUTPUT_DIR}/logs",
        logging_strategy            = "steps",
        logging_steps               = 10,
        save_strategy               = "epoch",
        # eval_strategy             = "no",
        eval_strategy               = "epoch",             # 每轮后使用测试集进行评估
        fp16                        = FP16,
        bf16                        = False,
        optim                       = "paged_adamw_8bit",  # 配合 4-bit 量化
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.03,
        report_to                   = "tensorboard",
        load_best_model_at_end      = False,
        ddp_find_unused_parameters  = False,
        group_by_length             = True,                # 动态填充
        length_column_name          = "length"             # 用于 group_by_length
    )
    
    # 5. 配置 DataCollator（关键：自动掩码非 assistant 部分）
    """
        mlm = true, 掩码语言模型：
        例如：输入 -> “我昨天去了[MASK]店，买了一个汉堡。”
        模型目标 -> 预测 [MASK] 位置是 “快餐”。
    """
    data_collator = DataCollatorForLanguageModeling(
        tokenizer          = tokenizer,
        mlm                = False,  # 是否使用masked language model（掩码语言模型）， False（因果语言模型）
        pad_to_multiple_of = 8       # 优化 GPU 计算
    )
    
    # 6. 创建 Trainer
    trainer = Trainer(
        model =  model,
        args  = training_args,
        train_dataset = tokenized_dataset,
        eval_dataset  = tokenized_data_test,
        data_collator = data_collator
    )
    
    # 7. 开始训练
    logger.info("✨ 开始微调训练...")
    train_result = trainer.train()
    
    # 8. 保存最终模型
    logger.info(f"✅ 训练完成! 保存模型到 {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 9. 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # 10. 验证模型（简单测试）
    logger.info("\n🔍 进行模型验证测试...")
    test_prompt = """<|im_start|>system
                    你是一个名为沐雪的AI女孩子<|im_end|>
                    <|im_start|>user
                    沐雪的功能是什么？<|im_end|>
                    <|im_start|>assistant
                    """
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens = 100,
        do_sample      = True,
        temperature    = 0.7,
        top_p          = 0.9,
        pad_token_id   = tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    logger.info(f"测试输入:\n{test_prompt}")
    logger.info(f"模型输出:\n{response[len(test_prompt):]}")

if __name__ == "__main__":
    # 检查 CUDA
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA 可用! 设备: {torch.cuda.get_device_name(0)}")
        logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logger.warning("❌ CUDA 不可用! 将使用 CPU（不推荐）")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 运行主函数
    main()