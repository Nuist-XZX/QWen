# 将数据集转换成 Qwen1.5 格式并保存为txt文件
import json
from datasets import Dataset

def convert_dataset(raw_data_path: str, output_path: str = None) -> Dataset:
    texts = []
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip()) # 去空格
                # 构建 Qwen1.5 格式的对话模板
                text = f"<|im_start|>system\n{data['system']}<|im_end|>\n"
                
                for turn in data['conversation']:
                    text += f"<|im_start|>user\n{turn['human']}<|im_end|>\n"
                    text += f"<|im_start|>assistant\n{turn['assistant']}<|im_end|>\n"
                
                texts.append({"text": text.strip()})
            except Exception as e:
                print(f"跳过无效数据行: {line.strip()}, 错误: {str(e)}")
    
    # 保存转换后的文本
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for item in texts:
                f_out.write(item["text"] + "\n\n")
    
    # 创建 Hugging Face Dataset
    dataset = Dataset.from_list(texts)
    return dataset

if __name__ == "__main__":
    convert_dataset(
        raw_data_path = "./Muice_Dataset/train.jsonl",
        output_path = "./Pre_dataset"
    )