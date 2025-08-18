import json
import logging
from datasets import Dataset
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
                data = json.loads(line.strip()) # 去空格
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
                # f_out.write(json.dumps(item, ensure_ascii=False) + "\n\n")
        logger.info(f"转换后的数据已保存至: {output_path}")
    
    print(texts)
    # 创建 Hugging Face Dataset
    dataset = Dataset.from_list(texts)
    logger.info(f"数据集转换完成! 共 {len(dataset)} 条样本")
    return dataset

if __name__ == "__main__":
    convert_dataset(
        raw_data_path = r"C:\Users\ywp\Desktop\QWen\Muice_Dataset\test.jsonl",
        output_path = None
    )