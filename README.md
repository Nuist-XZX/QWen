# QWen
# QWen1.5-0.5B沐雪数据集微调
　　QWen1.5-0.5B: https://huggingface.co/Qwen/Qwen1.5-0.5B/tree/main  
　　Muice-Dataset: https://modelscope.cn/datasets/Moemuu/Muice-Dataset/files  

# 目录
```
├── Code
│   ├── chat.py                   # 与模型交互
│   ├── Finetune.py               # 微调脚本（带掩码和验证）
│   ├── Muice_Fine_tuning.py      # 简易版微调脚本（无掩码和验证）
│   ├── pre_data.py               # 保存预处理数据
│   └── test.py                   # 测试脚本
├── Muice_Dataset                 # 原始数据集
│   └── ...
├── Pre_dataset                   # 保存预处理数据集
│   └── ...
├── qwen_finetuned                # 微调后的模型
│   └── ...
├── README.md
├── loss.txt                      # 无掩码损失
└── mask_loss.txt                 # 带掩码损失
```