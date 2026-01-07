import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# 1. 配置参数 (来源于文档 Milestone 1)
MODEL_NAME = "roberta-base"
DATA_FILE = "train.csv" # 请确保你有这个文件，或者先用 dummy_data.csv 测试
OUTPUT_DIR = "./roberta-toxic-finetuned"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
LEARNING_RATE = 2e-5  # 文档指定的优化器参数
EPOCHS = 3  # 文档指定的训练轮数


# 2. 数据处理函数
def preprocess_data(examples):
    # 处理文本
    encoding = tokenizer(examples["comment_text"], padding="max_length", truncation=True, max_length=128)

    # 处理标签
    labels_matrix = []
    for i in range(len(examples["comment_text"])):
        # ⚠️ 关键修改：加了 float(...) 强制转为小数
        row_labels = [float(examples[label][i]) for label in LABELS]
        labels_matrix.append(row_labels)

    encoding["labels"] = labels_matrix
    return encoding

# 3. 主训练流程
if __name__ == "__main__":
    # 检测 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载 Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    # 加载数据 (这里假设你是从 CSV 读取)
    # 实际项目中建议先清洗数据，这里直接读取
    try:
        df = pd.read_csv(DATA_FILE)
        # 取少量数据测试代码是否能跑通 (若要正式训练，请注释掉下面这行)
        #df = df.head(100)
        dataset = Dataset.from_pandas(df)
    except Exception as e:
        print(f"无法读取数据文件: {e}")
        exit()

    # 预处理数据
    encoded_dataset = dataset.map(preprocess_data, batched=True)
    # 拆分训练集和验证集 (文档建议 80/10/10，这里简化为 90/10)
    split_dataset = encoded_dataset.train_test_split(test_size=0.1)

    # 初始化模型
    # problem_type="multi_label_classification" 会自动使用 Binary Cross Entropy Loss
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification"
    )
    model.to(device)

    # 设置训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=8,  # 如果显存不够，减小这个数字
        per_device_eval_batch_size=8,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=10,
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
    )

    # 开始训练
    print("开始训练...")
    trainer.train()

    # 保存最终模型
    print(f"保存模型到 {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("训练完成！")