import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tqdm import tqdm  # 进度条库

# --- 配置 ---
MODEL_PATH = "./roberta-toxic-finetuned"  # 你训练好的模型路径
TEST_DATA_FILE = "test.csv"
TEST_LABEL_FILE = "test_labels.csv"
BATCH_SIZE = 32  # 显卡显存够大可以改大，比如 64 或 128

# 6个分类标签
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


class ToxicityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[item], dtype=torch.float)
        }


def evaluate():
    # 1. 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 2. 加载模型和分词器
    print("加载模型中...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    # 3. 准备数据 (这是 Jigsaw 数据集特有的处理步骤)
    print("正在读取和处理测试集...")
    df_test = pd.read_csv(TEST_DATA_FILE)
    df_labels = pd.read_csv(TEST_LABEL_FILE)

    # 通过 id 合并文本和标签
    test_data = df_test.merge(df_labels, on="id")

    # ⚠️ 关键步骤：过滤掉标签为 -1 的数据
    # 在这个数据集中，-1 表示该样本未被标记，不能用于评估
    test_data = test_data[test_data["toxic"] != -1]

    # 为了快速测试，你可以先取前 2000 条跑跑看。如果想跑全量，请注释掉下面这行。
    # test_data = test_data.head(2000)

    print(f"有效测试样本数量: {len(test_data)}")

    # 准备 Dataset 和 DataLoader
    dataset = ToxicityDataset(
        test_data["comment_text"].values,
        test_data[LABELS].values,
        tokenizer
    )
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 开始推理
    all_preds = []
    all_labels = []

    print("开始推理预测...")
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # 这里的输出是 logits，需要用 sigmoid 转成 0-1 的概率
            probs = torch.sigmoid(outputs.logits)

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. 计算指标
    print("\n" + "=" * 30)
    print("评估结果报告")
    print("=" * 30)

    # 计算 ROC-AUC (这是比赛官方指标)
    try:
        auc_score = roc_auc_score(all_labels, all_preds, average='micro')
        print(f"总体 ROC-AUC 分数: {auc_score:.4f}")
    except ValueError:
        print("样本量太少，无法计算 AUC")

    # 将概率转换为 0 或 1 (阈值 0.5) 用于计算准确率
    preds_binary = [[1 if p > 0.5 else 0 for p in row] for row in all_preds]

    print("\n详细分类报告:")
    print(classification_report(all_labels, preds_binary, target_names=LABELS, zero_division=0))


if __name__ == "__main__":
    evaluate()