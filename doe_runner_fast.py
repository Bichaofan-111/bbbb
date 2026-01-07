import pandas as pd
import torch
import gc
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import roc_auc_score
import os
import time
import numpy as np
from scipy.special import expit
import shutil

# --- 1. 定义实验配置 ---
experiments = [
    {"learning_rate": 2e-5, "per_device_train_batch_size": 16, "weight_decay": 0.01, "note": "Baseline"},
    {"learning_rate": 5e-5, "per_device_train_batch_size": 16, "weight_decay": 0.01, "note": "High LR"},
    {"learning_rate": 2e-5, "per_device_train_batch_size": 32, "weight_decay": 0.01, "note": "High BS"},
    {"learning_rate": 5e-5, "per_device_train_batch_size": 32, "weight_decay": 0.01, "note": "High BS & LR"}
]

REPETITIONS = 5
OUTPUT_FILE = "doe_fast_results.csv"
DATA_FILE = "train.csv"

# --- 2. 准备数据 ---
if not os.path.exists(DATA_FILE):
    print(f"❌ 错误: 找不到文件 {DATA_FILE}")
    exit()
else:
    print("正在加载数据...")
    # 保持 5% 数据量，如果觉得太慢，可以把 frac=0.05 改成 frac=0.01 (1%)
    raw_df = pd.read_csv(DATA_FILE).sample(frac=0.05, random_state=42)

print(f"🔥 极速模式已开启：本次实验仅使用 {len(raw_df)} 条数据")

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def preprocess(examples):
    return tokenizer(examples["comment_text"], padding="max_length", truncation=True, max_length=128)


def format_labels(examples):
    # 关键修复：强制转换为 float32
    labels_matrix = np.array([examples[l] for l in labels], dtype=np.float32).T
    return {"labels": labels_matrix.tolist()}


try:
    ds = Dataset.from_pandas(raw_df)
    ds = ds.map(preprocess, batched=True)
    ds = ds.map(format_labels, batched=True)

    cols_to_keep = ['input_ids', 'attention_mask', 'labels']
    ds = ds.remove_columns([c for c in ds.column_names if c not in cols_to_keep])
    ds.set_format("torch")

    # 打印检查
    print(f"Label 类型检查: {ds[0]['labels'].dtype} (必须是 float32)")

    ds = ds.train_test_split(test_size=0.1, seed=42)
    print("✅ 数据预处理完成。")
except Exception as e:
    print(f"❌ 数据处理失败: {e}")
    exit()


# --- 3. 辅助函数 ---
def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    probs = expit(preds)
    roc_auc = roc_auc_score(p.label_ids, probs, average="micro")
    return {"roc_auc": roc_auc}


def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


# --- 4. 开始循环运行 ---
results = []
total_runs = len(experiments) * REPETITIONS
current_run = 0
start_time_all = time.time()

print(f"🏁 开始 DOE 实验，总计 {total_runs} 次运行...")

for i, params in enumerate(experiments):
    for rep in range(REPETITIONS):
        current_run += 1
        run_name = f"{params['note']}_Rep{rep + 1}"
        output_dir = f"./doe_temp/run_{current_run}"

        print(
            f"\n[{current_run}/{total_runs}] 🚀 运行: {run_name} | LR={params['learning_rate']} | BS={params['per_device_train_batch_size']}")

        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=6,
            problem_type="multi_label_classification"
        )
        if torch.cuda.is_available():
            model.to('cuda')

        args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=params["learning_rate"],
            per_device_train_batch_size=params["per_device_train_batch_size"],
            per_device_eval_batch_size=32,
            weight_decay=params["weight_decay"],
            num_train_epochs=3,

            # --- 关键修改：开启可视化 ---
            disable_tqdm=False,  # ✅ 开启进度条
            logging_steps=50,  # ✅ 每 50 步打印一次日志
            eval_strategy="epoch",

            save_strategy="no",
            fp16=torch.cuda.is_available(),
            seed=42 + rep,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            processing_class=tokenizer,
            compute_metrics=compute_metrics
        )

        try:
            trainer.train()
            eval_res = trainer.evaluate()
            auc = eval_res["eval_roc_auc"]
            print(f"   ✅ 完成 -> ROC-AUC: {auc:.4f}")

            record = params.copy()
            record["repetition"] = rep + 1
            record["roc_auc"] = auc
            results.append(record)
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

        except Exception as e:
            print(f"   ❌ 训练出错: {e}")
            import traceback

            traceback.print_exc()

        finally:
            del model
            del trainer
            cleanup_gpu()
            if os.path.exists(output_dir):
                try:
                    shutil.rmtree(output_dir, ignore_errors=True)
                except:
                    pass

total_time = (time.time() - start_time_all) / 60
print(f"\n🎉 所有实验结束！总耗时: {total_time:.1f} 分钟")
print(f"📊 结果已保存到: {OUTPUT_FILE}")