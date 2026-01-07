import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# 推荐使用 AutoTokenizer 和 AutoModel，通用性更好，能自动适配 HF 上的配置
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import os

# 初始化 FastAPI 应用
app = FastAPI(title="Toxicity Detection Microservice", version="1.0")

# --- 配置部分 ---
# 【关键修改】将路径改为 Hugging Face 上的模型 ID
# 格式为: "用户名/仓库名"
# 如果你的仓库名不一样，请修改下面这一行
MODEL_PATH = "bibibi111111/roberta-toxic-finetuned"

# 文档定义的6个输出标签
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# --- 全局模型加载 (启动时运行) ---
print(f"Loading model from {MODEL_PATH}...")
try:
    # 使用 AutoTokenizer/AutoModel 可以自动读取 HF 上的 config.json
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # num_labels 会自动从 config.json 读取，但为了保险起见显式传入
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=len(LABELS))

    model.eval()  # 设置为评估模式
    print("Model loaded successfully from Hugging Face!")

except OSError as e:
    print(f"Error: 无法从 Hugging Face 下载或加载模型。")
    print(f"请检查: 1. 你的 Hugging Face 仓库名是否正确: {MODEL_PATH}")
    print(f"       2. 仓库是否设为 Public (公开)")
    print(f"详细错误: {e}")
    # 在生产环境中，这里应该抛出异常阻止服务启动
    # raise e
except Exception as e:
    print(f"Unexpected error loading model: {e}")


# --- 请求数据模型 ---
class CommentRequest(BaseModel):
    text: str


class ToxicityResponse(BaseModel):
    results: dict[str, float]
    is_toxic: bool
    processing_time_ms: float


# --- 推理接口 ---
@app.post("/predict", response_model=ToxicityResponse)
async def predict(request: CommentRequest):
    start_time = time.time()

    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # 1. 数据预处理 (Tokenization)
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        # 2. 模型推理
        with torch.no_grad():
            outputs = model(**inputs)

        # 3. 处理输出 (Logits -> Sigmoid -> Probabilities)
        # 使用 Sigmoid 因为这是多标签分类任务 (Multi-label classification)
        scores = torch.sigmoid(outputs.logits).squeeze().tolist()

        # 如果只有一条输入，squeeze可能会变成标量，需处理
        if isinstance(scores, float):
            scores = [scores]

        # 4. 结果映射
        result_dict = {label: score for label, score in zip(LABELS, scores)}

        # 简单的判定逻辑：只要任何一个标签分数 > 0.5 即视为有毒
        # 文档提到有人工审核阈值 (Human-in-the-loop threshold)，可在此处扩展 [cite: 17]
        is_toxic = any(score > 0.5 for score in scores)

        process_time = (time.time() - start_time) * 1000

        return {
            "results": result_dict,
            "is_toxic": is_toxic,
            "processing_time_ms": round(process_time, 2)
        }

    except NameError:
        raise HTTPException(status_code=500, detail="Model not initialized properly")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# 健康检查端点
@app.get("/health")
def health_check():
    # 简单的状态检查
    model_status = "loaded" if 'model' in globals() else "failed"
    return {"status": "healthy", "model_source": MODEL_PATH, "model_status": model_status}