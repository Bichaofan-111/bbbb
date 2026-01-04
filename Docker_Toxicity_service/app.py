import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import time

# 初始化 FastAPI 应用
app = FastAPI(title="Toxicity Detection Microservice", version="1.0")

# --- 配置部分 ---
# 根据文档 Milestone 1 Report，模型架构为 RoBERTa-base [cite: 37]
# 注意：实际部署时，请将 MODEL_PATH 替换为你微调后的模型路径或 Hugging Face ID
MODEL_PATH = "./roberta-toxic-finetuned"
# 文档定义的6个输出标签
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# --- 全局模型加载 (启动时运行) ---
print("Loading model...")
try:
    # 这里假设模型输出维度 num_labels=6
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=len(LABELS))
    model.eval()  # 设置为评估模式
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # 实际生产中应在此处中断启动


# --- 请求数据模型 ---
class CommentRequest(BaseModel):
    text: str


class PredictionResult(BaseModel):
    label: str
    score: float


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


# 健康检查端点
@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "RoBERTa-base"}