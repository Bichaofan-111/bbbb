# 使用轻量级 Python 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 1. 复制依赖清单并安装
# (这样做利用了 Docker 缓存，如果你只改代码不改依赖，构建会很快)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. 复制项目的所有文件到容器里
# (包括 app.py, frontend.py 和 关键的 roberta-toxic-finetuned 文件夹)
COPY . .

# 暴露端口 (仅作声明)
EXPOSE 8080 8501

# 默认指令 (稍后会被 docker-compose 覆盖)
CMD ["echo", "Ready"]