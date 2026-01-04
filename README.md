# bbbb
Docker_Toxicity_service
🛡️ Deep Learning & SE: Toxicity Detection System (Milestone 3)

📌 Overview

This repository contains the submission for Milestone 3 of the Deep Learning and Software Engineering course.

We have developed a complete Toxicity Detection System that classifies user comments into 6 toxicity categories (toxic, severe_toxic, obscene, threat, insult, identity_hate) using a fine-tuned RoBERTa model.

The system is fully containerized using Docker and Docker Compose, ensuring it runs consistently across different operating systems (Windows, macOS, Linux) and hardware configurations.

🚀 Quick Start Guide

Prerequisites

Docker Desktop must be installed and running on your machine.

No local Python environment or GPU is required.

How to Run

Unzip the project folder.

Open a terminal (Command Prompt, PowerShell, or Terminal) inside the project folder.

Run the following command to build and start the services:

docker-compose up --build


(Note: The first run may take a few minutes to download the base image and dependencies.)

Once the terminal shows the services are running, open your web browser and access:

👉 User Interface: http://localhost:8501

📄 Backend API Docs: http://localhost:8080/docs

To stop the application, press Ctrl + C in the terminal.

📂 Project Structure

.
├── app.py                      # Backend Microservice (FastAPI)
├── frontend.py                 # Frontend User Interface (Streamlit)
├── train.py                    # Model training script (Milestone 1 & 2)
├── evaluate.py                 # Model evaluation script
├── Dockerfile                  # Container build instructions
├── docker-compose.yml          # Service orchestration configuration
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── roberta-toxic-finetuned/    # [IMPORTANT] The fine-tuned model artifacts
├── train.csv                   # Training Dataset
└── test.csv                    # Testing Dataset


🏗️ System Architecture

The application follows a Microservices Architecture composed of two containers:

Backend Container (toxic-backend):

Runs FastAPI.

Loads the RoBERTa model (fine-tuned on the Jigsaw dataset).

Exposes a REST API endpoint /predict on port 8080.

Frontend Container (toxic-frontend):

Runs Streamlit.

Provides a user-friendly interface for text input and visualization.

Communicates with the backend via the Docker network using the environment variable API_URL.

⚙️ Technical Highlights

Portability: We used torch --index-url .../cpu in requirements.txt to install the CPU-optimized version of PyTorch. This ensures the Docker image is lightweight and runs smoothly on laptops without dedicated GPUs.

Networking: Docker Compose handles the internal networking, allowing the containers to resolve each other by service name.

🔧 Troubleshooting

1. Port Conflict Error
If you see an error saying Bind for 0.0.0.0:8080 failed: port is already allocated, it means a previous container is still running in the background. Run the following command to clean up:

docker rm -f toxic-backend toxic-frontend


Then run docker-compose up again.

2. "Model not found" Error
Ensure that the folder roberta-toxic-finetuned exists in the project root and is not empty. This folder must contain the model weights (pytorch_model.bin or model.safetensors) and configuration files.

👥 Authors

Group: [Your Group Number]

Members: [List Member Names Here]
