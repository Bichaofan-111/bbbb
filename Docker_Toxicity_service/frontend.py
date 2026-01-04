import streamlit as st
import requests
import pandas as pd
import os 

# --- Page Configuration ---
st.set_page_config(page_title="Toxicity Detection System", page_icon="🛡️")

# --- Title and Introduction ---
st.title("🛡️ Online Toxicity Detection System")
st.markdown(
    "This system utilizes a **RoBERTa (Fine-tuned)** model to identify harmful language, insults, and hate speech in user-generated text.")
st.markdown("---")

# --- Sidebar: Input Control ---
with st.sidebar:
    st.header("Test Console")
    st.write("Enter a comment below to test the model:")
    user_input = st.text_area("Input Text", height=200, placeholder="e.g., You are a stupid idiot...")

    predict_btn = st.button("Analyze Content", type="primary")

# --- Main Area: Results ---
if predict_btn and user_input:
    # 1. Call Backend API
    # 这里的逻辑是：如果在 Docker 里运行，就读取环境变量 API_URL；如果本地直接跑，就默认连 localhost
    api_url = os.getenv("API_URL", "http://127.0.0.1:8080/predict")
    #api_url = "http://backend:8080/predict"

    with st.spinner('AI is analyzing the text...'):
        try:
            response = requests.post(api_url, json={"text": user_input})

            if response.status_code == 200:
                data = response.json()
                results = data["results"]
                is_toxic = data["is_toxic"]
                process_time = data["processing_time_ms"]

                # 2. Display Core Verdict
                st.subheader("Analysis Results")
                if is_toxic:
                    st.error(f"⚠️ Warning: Toxic Content Detected (Time: {process_time}ms)")
                else:
                    st.success(f"✅ Safe: No significant toxicity detected (Time: {process_time}ms)")

                # 3. Visualization (Bar Chart)
                st.markdown("### Detailed Toxicity Scores")

                # Convert dictionary to DataFrame for plotting
                df = pd.DataFrame(list(results.items()), columns=["Category", "Score"])
                df = df.set_index("Category")

                # Draw Bar Chart (Red for toxic, Green for safe)
                st.bar_chart(df, color="#ff4b4b" if is_toxic else "#00cc00")

                # 4. Raw Data Table
                with st.expander("View Raw Score Data"):
                    st.table(df)

            else:
                st.error("❌ Backend Service Error. Please check if Docker is running.")

        except Exception as e:
            st.error(f"Cannot connect to server: {e}")
            st.info(f"Trying to connect to: {api_url}")
            st.info("Tip: Ensure you have run 'docker-compose up'")

else:
    st.info("👈 Please enter text in the sidebar and click 'Analyze Content'.")

# --- Footer ---
st.markdown("---")
st.caption("Milestone 1 Demo | Powered by FastAPI & RoBERTa | Jigsaw Dataset")