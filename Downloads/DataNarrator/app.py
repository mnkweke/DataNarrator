import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

st.set_page_config(page_title="DataNarrator", layout="centered")
st.title("📊 DataNarrator")
st.markdown("Upload a CSV or Excel file to preview your data and get AI-generated business insights.")

groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


@st.cache_data
def load_file(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "csv":
        return pd.read_csv(uploaded_file)
    elif ext in ("xls", "xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None


def build_dataset_summary(df: pd.DataFrame) -> str:
    desc = df.describe(include="all", percentiles=[]).round(2).to_string()

    cols = []
    for col in df.columns:
        dtype = df[col].dtype
        nulls = df[col].isna().sum()
        uniques = df[col].nunique()
        sample = df[col].dropna().iloc[0] if df[col].nunique() > 0 else "N/A"
        cols.append(f"  - {col} (dtype={dtype}, nulls={nulls}, unique={uniques}, e.g. {sample})")

    col_info = "\n".join(cols)

    return f"## Columns\n{col_info}\n\n## Descriptive Statistics\n{desc}"


def generate_insights(summary: str) -> str:
    prompt = f"""
You are a senior data analyst.

Analyze the dataset summary below and produce:

1. Key business insights (max 5 bullet points)
2. Revenue or performance trends (if applicable)
3. Risks or anomalies in the data
4. One practical recommendation the business can act on immediately

Rules:
- Be specific, not generic
- Use numbers when available
- Avoid long paragraphs
- Think like a consultant writing for a manager

Dataset summary:
{summary}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
    )

    return response.choices[0].message.content


def ask_question(summary: str, question: str) -> str:
    prompt = f"""
You are a senior data analyst.

Answer the user's question using ONLY the dataset summary below.

Dataset summary:
{summary}

User question:
{question}

Rules:
- Be specific and data-driven
- Do not hallucinate missing data
- Keep answers short and clear
- Think like a business consultant
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=800,
    )

    return response.choices[0].message.content


uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = load_file(uploaded_file)

    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Quick Stats")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Cells", int(df.isna().sum().sum()))
        col4.metric("Duplicates", df.duplicated().sum())

        summary = build_dataset_summary(df)

        with st.expander("Show dataset summary"):
            st.text(summary)

        if st.button("Generate Insights", type="primary"):
            if not groq_api_key:
                st.warning("GROQ_API_KEY is not set.")
            else:
                with st.spinner("Generating insights..."):
                    insights = generate_insights(summary)

                    st.subheader("AI-Generated Insights")
                    st.markdown(insights)

        st.divider()

        # ---------------- CHAT MODE ----------------
        st.subheader("💬 Ask a question about your data")

        question = st.text_input("Ask something about your dataset")

        if question:
            with st.spinner("Thinking..."):
                try:
                    answer = ask_question(summary, question)

                    st.markdown("### Answer")
                    st.write(answer)

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer
                    })

                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.chat_history:
            st.subheader("🧠 Chat History")

            for chat in reversed(st.session_state.chat_history):
                st.markdown(f"**You:** {chat['question']}")
                st.markdown(f"**AI:** {chat['answer']}")
                st.markdown("---")