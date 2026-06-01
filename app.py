
import streamlit as st
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Urdu NER AI System",
    page_icon="🧠",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.title {
    text-align:center;
    color:#1565C0;
    font-size:40px;
    font-weight:bold;
}
.subtitle {
    text-align:center;
    color:#555;
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# TITLE
# ==================================================
st.markdown('<p class="title">🧠 Urdu Named Entity Recognition System</p>',
            unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">XLM-RoBERTa Fine-Tuned Urdu NER Model</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ==================================================
# MODEL PATH
# REPLACE THIS WITH YOUR HF MODEL
# ==================================================
MODEL_PATH = "kiran-razaq123/urdu-ner-xlm-roberta"

# Example:
# MODEL_PATH = "kiran-razaq123/urdu-ner-xlm-roberta"

# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_PATH
    )

    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )

    return ner_pipeline

ner = load_model()

# ==================================================
# LABEL MAPPING
# ==================================================
LABEL_MAP = {
    "PER": "👤 Person",
    "LOC": "📍 Location",
    "ORG": "🏢 Organization",
    "DATE": "📅 Date",
    "TIME": "⏰ Time",
    "MISC": "🔹 Miscellaneous"
}

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("📊 Model Information")

st.sidebar.success("Model: XLM-RoBERTa")

st.sidebar.info("""
Supported Labels:

👤 Person

📍 Location

🏢 Organization

📅 Date

⏰ Time

🔹 Miscellaneous
""")

# ==================================================
# EXAMPLES
# ==================================================
st.subheader("📌 Example Sentences")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("علی کراچی گیا")

with col2:
    st.success("گوگل ایک بڑی کمپنی ہے")

with col3:
    st.success("23 اپریل کو اجلاس ہوا")

# ==================================================
# TEXT INPUT
# ==================================================
text = st.text_area(
    "✍️ Enter Urdu or English Text",
    height=180,
    placeholder="مثال: علی کراچی میں گوگل کمپنی گیا"
)

# ==================================================
# ANALYZE BUTTON
# ==================================================
if st.button("🚀 Analyze Text"):

    if text.strip():

        with st.spinner("Analyzing..."):

            results = ner(text)

        st.success("Analysis Completed Successfully ✅")

        if len(results) == 0:

            st.warning("No entities found.")

        else:

            entity_data = []

            for item in results:

                label = item["entity_group"]

                label = label.replace("B-", "")
                label = label.replace("I-", "")

                entity_data.append({
                    "Entity": item["word"],
                    "Label": LABEL_MAP.get(label, label),
                    "Confidence": round(item["score"], 4)
                })

            df = pd.DataFrame(entity_data)

            st.subheader("🔍 Extracted Entities")

            st.dataframe(
                df,
                use_container_width=True
            )

            st.subheader("📈 Label Distribution")

            st.bar_chart(
                df["Label"].value_counts()
            )

            avg_conf = df["Confidence"].mean()

            st.subheader("📊 Summary")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Entities",
                len(df)
            )

            c2.metric(
                "Avg Confidence",
                f"{avg_conf:.2f}"
            )

            c3.metric(
                "Model",
                "XLM-R"
            )

    else:

        st.error("⚠️ Please enter text first.")

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")

st.caption(
    "🧠 Urdu Named Entity Recognition System | "
    "Powered by XLM-RoBERTa"
)
```

