import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Urdu NER AI System",
    page_icon="🧠",
    layout="wide"
)

# ================= UI DESIGN =================
st.markdown("""
<style>
.title {
    font-size: 42px;
    text-align: center;
    color: #1f4e79;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #555;
    font-size: 18px;
}

.card {
    background: #ffffff;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="title">🧠 Urdu Named Entity Recognition System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">XLM-RoBERTa Based AI Model (6 Entity Types)</div>', unsafe_allow_html=True)

st.write("---")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model_path =  "xlm_roberta_urdu_ner"  # 👈 your trained model folder

    tokenizer = AutoTokenizer.from_pretrained( "xlm_roberta_urdu_ner")
    model = AutoModelForTokenClassification.from_pretrained( "xlm_roberta_urdu_ner")

    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )

ner = load_model()

# ================= LABEL MAP (6 LABELS) =================
label_map = {
    "PER": "👤 Person",
    "LOC": "📍 Location",
    "ORG": "🏢 Organization",
    "DATE": "📅 Date",
    "TIME": "⏰ Time",
    "MISC": "🧩 Misc"
}

# ================= EXAMPLES =================
st.subheader("📌 Example Sentences")

col1, col2, col3 = st.columns(3)

col1.info("علی کراچی گیا۔")
col2.info("گوگل ایک بڑی کمپنی ہے۔")
col3.info("23 اپریل کو 3 بجے اجلاس ہوا۔")

# ================= INPUT =================
text = st.text_area("✍️ Enter Urdu or English Text", height=150)

# ================= PREDICTION =================
def predict(text):

    output = ner(text)

    results = []

    for item in output:

        label = item["entity_group"].replace("B-", "").replace("I-", "")

        results.append({
            "Word": item["word"],
            "Label": label_map.get(label, label),
            "Confidence": round(item["score"], 3)
        })

    return results

# ================= ANALYZE BUTTON =================
if st.button("🚀 Analyze Text"):

    if text.strip():

        with st.spinner("Analyzing text..."):

            results = predict(text)

        st.success("Analysis Completed ✅")

        df = pd.DataFrame(results)

        # ================= TABLE =================
        st.subheader("🔎 Extracted Entities")
        st.dataframe(df, use_container_width=True)

        # ================= CHART =================
        st.subheader("📊 Entity Distribution")
        st.bar_chart(df["Label"].value_counts())

        # ================= SUMMARY =================
        st.subheader("📌 Summary")

        st.info(f"""
        Total Entities Found: {len(df)}
        Model: XLM-RoBERTa
        Labels: 6 (PER, LOC, ORG, DATE, TIME, MISC)
        """)

    else:
        st.error("⚠️ Please enter some text first")

# ================= SIDEBAR =================
st.sidebar.title("📊 Model Info")

st.sidebar.success("XLM-RoBERTa Urdu NER")

st.sidebar.write("""
### Supported Labels (6)
👤 Person  
📍 Location  
🏢 Organization  
📅 Date  
⏰ Time  
🧩 Misc  
""")

# ================= FOOTER =================
st.write("---")
st.caption("🧠 Urdu NER System | XLM-RoBERTa | FYP Project")
