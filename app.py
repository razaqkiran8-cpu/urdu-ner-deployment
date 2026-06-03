import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Urdu NER App", layout="wide")

# =========================
# HEADER (COLORED + BOLD)
# =========================
st.markdown(
    """
    <h1 style='text-align:center; color:white; background-color:#4B0082; padding:15px; border-radius:10px;'>
    🧠 URDU NAMED ENTITY RECOGNITION (NER)
    </h1>
    """,
    unsafe_allow_html=True
)

# =========================
# MODEL INFO
# =========================
st.markdown("### 🤖 Model: XLM-RoBERTa")
st.markdown("### 📊 Task: Named Entity Recognition (Person, Location, Organization, Date, Misc)")

MODEL_NAME = "kiran-razaq123/urdu-ner-xlm-roberta"

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )

ner = load_model()

# =========================
# EXAMPLES
# =========================
st.markdown("## 📌 Example Inputs")

examples = [
    "علی کراچی میں گوگل پاکستان میں کام کرتا ہے",
    "محمد احمد 12 مارچ 2024 کو لاہور گیا",
    "فاطمہ اسلام آباد میں مائیکروسافٹ میں کام کرتی ہے"
]

for i, ex in enumerate(examples):
    if st.button(f"Example {i+1}"):
        st.session_state["text"] = ex

# =========================
# INPUT BOX
# =========================
st.markdown("## ✍️ Enter Text")

text = st.text_area("Urdu Text Input", value=st.session_state.get("text", ""))

# =========================
# PREDICTION
# =========================
if st.button("🚀 Analyze"):
    if text.strip():

        results = ner(text)

        st.markdown("## 🎯 Results")

        if len(results) == 0:
            st.warning("No entities detected.")
        else:
            for r in results:
                st.success(
                    f"**Word:** {r['word']}  \n"
                    f"**Entity:** {r['entity_group']}  \n"
                    f"**Confidence:** {round(r['score']*100, 2)}%"
                )

    else:
        st.warning("Please enter some text first!")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🚀 Built with Streamlit + XLM-RoBERTa for Urdu NER")
