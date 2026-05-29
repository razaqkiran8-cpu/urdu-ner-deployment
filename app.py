import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ---------------- PAGE CONFIG (MUST BE FIRST) ----------------
st.set_page_config(page_title="Urdu NER AI System", layout="centered")

st.title("🧠 Urdu Named Entity Recognition (NER)")
st.write("Enter Urdu or English text and extract entities using AI model")

# ---------------- MODEL LOADING (SAFE FOR DEPLOYMENT) ----------------
@st.cache_resource
def load_model():
    model_name = "Davlan/xlm-roberta-base-ner-hrl"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )

ner_pipeline = load_model()

# ---------------- 9+ LABEL MAPPING ----------------
def to_urdu(label):
    return {
        "PER": "👤 شخص (Person)",
        "LOC": "📍 مقام (Location)",
        "ORG": "🏢 ادارہ (Organization)",
        "MISC": "🧩 دیگر (Misc)",
        "DATE": "📅 تاریخ (Date)",
        "TIME": "⏰ وقت (Time)",
        "MONEY": "💰 رقم (Money)",
        "PERCENT": "📊 فیصد (Percent)",
        "PRODUCT": "📦 پروڈکٹ (Product)",
        "EVENT": "🎉 واقعہ (Event)",
        "GPE": "🌍 ملک/شہر (Geo-Political)",
        "FAC": "🏛 عمارت/جگہ (Facility)"
    }.get(label, label)

# ---------------- INPUT ----------------
text = st.text_area(
    "✍️ Enter Urdu or English text:",
    height=150,
    placeholder="مثال: علی کراچی میں گوگل کمپنی گیا اور 5000 روپے خرچ کیے"
)

# ---------------- ANALYZE BUTTON ----------------
if st.button("🚀 Analyze Text"):

    if text.strip():

        with st.spinner("Analyzing... ⏳"):
            results = ner_pipeline(text)

        st.success("Analysis Complete ✅")

        if len(results) == 0:
            st.warning("No entities found")

        else:
            st.subheader("🔎 Extracted Entities")

            for r in results:

                col1, col2, col3 = st.columns([3, 3, 2])

                with col1:
                    st.markdown(f"**{r['word']}**")

                with col2:
                    st.markdown(f"`{to_urdu(r['entity_group'])}`")

                with col3:
                    st.markdown(f"🎯 {round(r['score'], 3)}")

    else:
        st.error("⚠️ Please enter text first!")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🧠 Powered by XLM-Roberta | Urdu NER AI System")
