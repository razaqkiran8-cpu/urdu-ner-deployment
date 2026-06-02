import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# =========================
# MODEL FROM HUGGING FACE
# =========================
MODEL_NAME = "kiran-razaq123/urdu-ner-xlm-roberta"

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
# UI TITLE
# =========================
st.title("🧠 Urdu Named Entity Recognition (NER)")

st.write("Enter Urdu text and get Person, Location, Organization detection.")

# =========================
# EXAMPLES
# =========================
st.subheader("📌 Example Inputs")

examples = [
    "علی کراچی میں رہتا ہے",
    "محمد علی جناح پاکستان کے بانی ہیں",
    "لاہور ایک خوبصورت شہر ہے"
]

for i, ex in enumerate(examples):
    if st.button(f"Example {i+1}"):
        st.session_state["text"] = ex

# =========================
# INPUT BOX
# =========================
text = st.text_area("Enter Urdu Text", value=st.session_state.get("text", ""))

# =========================
# PREDICTION
# =========================
if st.button("Analyze"):
    if text.strip():
        results = ner(text)

        st.subheader("🎯 Results")

        for entity in results:
            st.success(
                f"Word: {entity['word']} | "
                f"Label: {entity['entity_group']} | "
                f"Confidence: {entity['score']:.2%}"
            )
    else:
        st.warning("Please enter text first!")

