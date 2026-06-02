import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

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

st.title("🧠 Urdu NER System")

text = st.text_area("Enter Urdu text")

if st.button("Analyze"):
    results = ner(text)

    for r in results:
        st.success(f"{r['word']} → {r['entity_group']} ({round(r['score']*100,2)}%)")
