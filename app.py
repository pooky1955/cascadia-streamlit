import streamlit as st
from util_huggingface import convert_to_df
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tokenizers import Tokenizer
import transformers

TEN_MINUTES = 10 * 60
MODEL_PATH = "abhibisht89/spanbert-large-cased-finetuned-ade_corpus_v2"

@st.cache(allow_output_mutation=True,suppress_st_warning=True, show_spinner=False,max_entries=2,ttl=TEN_MINUTES,hash_funcs={Tokenizer : id})
def load_ner(use_gpu=False, model_path=MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    return pipeline("ner", model=model, tokenizer=tokenizer)

def main():
    st.write("<h1 style='text-align : center'> Cascadia: Detecting Drug Prescribing Cascades Through Deep Learning </h1>",
             unsafe_allow_html=True)
    st.write("<p style='text-align : center; font-style : italic;'> A demo for the MRSTF 2022 </p>",
             unsafe_allow_html=True)
    st.write("""
            <p style='text-align : justify'>
            Hello! This is a simple website to demonstrate some of the core features of Cascadia. Here, you can enter sentences and the model will output which drugs and ADEs (adverse drug events) it detected along with whether the sentence is ADE-related or not in the first place!
            </p>
            """, unsafe_allow_html=True)
    model = load_ner()
    RAW_TEXT_PLACEHOLDER = "The patient has been taking Advil but experienced stomach ache. He was relieved with omeprazole."
    raw_text = st.text_area("", RAW_TEXT_PLACEHOLDER)

    if len(raw_text) < 10:
        st.info("Please make sure to write at least more than 10 characters.")
        return
    try:
        outputs = model(raw_text)
        ade_df, drug_df = convert_to_df(outputs)
    except Exception as e:
        st.info("Sorry! We weren't able to process your request. Maybe try with another sentence?")
        st.write(e)
        return
    st.write("This is what the model outputs (with post-processing)!")
    st.write("#### Adverse Events Detected by SpanBERT")
    st.write(ade_df)
    st.write("#### Drugs Detected by SpanBERT")
    st.write(drug_df)
    st.write("#### Below is what the model directly outputs")
    st.write(outputs)
    st.write("## Want to know more about this project?")
    st.video("https://youtu.be/iiF39aEqLJ4")
    st.write("View the source code at https://github.com/pooky1955/cascadia-streamlit")





if __name__ == "__main__":
    st.set_page_config(page_title="Cascadia Showcase",page_icon="💊")
    main()

