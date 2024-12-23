import streamlit as st
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from newspaper import Article
import re

@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return tokenizer, model

def summarize_text(text, model, tokenizer, max_length=130, min_length=30):
    inputs = tokenizer.encode(
        "summarize: " + text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )
    
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def extract_article(url):
    article = Article(url)
    try:
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return None

def main():
    st.title("📝 Text Summarization Tool")
    st.write("This app summarizes text using the BART model from Facebook/Meta AI.")

    tokenizer, model = load_model()

    input_method = st.radio(
        "Choose input method:",
        ("Enter Text", "Paste URL")
    )

    text_to_summarize = ""

    if input_method == "Enter Text":
        text_to_summarize = st.text_area("Enter the text to summarize:", height=200)
    else:
        url = st.text_input("Enter the URL of the article:")
        if url:
            with st.spinner("Extracting article..."):
                text_to_summarize = extract_article(url)
                if text_to_summarize:
                    st.success("Article extracted successfully!")
                    with st.expander("Show original article"):
                        st.write(text_to_summarize)
                else:
                    st.error("Failed to extract article. Please check the URL or try another one.")

    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Maximum summary length", 50, 200, 130)
    with col2:
        min_length = st.slider("Minimum summary length", 20, 100, 30)

    if st.button("Summarize") and text_to_summarize:
        with st.spinner("Generating summary..."):
            summary = summarize_text(text_to_summarize, model, tokenizer, max_length, min_length)
            
        st.write("### Summary")
        st.write(summary)

        original_length = len(text_to_summarize.split())
        summary_length = len(summary.split())
        reduction = ((original_length - summary_length) / original_length) * 100

        st.write("### Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Original Length", f"{original_length} words")
        col2.metric("Summary Length", f"{summary_length} words")
        col3.metric("Reduction", f"{reduction:.1f}%")

if __name__ == "__main__":
    main() 