
import spacy
import streamlit as st

# this app will use pre-trained classifiers to categorize
# input headline

st.markdown("# Headline Title Classifier")

nlp = spacy.load("models/model-best")

st.markdown(
    """
    This app will categorize a headline title using a pre-trained
    NLP classifier. Please feel free to try your own headline text.
    """
)

# text input section and seed with default headline
title = st.text_input("Headline", "Men Walk on Moon")

if st.button("Run"):
    doc = nlp(title)
    category = max(doc.cats, key=doc.cats.get)
    score = max(doc.cats.values())
    st.write("predicted category: ", category)
    st.write("predicted score: ", score)
else:
    st.markdown("__Please click Run to generate predictions__")
