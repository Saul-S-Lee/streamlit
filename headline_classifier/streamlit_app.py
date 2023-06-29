
import spacy
import streamlit as st


def get_category(nlp, model_name, title):
    doc = nlp(title)
    category = max(doc.cats, key=doc.cats.get)
    score = max(doc.cats.values())
    st.write(model_name)
    st.write("predicted category: ", category)
    st.write("predicted score: ", score)


# this app will use pre-trained classifiers to categorize
# input headline

st.markdown("# Headline Title Classifier")
st.markdown(
    """
    This app will categorize a headline title using a pre-trained
    NLP classifier. Please feel free to try your own headline text.
    """
)

# load models
nlp = spacy.load("models/spacy_base/textcat_model_2023-06-25/model-best")
nlp_distilbert = spacy.load("models/distilbert/textcat_model_transformer_2023-06-23/model-best")


# text input section and seed with default headline
title = st.text_input("Headline", "Men Walk on Moon")

if st.button("Run"):
    get_category(nlp, "spacy_base", title)
    get_category(nlp_distilbert, "distilbert", title)
else:
    st.markdown("__Please click Run to generate predictions__")
