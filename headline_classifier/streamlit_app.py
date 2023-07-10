import streamlit as st

from utils.models import get_category, nlp_model_helper

# this app will use pre-trained classifiers to categorize
# input headline text


# setup variables
s3_bucket = st.secrets["S3_BUCKET"]
s3_key_spacy_base = """prod/streamlit/headline_classifier/models/spacy_base/
    textcat_model_2023-06-25/model-best.zip"""
s3_key_distilbert = """prod/streamlit/headline_classifier/models/distilbert/
    textcat_model_2023-06-23/model-best.zip"""

helper_spacy_base = nlp_model_helper("spacy_base")
helper_distilbert = nlp_model_helper("distilbert")


# begin the UI portion of the app
st.markdown("# Headline Title Classifier")
st.markdown(
    """
    This app will categorize a headline title using a pre-trained
    NLP classifier. Please feel free to try your own headline text.
    """
)

# load models
model_spacy_base = helper_spacy_base.load_model_from_s3(
    s3_bucket, s3_key_spacy_base
)

model_distilbert = helper_distilbert.load_model_from_s3(
    s3_bucket, s3_key_distilbert
)

# text input section and seed with default headline
title = st.text_input("Enter Headline Text:", "Man Walks on Moon")

st.markdown(f"__Category predictions for the headline phrase: {title}__")
get_category(model_spacy_base, helper_spacy_base.model_name, title)
get_category(model_distilbert, helper_distilbert.model_name, title)
