import streamlit as st

from utils.models import get_predictions, nlp_model_helper

# this app will use pre-trained classifiers to categorize
# input headline text


# setup variables
s3_bucket = st.secrets["S3_BUCKET"]
s3_key_spacy_base = (
    "prod/streamlit/headline_classifier/models/spacy_base/"
    "textcat_model_2023-06-25/model-best.zip"
)
s3_key_distilbert = (
    "prod/streamlit/headline_classifier/models/distilbert/"
    "textcat_model_2023-06-23/model-best.zip"
)
s3_key_bert = (
    "prod/streamlit/headline_classifier/models/bert/"
    "textcat_model_2023-06-23/model-best.zip"
)

helper_spacy_base = nlp_model_helper("spacy_base")
helper_distilbert = nlp_model_helper("distilbert")
helper_bert = nlp_model_helper("bert")


# begin the UI portion of the app
st.markdown("# Headline Title Classifier")
st.markdown(
    """
    __The code for this app can be found on Github
    [here](https://github.com/Saul-S-Lee/streamlit/tree/main/headline_classifier)__.
    """
)
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

model_bert = helper_distilbert.load_model_from_s3(
    s3_bucket, s3_key_bert
)

# text input section and seed with default headline
title = st.text_input("Enter Headline Text:", "Man Walks on Moon")

st.divider()

# generate predictions
st.markdown(f"__Category predictions for the headline phrase: \"{title}\"__")
model_list = [
    (model_spacy_base, helper_spacy_base.model_name),
    (model_distilbert, helper_distilbert.model_name),
    (model_bert, helper_bert.model_name),
]

df = get_predictions(title, model_list)

st.dataframe(df, hide_index=True)

st.markdown(
    """
    Model Description:
    - spacy_base: bag-of-words model using `spacy.TextCatBOW.v2`
    - distilbert: `distilbert-base-uncased` transformer model
    [more info](https://huggingface.co/distilbert-base-uncased)
    - bert: `bert-base-uncased` transformer model
    [more info](https://huggingface.co/bert-base-uncased)
    """
)
