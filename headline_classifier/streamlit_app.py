
import spacy
import streamlit as st
import s3fs
import os


def get_category(nlp, model_name, title):
    doc = nlp(title)
    category = max(doc.cats, key=doc.cats.get)
    score = max(doc.cats.values())
    st.write(model_name)
    st.write("predicted category: ", category)
    st.write("predicted score: ", score)


# use cache_resource to reduce the number of times the model is loaded 
@st.cache_resource
def load_model_from_s3(
    s3_bucket,
    s3_model_key,
    model_type,
    save_basepath,
    save_filename,
    model_path="model-best",
):

    if not os.path.exists(save_basepath):
        os.makedirs(save_basepath)

    save_path = os.path.join(save_basepath, save_filename)

    if not os.path.exists(save_path):
        with st.spinner(
            f"""Loading NLP ({model_type}) models...
            (note this could take a few min as models can be large)
            """
        ):
            remote_path = os.path.join(s3_bucket, s3_model_key)
            fs = s3fs.S3FileSystem(anon=False)
            fs.get_file(remote_path, save_path)

    save_model_path = os.path.join(save_basepath, model_path)

    if not os.path.exists(save_model_path):
        import shutil

        shutil.unpack_archive(save_path, save_model_path)

    model = spacy.load(save_model_path)

    return model


@st.cache_resource
def load_local_model(model_path, model_type):

    with st.spinner(
        f"""Loading NLP ({model_type}) models...
        (note this could take a few min as models can be large)
        """
    ):
        model = spacy.load(model_path)
    st.success("Done loading models!")

    return model


s3_bucket = st.secrets["S3_BUCKET"]
s3_model_key = "prod/streamlit/headline_classifier/models/spacy_base/textcat_model_2023-06-25/model-best.zip"

save_basepath = "models/spacy_base"
save_filename = "model-best.zip"


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

# nlp = spacy.load("models/spacy_base/textcat_model_2023-06-25/model-best")
# nlp_distilbert = load_local_model(
#     "models/distilbert/textcat_model_transformer_2023-06-23/model-best",
#     "distilbert"
# )

nlp = load_model_from_s3(
    s3_bucket, s3_model_key, "spacy_base", save_basepath, save_filename
)

nlp_distilbert = load_model_from_s3(
    s3_bucket,
    "prod/streamlit/headline_classifier/models/distilbert/textcat_model_2023-06-23/model-best.zip",
    "distilbert",
    "models/distilbert",
    "model-best.zip"
)

# text input section and seed with default headline
title = st.text_input("Headline", "Man Walks on Moon")

if st.button("Run"):
    st.markdown(f"__Category predictions for phrase: {title}__")
    get_category(nlp, "spacy_base", title)
    get_category(nlp_distilbert, "distilbert", title)
else:
    st.markdown("__Please click Run to generate predictions__")
