import pandas as pd
import spacy
import streamlit as st
import s3fs
import os


def get_category(nlp, model_name, title, print_output=False):

    doc = nlp(title)
    category = max(doc.cats, key=doc.cats.get)
    score = max(doc.cats.values())

    if print_output:
        st.markdown(f"__Model: {model_name}__")
        st.markdown(f"Predicted Category: {category}")
        st.markdown(f"Predicted Score: {score:.3f}\n\n")

    return model_name, category, score


def get_predictions(title, model_list):

    predictions_list = []
    col_names = ["Model Name", "Predicted Category", "Predicted Score"]

    for cur_model, cur_model_name in model_list:
        predictions_list.append(
            get_category(cur_model, cur_model_name, title)
        )

    df = pd.DataFrame(predictions_list, columns=col_names)

    return df


class nlp_model_helper():

    def __init__(
        self, model_name, save_basepath=None, save_filename="model-best.zip"
    ):
        self.model_name = model_name
        if save_basepath:
            self.save_basepath = save_basepath
        else:
            self.save_basepath = os.path.join("models", self.model_name)
        self.save_filename = save_filename

    @st.cache_resource
    # use cache_resource to reduce the number of times the model is loaded
    # Note: use _self instead of self as the @cache_resource decorator
    # requires input args to be hashable, the underscore excludes the arg
    def load_model_from_s3(
        _self,
        s3_bucket,
        s3_model_key,
        model_path="model-best",
    ):
        import shutil

        if not os.path.exists(_self.save_basepath):
            os.makedirs(_self.save_basepath)

        save_path = os.path.join(_self.save_basepath, _self.save_filename)

        # get the file from S3
        with st.spinner(
            f"""Loading NLP ({_self.model_name}) models...
            (note this could take a few min as models can be large)
            """
        ):
            remote_path = os.path.join(s3_bucket, s3_model_key)
            fs = s3fs.S3FileSystem(anon=False)
            fs.get_file(remote_path, save_path)

        save_model_path = os.path.join(_self.save_basepath, model_path)

        # unpack the zip file
        shutil.unpack_archive(save_path, save_model_path)

        # delete the zip file to save room
        os.remove(save_path)

        model = spacy.load(save_model_path)

        return model

    @st.cache_resource
    # use cache_resource to reduce the number of times the model is loaded
    # Note: use _self instead of self as the @cache_resource decorator
    # requires input args to be hashable, the underscore excludes the arg
    def load_local_model(_self, model_path):

        with st.spinner(
            f"""Loading NLP ({_self.model_name}) models...
            (note this could take a few min as models can be large)
            """
        ):
            model = spacy.load(model_path)

        return model
