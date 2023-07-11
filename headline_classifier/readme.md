# Headline Text Classifier

__Categorize Headline Text using Pre-Trained NLP Classifiers__

This Streamlit app allows users to use pre-trained NLP models to categorize headline
texts from news articles.

Several models were trained and are available to comparison:
- spacy_base: bag-of-words model using `spacy.TextCatBOW.v2`
- distilbert: `distilbert-base-uncased` transformer model [more info](https://huggingface.co/distilbert-base-uncased)
- bert: `bert-base-uncased` transformer model [more info](https://huggingface.co/bert-base-uncased)

The app is hosted on Streamlit Community Cloud:

https://saulslee-headline-classifier.streamlit.app/

### Pre-trained model storage and loading
The models used in this app for text classification require a lot of disk space (400MB+ for bert) and is not practically stored in a Github repository. Instead, the models are stored in a staging bucket on AWS S3, and fetched by the Streamlit app upon first load. The app also uses Streamlit's `@st.cache_resource` functionalilty to cache the models and minimize load time for the user.
