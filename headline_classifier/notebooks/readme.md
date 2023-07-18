# Headline Text Classifier: Model Training

This folder contains the notebooks and supporting files used to train the Spacy textcat classification models. The model data preparation and training scripts were adapted from Catherine Breslin's Medium post [here](https://catherinebreslin.medium.com/text-classification-with-spacy-3-0-d945e2e8fc44).

## Training data
This model was trained using the News Category Dataset that is freely available on Kaggle [here](https://www.kaggle.com/datasets/rmisra/news-category-dataset). Some preprocessing was required to extract and convert the data from the json format to the Spacy doc format.

## Environment
The models were trained on a Nvidia GPU-enabled Windows 11 machine running Ubuntu in WSL2. Though not stricly necessary, an Nvidia GPU did speed up the training process.

Transformer models (BERT, DistilBERT, and RoBERTa) were run in a separate virtual environment with `spacy-transformers` installed.

Although most of the training scripts were shell commands run on different cells, Jupyter notebooks were still used to help organize the training workflow.
