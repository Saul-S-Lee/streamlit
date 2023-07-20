# streamlit
This repository houses the scripts for Streamlit apps which can be run on Streamlit Community Cloud. The apps are organized into different subfolders. The intent is for the app within each top level subfolder to run independently.

Each top level folder should contain a driver file for the streamlit app (e.g. `streamlit_app.py`) and an optional `requirements.txt` file to let Streamlit Community Cloud know how to set up the environment.

#### List of Apps
- [headline_classifier](headline_classifier): This app allows users to use pre-trained NLP models to categorize headline
texts from news articles.
- [hello](hello): This is a hello world app to test out Streamlit