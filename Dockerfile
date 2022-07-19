FROM jupyter/datascience-notebook:latest

RUN pip install imblearn black black[jupyter] kaggle lazypredict sklearn lingam torch transformers sklearn

COPY --chown=jovyan ./kaggle.json /home/jovyan/.kaggle/