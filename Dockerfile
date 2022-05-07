FROM jupyter/datascience-notebook:latest

RUN pip install imblearn black black[jupyter]