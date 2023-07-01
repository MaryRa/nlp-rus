FROM jupyter/datascience-notebook

USER root

FROM python:3.8-slim
RUN python -m pip install \
        parse \
        realpython-reader

RUN mkdir /src
RUN mkdir /nlp-rus

COPY . /src

WORKDIR /src

RUN pip install -r requirements.txt

CMD python lib/run.py