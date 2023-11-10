FROM amsterdam/python

ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN useradd --no-create-home classification

COPY app /app
COPY requirements-train.txt /app/requirements-train.txt

RUN set -eux; \
    pip install --no-cache -r /app/requirements-train.txt; \
    chgrp classification /app; \
    chmod g+w /app;

RUN set -eux; \
    mkdir -p /nltk /output; \
    chown classification /nltk; \
    chown classification /output;

ENV NLTK_DATA /nltk

USER classification
