FROM amsterdam/python
MAINTAINER datapunt@amsterdam.nl

ENV PYTHONUNBUFFERED 1

EXPOSE 8000

RUN mkdir -p /static && chown datapunt /static

COPY app /app/
COPY requirements.txt /app/

WORKDIR /app


RUN pip install --no-cache-dir -r requirements.txt

USER datapunt

ENV UWSGI_HTTP :8000
ENV UWSGI_MODULE app:application
ENV UWSGI_PROCESSES 8
ENV UWSGI_MASTER 1
ENV UWSGI_OFFLOAD_THREADS 1
ENV UWSGI_HARAKIRI 25

CMD uwsgi
