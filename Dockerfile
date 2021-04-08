FROM python:3.8.5

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

ENV NAME model

CMD ["python", "main.py"]