FROM python:3.10.10-slim

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

RUN pwd && ls

COPY ./serving ./serving

COPY ./artifacts ./artifacts

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "80"]