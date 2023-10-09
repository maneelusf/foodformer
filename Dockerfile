FROM python:3.10.10-slim

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html

# Stage 2: Copy your application code
COPY ./serving ./serving

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "80"]

