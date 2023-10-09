FROM locustio/locust:latest
COPY images /dataset
COPY locustfile.py /locustfile.py
CMD ["-f", "/locustfile.py"]