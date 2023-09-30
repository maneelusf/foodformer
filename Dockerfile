FROM python:3.10.10-slim

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY ./serving ./serving

FROM busybox

RUN mkdir /tmp/build/
# Add context to /tmp/build/
COPY . /tmp/build

# this last command outputs the list of files added to the build context:
RUN find /tmp/build/

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "80"]