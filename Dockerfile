FROM python:3.10.6

RUN pip install --upgrade pip
RUN pip install torch==1.12.0

WORKDIR /workspace

# COPY weights /workspace/weights

COPY . /workspace

ENTRYPOINT ["python", "server.py"]
