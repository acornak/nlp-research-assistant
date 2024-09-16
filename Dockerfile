FROM python:3.11-slim AS builder 

WORKDIR /src

RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    wget \
    curl \
    gcc \
    g++ \
    make \
    libsqlite3-dev \
    sqlite3 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ca-certificates \
    && apt-get clean

RUN sqlite3 --version

RUN pip install poetry==1.8.3

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.in-project false && \
    poetry install --no-root

COPY ./app ./app

COPY ./experiments/__init__.py ./experiments/__init__.py
COPY ./experiments/fine_tuning.py ./experiments/fine_tuning.py
COPY ./experiments/visualize.py ./experiments/visualize.py

COPY ./main.py ./main.py

COPY ./preprocess ./preprocess

COPY ./serving ./serving

EXPOSE 8000

CMD ["poetry", "run", "python", "main.py"]
