from tensorflow/tensorflow:latest-gpu
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
WORKDIR /home/eduardo
COPY pyproject.toml .
RUN poetry install
# RUN pip install sentence-transformers