FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    RL_DEVICE=cpu

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py ./
COPY models/ ./models/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Prints target weights for today. Override the command to train: python run.py
ENTRYPOINT ["python"]
CMD ["predict.py"]
