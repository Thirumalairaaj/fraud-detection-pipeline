FROM python:3.9-slim

#headless matplotlib

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY test/ test/
COPY pytest.ini .

CMD ["python","-m","src.train"]