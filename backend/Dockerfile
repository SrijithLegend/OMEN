'
FROM python:3.15-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "psycopg[binary]>=3.1"
COPY app app
COPY omen_bench.py .
RUN useradd --system --no-create-home omen && chown omen /srv  # the server never needs root; /srv stays writable for the SQLite default
USER omen
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
