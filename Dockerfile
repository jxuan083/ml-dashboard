FROM python:3.13-slim

WORKDIR /app

COPY sdk/ ./sdk/
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/
COPY index.html .
COPY firebase-config.js .

WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
