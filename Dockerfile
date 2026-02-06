FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install production server
RUN pip install gunicorn

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 5000

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# Use Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "2", "src.app:app"]