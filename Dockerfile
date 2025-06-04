FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && \
    pip install --no-cache-dir -r requirements.txt

CMD ["bash", "start.sh"]
