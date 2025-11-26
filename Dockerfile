FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    nginx \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip3 install -r requirements.txt

COPY nginx.conf /etc/nginx/sites-enabled/default

COPY start.sh /app/start.sh
# Bu komut, Windows satır sonlarını (CRLF) silerek start.sh'ın Linux'ta çalışmasını sağlar.
RUN sed -i 's/\r$//' /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["/app/start.sh"]
