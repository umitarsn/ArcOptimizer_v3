FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# BURAYA DİKKAT: conf.d değil, sites-enabled/default
COPY nginx.conf /etc/nginx/sites-enabled/default

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 80 8501
ENTRYPOINT ["/start.sh"]
