FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx curl build-essential gettext-base \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Nginx config'i template olarak kopyalıyoruz
COPY nginx.conf /etc/nginx/conf.d/default.conf.template

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8501

ENTRYPOINT ["/start.sh"]
