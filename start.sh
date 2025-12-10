#!/bin/bash

# PORT ortam değişkenini nginx config'e enjekte et
envsubst '${PORT}' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf

# Nginx'i başlat
nginx &

# Streamlit'i başlat
streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
