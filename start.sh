#!/bin/bash
nginx -g 'daemon off;' &
streamlit run app.py --server.port=8501 --server.address=127.0.0.1 --server.enableCORS=false --server.enableXsrfProtection=false
