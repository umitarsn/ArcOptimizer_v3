# ------------------------------------------------------------
# Railway için Streamlit Dockerfile'ı (PYTHON 3.11)
# ------------------------------------------------------------

# 1. TEMEL İMAJ
# Streamlit uygulamaları için resmi Python imajını kullanıyoruz
FROM python:3.11-slim

# 2. ORTAM DEĞİŞKENLERİ
# Streamlit'in statik dosyaları sunmasına izin vermek ve portu ayarlamak için
ENV STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true
ENV PYTHONUNBUFFERED=1

# 3. ÇALIŞMA DİZİNİ
WORKDIR /usr/src/app

# 4. BAĞIMLILIKLAR
# requirements.txt dosyasını kopyalayın
COPY requirements.txt .

# Gerekli sistem paketlerini kurun (özellikle Pillow/matplotlib için)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    # Gerekli diğer sistem paketleri buraya eklenebilir (örn: libsm6 libxext6) \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını kurun
RUN pip install --no-cache-dir -r requirements.txt

# 5. UYGULAMA DOSYALARINI KOPYALAMA
# Gerekli tüm dosya ve klasörleri kopyalayın
COPY . .

# 6. iOS SABİTLEME İÇİN STATİK DOSYA ENJEKSİYONU (KRİTİK ADIM)
# iOS kısayolunun ana ekran ikonunu doğru göstermesi için
RUN if [ -d "static" ]; then \
    mkdir -p /usr/share/streamlit/static_assets/static; \
    cp -r static/* /usr/share/streamlit/static_assets/static/; \
    fi

# 7. UYGULAMAYI BAŞLATMA (KRİTİK DÜZELTME)
# Streamlit'i Railway'in otomatik olarak atadığı $PORT'ta başlatır.
# ENTRYPOINT'in tırnak içinde 'sh -c' ile kullanılması, $PORT değişkeninin
# doğru şekilde sayı olarak yorumlanmasını sağlar.

ENTRYPOINT ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
