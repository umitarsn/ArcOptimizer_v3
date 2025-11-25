# ------------------------------------------------------------
# Railway için Streamlit Dockerfile (PYTHON 3.11)
# ------------------------------------------------------------

# 1. TEMEL İMAJ
FROM python:3.11-slim

# 2. ORTAM DEĞİŞKENLERİ
# Streamlit'in statik dosyaları sunmasına izin vermek için
ENV STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true
ENV PYTHONUNBUFFERED=1

# 3. ÇALIŞMA DİZİNİ
WORKDIR /usr/src/app

# 4. BAĞIMLILIKLAR
COPY requirements.txt .

# Gerekli sistem paketlerini kurun (özellikle Pillow/matplotlib ve diğer bağımlılıklar için)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    # Gerekli olabilecek diğer sistem paketleri buraya eklenebilir \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını kurun
RUN pip install --no-cache-dir -r requirements.txt

# 5. UYGULAMA DOSYALARINI KOPYALAMA
# Gerekli tüm dosya ve klasörleri (app.py, data/, static/ vb.) kopyalayın
COPY . .

# 6. iOS SABİTLEME İÇİN STATİK DOSYA ENJEKSİYONU (YENİ KESİN ÇÖZÜM)
# Streamlit, iOS ikonunu ve diğer statik dosyaları burada arar.
# 'static' klasörümüzdeki içeriği doğrudan Streamlit'in statik varlıklar dizinine kopyalarız.
COPY static /usr/share/streamlit/static_assets

# 7. UYGULAMAYI BAŞLATMA (PORT DÜZELTMESİ)
# Streamlit'i Railway'in otomatik olarak atadığı $PORT'ta başlatır.
ENTRYPOINT ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
