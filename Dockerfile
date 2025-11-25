# Dockerfile

# En stabil ve uyumlu Python sürümünü kullan (runtime.txt'yi destekler)
FROM python:3.11-slim

# Uygulama için ortam değişkenlerini ayarla
ENV PYTHONUNBUFFERED=1

# Linux sistem bağımlılıklarını kur (özellikle scikit-learn, numpy gibi kütüphanelerin 
# derlenmesi için gerekli olan build araçlarını)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    # Opsiyonel: Eğer veritabanı kullanıyorsanız (PostgreSQL) libpq-dev'i ekleyebilirsiniz
    # libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini oluştur
WORKDIR /app

# Gereksinimleri kopyala ve kur
# Bu adım, Railway'de yaşadığınız 'Build Failed' sorununu çözen runtime.txt yerine geçer, 
# çünkü artık Python sürümünü Dockerfile belirliyor.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu ve tüm varlıkları konteynere kopyalayın
# Bu, app.py, static/ klasörü ve içindeki ikon dosyasını da kopyalar.
COPY . /app

# KRİTİK ADIM: iOS İkon Enjeksiyonu
# apple-touch-icon'un statik dosyaya işaret eden mutlak yolu
ARG MOBILE_ICON_INJECTION="<link rel='apple-touch-icon' sizes='180x180' href='/app/static/apple-touch-icon-180x180.png'>\
<link rel='manifest' href='/app/static/manifest.json'>"

# Streamlit'in dahili HTML şablonunu (index.html) Stream Editor (sed) komutu ile değiştirir.
# Bu, mobil işletim sistemlerinin </head> etiketinden hemen önce ikon bağlantısını bulmasını sağlar.
RUN STREAMLIT_STATIC_PATH=$(python -c "import streamlit, os; print(os.path.join(os.path.dirname(streamlit.__file__), 'static', 'index.html'))") && \
    sed -i "/<\/head>/i\ ${MOBILE_ICON_INJECTION}" $STREAMLIT_STATIC_PATH

# Streamlit uygulamasını çalıştırın
# Railway'e özel PORT ve adresi kullanıyoruz
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
