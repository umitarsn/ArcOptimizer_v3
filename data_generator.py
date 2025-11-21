import pandas as pd
import numpy as np
import os

def generate_synthetic_data(n_samples=500):
    np.random.seed(42)  # Her seferinde aynı veriyi üretmek için

    # 1. GİRDİLER (INPUTS)
    # ---------------------------------------------------------
    # Heat ID: 1'den başlayan sıra numarası
    heat_ids = np.arange(1, n_samples + 1)

    # Hurda Dağılımı (%) - Toplamı her zaman 100 olmak zorunda değil ama yakın olsun
    # HMS (Heavy Melting Scrap): Ana yük
    scrap_hms = np.random.normal(60, 10, n_samples).clip(40, 80) 
    # HBI (Hot Briquetted Iron): Temiz kaynak
    scrap_hbi = np.random.normal(10, 5, n_samples).clip(0, 20)
    # Shredded (Parçalanmış):
    scrap_shredded = 100 - (scrap_hms + scrap_hbi) + np.random.normal(0, 2, n_samples)
    scrap_shredded = scrap_shredded.clip(10, 40)

    # Enerji Girdileri (10 tonluk küçük bir ocak simülasyonu baz alınmıştır - snippet'e göre)
    # Ort. 4000 kWh (400 kWh/ton)
    power_kwh = np.random.normal(4100, 300, n_samples).clip(3500, 5000)
    
    # Oksijen (Nm3)
    oxygen_nm3 = np.random.normal(180, 20, n_samples).clip(140, 250)

    # Döküm Süresi (dk)
    tap_time = np.random.normal(55, 5, n_samples).clip(45, 75)

    # Panel Soğutma Verileri (Feature Engineering için)
    panel_t_in = np.random.normal(28, 2, n_samples).clip(25, 32)
    # Çıkış suyu sıcaklığı enerjiyle hafif korele artar
    panel_t_out = panel_t_in + np.random.normal(10, 2, n_samples) + (power_kwh / 1000)
    panel_flow = np.random.normal(25, 3, n_samples).clip(20, 35)

    # 2. HEDEF DEĞİŞKEN (TARGET) SİMÜLASYONU
    # ---------------------------------------------------------
    # Fiziksel formül simülasyonu:
    # Base Temp + (Elektrik * k1) + (Oksijen * k2) - (Hurda_Zorluğu * k3) - (Süre * Kayıp)
    
    base_temp = 1520 # Başlangıç sıvılaşma sıcaklığı
    
    # Isı kazançları
    gain_elec = power_kwh * 0.04  # Elektrik etkisi
    gain_chem = oxygen_nm3 * 0.2  # Ekzotermik reaksiyon
    
    # Isı kayıpları ve yük etkisi
    # HBI eritmek daha zordur, HMS daha kolaydır varsayımı
    load_factor = (scrap_hbi * 0.5) + (scrap_shredded * 0.1) 
    loss_time = tap_time * 0.8 # Süre uzadıkça ısı kaybı artar (duvar kayıpları)
    
    # Rastgelelik (Gürültü)
    noise = np.random.normal(0, 8, n_samples)

    # Sonuç Sıcaklık (Tap Temperature)
    tap_temp = base_temp + gain_elec + gain_chem - load_factor - loss_time + noise
    
    # Melt Temperature (Erime sıcaklığı - genelde Tap'ten biraz düşük olur)
    melt_temp = tap_temp - np.random.normal(10, 3, n_samples)

    # 3. DATAFRAME OLUŞTURMA
    # ---------------------------------------------------------
    df = pd.DataFrame({
        'heat_id': heat_ids,
        'scrap_HMS80_20_pct': np.round(scrap_hms, 2),
        'scrap_HBI_pct': np.round(scrap_hbi, 2),
        'scrap_Shredded_pct': np.round(scrap_shredded, 2),
        'power_kWh': np.round(power_kwh, 1),
        'oxygen_Nm3': np.round(oxygen_nm3, 1),
        'tap_temperature_C': np.round(tap_temp, 1),
        'tap_time_min': np.round(tap_time, 1),
        'panel_T_in_C': np.round(panel_t_in, 1),
        'panel_T_out_C': np.round(panel_t_out, 1),
        'panel_flow_kg_s': np.round(panel_flow, 2),
        'melt_temperature_C': np.round(melt_temp, 1)
    })

    return df

# Klasör yoksa oluştur
if not os.path.exists('data'):
    os.makedirs('data')

# Veriyi üret ve kaydet
df = generate_synthetic_data(500)
file_path = 'data/BG_EAF_panelcooling_demo.csv'
df.to_csv(file_path, index=False)

print(f"✅ Veri seti başarıyla oluşturuldu: {file_path}")
print(df.head())
