import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import os
import joblib
# PIL, io, base64 gibi ikonla ilgili importlar tamamen kaldırıldı.

# ------------------------------------------------------------
# 1. SAYFA AYARLARI
#    page_icon=None yapılarak Streamlit'in kendi ikon ataması engellendi.
#    İkon sunumu (apple-touch-icon.png) Nginx tarafından yapılacaktır.
# ------------------------------------------------------------

st.set_page_config(
    page_title="Ferrokrom AI Optimizasyon",
    layout="wide",
    page_icon=None,  # <-- İkon ataması kaldırıldı ve NONE yapıldı.
    initial_sidebar_state="expanded"
)

# st.logo ve tüm HTML enjeksiyonları (Base64/Manifest/Link Etiketleri) kaldırıldı.

# ------------------------------------------------------------
# 2. VERİ VE SİMÜLASYON FONKSİYONLARI (Cache'li)
# ------------------------------------------------------------

@st.cache_data
def generate_dummy_trend_data(n_points=50):
    dates = pd.date_range(start="2023-01-01", periods=n_points, freq="D")
    panel_temps = np.random.normal(35, 5, n_points) + np.linspace(0, 10, n_points)
    stability = np.random.normal(85, 5, n_points) + np.sin(np.linspace(0, 4*np.pi, n_points)) * 5
    return pd.DataFrame({
        "Tarih": dates,
        "Panel_Temp_Avg": panel_temps,
        "Arc_Stability_KPI": stability.clip(0, 100)
    })

@st.cache_data
def generate_dummy_scrap_data(n_suppliers=4, n_lots=40):
    np.random.seed(42)
    suppliers = [f"Tedarikçi {chr(65 + i)}" for i in range(n_suppliers)]
    scrap_types = ["Krom İçi (High C)", "Paslanmaz Hurda", "Düşük Tenörlü cevher", "Şarj Kromu"]
    rows = []
    for i in range(n_lots):
        rows.append({
            "Supplier": np.random.choice(suppliers),
            "Scrap_Type": np.random.choice(scrap_types),
            "Price_USD_t": round(np.random.uniform(200, 500), 1),
            "Quality_Index": round(np.random.uniform(60, 95), 1),
            "Lot_tonnage": round(np.random.uniform(30, 90), 1),
            "Yield_pct": round(np.random.uniform(85, 98), 1),
            "kWh_per_t": round(np.random.uniform(350, 450), 1),
            "Electrode_kg_per_t": round(np.random.uniform(1.5, 2.5), 2),
        })
    return pd.DataFrame(rows)

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_thermal_cols = ["panel_T_in_C", "panel_T_out_C", "panel_flow_kg_s", "power_kWh"]
    if all(col in df.columns for col in required_thermal_cols):
        cp_kJ = 4.18  
        df['Q_Panel_kW'] = df['panel_flow_kg_s'] * (df['panel_T_out_C'] - df['panel_T_in_C']) * cp_kJ 
        df['Thermal_Stress_Index'] = (df['Q_Panel_kW'] * 0.1) + (df['power_kWh'] * 0.005) 
        max_val = df['Thermal_Stress_Index'].max()
        df['Thermal_Stress_Index'] = (df['Thermal_Stress_Index'] / max_val * 100) if max_val > 0 else 50.0
        df = df.drop(columns=['Q_Panel_kW'])

    required_scrap_cols = ["scrap_HMS80_20_pct", "scrap_HBI_pct", "scrap_Shredded_pct"]
    if all(col in df.columns for col in required_scrap_cols):
        df['Scrap_Quality_Index'] = (
            df['scrap_HBI_pct'] * 1.0 + 
            df['scrap_Shredded_pct'] * 0.7 + 
            df['scrap_HMS80_20_pct'] * 0.4
        )
        df = df.drop(columns=required_scrap_cols, errors='ignore') 
    
    return df.rename(columns={'Thermal_Imbalance_Index': 'Thermal_Stress_Index'})

# ------------------------------------------------------------
# 2.b AI Model Fonksiyonları (Tap Sıcaklığı Tahmini)
# ------------------------------------------------------------
MODEL_PATH = "models/tap_temperature_model.pkl"

def train_tap_temperature_model(df: pd.DataFrame):
    """Demo amaçlı: panel ve proses verilerinden tap sıcaklığını tahmin eden RF modeli eğitir."""
    target_col = "tap_temperature_C"
    if target_col not in df.columns:
        return None, None

    # Eğitim için kullanılmayacak kolonlar
    drop_cols = [
        "heat_id",
        "tap_temperature_C",
        "melt_temperature_C",
        "panel_T_in_C",
        "panel_T_out_C",
        "panel_flow_kg_s",
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    # Her ihtimale karşı, target kolonu eğitim setinde bulunmasın
    X = X.drop(columns=[col for col in X.columns if col == target_col], errors="ignore")
    y = df[target_col]

    # Çok küçük dataset durumunda hata almamak için kontrol
    if len(X) < 10:
        return None, None

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42,
    )
    model.fit(X, y)

    # Modeli diske kaydet
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_cols": list(X.columns),
        },
        MODEL_PATH,
    )
    return model, list(X.columns)

def load_tap_temperature_model():
    """Kaydedilmiş modeli ve feature listesini yükler. Yoksa (None, None) döner."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    data = joblib.load(MODEL_PATH)
    return data.get("model"), data.get("feature_cols")

def create_gauge_chart(value, title="Sıcaklık", min_v=1500, max_v=1750, target=1620):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        title = {'text': title},
        delta = {'reference': target, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [min_v, max_v], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'steps': [
                {'range': [min_v, 1600], 'color': '#4dabf5'},
                {'range': [1600, 1640], 'color': '#66ff66'},
                {'range': [1640, max_v], 'color': '#ff6666'}
            ]
        }
    ))
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=260)
    return fig

@st.cache_data
def generate_cfd_fields(power, deviation):
    nx, ny = 30, 30
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 10, ny)
    X, Y = np.meshgrid(x, y)
    
    deviation_amount = (1.0 - deviation/100.0) * 2.0
    center_x = 5.0 + deviation_amount * np.cos(np.pi/4)
    center_y = 5.0 + deviation_amount * np.sin(np.pi/4)
    dist_sq = (X - center_x)**2 + (Y - center_y)**2
    
    diffusion_factor = 8.0 + (power / 400.0) 
    max_arc_temp = 1600 + (power * 0.06) 
    temp_field = max_arc_temp * np.exp(-dist_sq / diffusion_factor)
    temp_field = np.maximum(temp_field, 1500) 
    
    angle = np.arctan2(Y - center_y, X - center_x)
    radius = np.sqrt(dist_sq)
    vel_mag = (power / 5000.0) * np.exp(-radius/3.0)
    V_x = -vel_mag * np.sin(angle) + (vel_mag * 0.3 * np.cos(angle))
    V_y = vel_mag * np.cos(angle) + (vel_mag * 0.3 * np.sin(angle))
    
    return X, Y, temp_field, V_x, V_y

# ------------------------------------------------------------
# 3. UYGULAMA ANA AKIŞI
# ------------------------------------------------------------
def main():
    # --- SOL MENÜ BAŞLIĞI ---
    st.sidebar.header("Ferrokrom AI")
    st.sidebar.markdown("**Akıllı Karar Destek Sistemi**")
    st.sidebar.markdown("---")
    
    selected_module = st.sidebar.radio(
        "📑 Modül Seçimi:",
        [
            "1️⃣ AI Bakım ve Duruş Engelleme",
            "2️⃣ AI Girdi Maliyetleri Düşürme",
            "3️⃣ AI Proses Stabilizasyonu",
            "4️⃣ Alarm, Tavsiye ve KPI'lar",
            "5️⃣ AI Enterprise Level (EBITDA)",
            "6️⃣ Scrap & Purchase Intelligence",
        ],
        index=0
    )
    
    st.markdown("### 🔍 Ferrokrom Ark Ocağı için AI Destekli Karar Sistemi")

    # --- VERİ YÜKLEME ---
    try:
        # NOTE: Dosya yolu değişmiş olabilir, kontrol ediniz.
        df = pd.read_csv("data/BG_EAF_panelcooling_demo.csv") 
    except FileNotFoundError:
        st.error("❌ Veri dosyası bulunamadı! data/BG_EAF_panelcooling_demo.csv'yi kontrol edin.")
        if selected_module != "4️⃣ Alarm, Tavsiye ve KPI'lar": 
            st.stop()

    
    df = feature_engineering(df)

    # --- AI Modeli: Tap Sıcaklığı Tahmini ---
    st.sidebar.markdown("### 🔥 AI Modeli (Tap Sıcaklığı)")
    model, feature_cols = load_tap_temperature_model()
    retrain_clicked = st.sidebar.button("Bu verilerle modeli eğit / güncelle")

    if retrain_clicked or model is None or feature_cols is None:
        model, feature_cols = train_tap_temperature_model(df)
        if model is not None:
            st.sidebar.success("Model güncellendi.")
        else:
            st.sidebar.error("Model eğitimi için yeterli veya uygun veri bulunamadı.")

    if model is not None and feature_cols is not None:
        X = df[feature_cols]
    else:
        # Fallback: tüm sayısal sütunları kullan
        X = df.select_dtypes(include=["number"])
        feature_cols = list(X.columns)
    
    trend_df = generate_dummy_trend_data()
    tonnage = 10.0 

    # ------------------------------------------------------------------
    # ORTAK GİRDİLER (SIDEBAR)
    # ------------------------------------------------------------------
    st.sidebar.header("🎛️ Simülasyon Kontrol Paneli")
    
    input_data = {}

    arc_stability_factor = st.sidebar.slider("⚡ Ark Stabilizasyon Faktörü (0-1)", 0.0, 1.0, 0.90, 0.01)
    calculated_stress = (1.0 - arc_stability_factor) * 100
    input_data['Thermal_Stress_Index'] = calculated_stress
    
    for col in X.columns:
        if col == 'power_kWh':
            input_data[col] = st.sidebar.slider("Güç (kWh)", 3000.0, 5000.0, 4000.0)
        elif col == 'oxygen_Nm3':
            input_data[col] = st.sidebar.slider("Oksijen (Nm³)", 8000, 20000, 12000, 500)
        elif col == 'Thermal_Stress_Index':
            continue
        elif col != 'Thermal_Stress_Index':
            input_data[col] = df[col].mean()

    if selected_module in ["2️⃣ AI Girdi Maliyetleri Düşürme", "5️⃣ AI Enterprise Level (EBITDA)", "6️⃣ Scrap & Purchase Intelligence"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("💰 Piyasa Fiyatları")
        price_scrap = st.sidebar.number_input("Hurda Fiyatı ($/t)", 200., 600., 400.)
        price_elec = st.sidebar.number_input("Elektrik Fiyatı ($/MWh)", 30, 200, 90)
        price_oxy = st.sidebar.number_input("Oksijen Fiyatı ($/Nm³)", 0.02, 1.00, 0.08, step=0.01)
        price_electrode = st.sidebar.number_input("Elektrot Fiyatı ($/kg)", 2.0, 15.0, 4.0, step=0.5)
    else:
        price_scrap, price_elec, price_oxy, price_electrode = 400, 90, 0.08, 4.0

    input_df = pd.DataFrame([input_data])[X.columns]
    prediction = model.predict(input_df)[0] if model is not None else np.nan
    panel_health_index = 100 - calculated_stress
    arc_deviation_pct = (1.0 - arc_stability_factor) * 40.0 

    # --- MODÜL İÇERİKLERİ ---
    if selected_module == "1️⃣ AI Bakım ve Duruş Engelleme":
        st.title("🛡️ Modül 1: AI Bakım & Duruş Engelleme")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Panel Sıcaklık Trendi")
            fig_trend = px.line(trend_df, x="Tarih", y="Panel_Temp_Avg", title="Panel Çıkış Suyu Sıcaklığı")
            fig_trend.add_hline(y=45, line_dash="dot", annotation_text="Limit", line_color="red")
            st.plotly_chart(fig_trend, use_container_width=True)
        with col2:
            st.subheader("Panel Sağlık Skoru")
            fig_health = go.Figure(go.Indicator(
                mode="gauge+number",
                value=panel_health_index,
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "green" if panel_health_index > 50 else "red"}
                },
                title={'text': "Panel Sağlık"}
            ))
            fig_health.update_layout(height=250)
            st.plotly_chart(fig_health, use_container_width=True)
            if panel_health_index < 40: 
                st.error("🚨 **KRİTİK:** Yüksek termal stres!")
            else: 
                st.success("✅ Sistem Stabil.")

    elif selected_module == "2️⃣ AI Girdi Maliyetleri Düşürme":
        st.title("💰 Modül 2: Girdi Maliyetleri Optimizasyonu")
        cost_elec = (input_data['power_kWh'] * (price_elec / 1000.0))
        cost_oxy = input_data['oxygen_Nm3'] * price_oxy
        cost_scrap = tonnage * price_scrap
        cost_electrode = tonnage * 1.8 * price_electrode
        total_cost = cost_scrap + cost_elec + cost_oxy + cost_electrode
        unit_cost = total_cost / tonnage
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Toplam Döküm Maliyeti", f"${total_cost:,.2f}")
            st.metric("Birim Maliyet ($/t)", f"${unit_cost:.2f}")
        with c2:
            df_cost = pd.DataFrame({
                "Kalem": ["Hurda", "Elektrik", "Oksijen", "Elektrot"],
                "Maliyet": [cost_scrap, cost_elec, cost_oxy, cost_electrode]
            })
            fig_pie = px.pie(df_cost, values='Maliyet', names='Kalem', title="Maliyet Kırılımı", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

    elif selected_module == "3️⃣ AI Proses Stabilizasyonu":
        st.title("⚙️ Modül 3: AI Proses Stabilizasyonu")
        st.write("Bu modülde ark stabilitesi, termal stres ve tap sıcaklığı AI tahmini birlikte analiz edilir.")
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig = create_gauge_chart(prediction, title="Tahmini Tap Sıcaklığı (°C)")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            st.metric("Termal Stres İndeksi", f"{calculated_stress:.1f}")
            st.metric("Ark Stabilizasyon Faktörü", f"{arc_stability_factor:.2f}")
        st.markdown("**Yorum:** Model, panel ve proses koşullarına göre tap sıcaklığını tahmin eder. Termal stres yükseldikçe, stabil çalışma bölgesinden uzaklaşılır.")

    elif selected_module == "4️⃣ Alarm, Tavsiye ve KPI'lar":
        st.title("🚨 Modül 4: Alarm, Tavsiye ve KPI'lar")
        st.write("Bu modül demo amaçlı statik öneriler içerir. Gerçek sahada historian verisi ve alarm kayıtlarıyla beslenecektir.")
        st.markdown("""
        - 🔴 **Kritik Alarm:** Panel çıkış suyu sıcaklığı > 50°C
        - 🟠 **Uyarı:** Ark stabilizasyon faktörü < 0.8
        - 🟢 **Bilgi:** Proses normal çalışma bandında.
        """)
        st.markdown("---")
        st.subheader("Örnek KPI'lar")
        kcol1, kcol2, kcol3 = st.columns(3)
        kcol1.metric("Ortalama Panel ΔT", "12.5°C", "-0.3°C")
        kcol2.metric("Planlanmamış Duruş Sayısı", "3", "-1")
        kcol3.metric("Enerji Tüketimi (kWh/t)", "410", "-8")

    elif selected_module == "5️⃣ AI Enterprise Level (EBITDA)":
        st.title("🏢 Modül 5: Kurumsal İş Zekası (EBITDA)")
        with st.expander("Finansal Hedef Ayarları", expanded=True):
            col_e1, col_e2 = st.columns(2)
            sales_price = col_e1.number_input("Hedef Satış Fiyatı ($/ton)", 500, 3000, 1500)
            monthly_target = col_e2.number_input("Aylık Hedef Tonaj", 1000, 50000, 10000)
            fixed_cost = st.number_input("Aylık Sabit Giderler ($)", 100000, 2000000, 500000)
        
        cost_elec = (input_data['power_kWh'] * (price_elec / 1000.0))
        cost_oxy = input_data['oxygen_Nm3'] * price_oxy
        cost_scrap = tonnage * price_scrap
        cost_electrode = tonnage * 1.8 * price_electrode
        unit_var_cost = (cost_scrap + cost_elec + cost_oxy + cost_electrode) / tonnage

        contribution = sales_price - unit_var_cost
        monthly_margin = contribution * monthly_target - fixed_cost

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Değişken Birim Maliyet", f"${unit_var_cost:.2f}")
            st.metric("Katkı Payı", f"${contribution:.2f}")
        with c2:
            st.metric("Aylık EBITDA Tahmini", f"${monthly_margin:,.0f}")
        st.write("Bu modülde, proses optimizasyonunun şirket EBITDA'sına etkisi simüle edilir.")

    elif selected_module == "6️⃣ Scrap & Purchase Intelligence":
        st.title("♻️ Modül 6: Scrap & Purchase Intelligence")
        st.write("Hurda tedarikçileri, lot kalitesi ve gerçek maliyet analizi (enerji + fire dahil).")
        uploaded_scrap = st.file_uploader("Hurda Verisi (CSV)", type=["csv"])
        scrap_df = pd.read_csv(uploaded_scrap) if uploaded_scrap else generate_dummy_scrap_data()
        with st.expander("Veri Önizleme"):
            st.dataframe(scrap_df.head(), use_container_width=True)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_scatter = px.scatter(
                scrap_df,
                x="Price_USD_t",
                y="Quality_Index",
                color="Supplier",
                title="Tedarikçi Fiyat / Kalite Matrisi",
                hover_data=["Scrap_Type"]
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        with col_s2:
            scrap_df["Energy_Cost"] = scrap_df["kWh_per_t"] * (price_elec / 1000.0)
            scrap_df["True_Cost"] = scrap_df["Price_USD_t"] + scrap_df["Energy_Cost"]
            fig_bar = px.bar(
                scrap_df.groupby("Supplier")[["Price_USD_t", "True_Cost"]].mean().reset_index(),
                x="Supplier",
                y=["Price_USD_t", "True_Cost"],
                barmode="group",
                title="Nominal Fiyat vs Gerçek Maliyet"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

if __name__ == "__main__":
    main()
