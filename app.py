import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image # Sadece Image'ı import ediyoruz, ImageOps silindi.
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
# Diğer gerekli importlar


# --- LOGO YÜKLEME ---
# logo.jpg, tarayıcı favicon'u ve yan panel logosu için kullanılır!
try:
    # logo.jpg'yi PIL nesnesi olarak yükleyin
    im = Image.open("logo.jpg")
except FileNotFoundError:
    im = None
except Exception:
    im = None 
    
# ------------------------------------------------------------
# 1. SAYFA AYARLARI
# ------------------------------------------------------------

st.set_page_config(
    page_title="Ferrokrom AI Optimizasyon",
    layout="wide",
    page_icon=im, # Tarayıcı sekmesi ikonunu (favicon) ayarlar
    initial_sidebar_state="expanded"
)

# Streamlit Üst Bar Logosu (Yan panelin üstü)
try:
    if im:
        st.logo(im, icon_image=im)
except:
    pass


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
def generate_dummy_scrap_data():
    data = {
        'Scrap_Type': ['Heavy Melt Steel', 'Shredded Scrap', 'Busheling Scrap', 'Heavy Melt Steel', 'Plate/Structural'],
        'Supplier': ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier A'],
        'Price_USD_t': [420, 380, 450, 410, 480],
        'Quality_Index': [90, 75, 95, 85, 98],
        'Lot_tonnage': [1500, 1000, 800, 2000, 1200],
        'kWh_per_t': [380, 450, 350, 400, 320]
    }
    return pd.DataFrame(data)

@st.cache_data
def feature_engineering(df):
    if 'panel_T_out_C' in df.columns and 'panel_T_in_C' in df.columns:
        df['Panel_Temp_Delta_C'] = df['panel_T_out_C'] - df['panel_T_in_C']
    if 'power_kWh' in df.columns and 'tap_time_min' in df.columns:
        df['Energy_Rate'] = df['power_kWh'] / df['tap_time_min']
    
    if 'Scrap_Quality_Index' not in df.columns:
        df['Scrap_Quality_Index'] = 80
    if 'Thermal_Stress_Index' not in df.columns:
        df['Thermal_Stress_Index'] = 10
        
    return df

def create_gauge_chart(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Tahmini Döküm Sıcaklığı (°C)"},
        gauge={
            'axis': {'range': [1500, 1700], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkgreen"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [1500, 1600], 'color': 'red'},
                {'range': [1600, 1650], 'color': 'yellow'},
                {'range': [1650, 1700], 'color': 'green'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 1680
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(t=50, b=10, l=10, r=10))
    return fig

def generate_cfd_fields(power, deviation_pct, size=20):
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    T = 1500 + (power / 5000) * 200 + 50 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.1)
    Vx = (X - 0.5) * (0.1 + deviation_pct / 100)
    Vy = (Y - 0.5) * (0.1 + deviation_pct / 100)
    Vx += 0.05 * np.sin(5 * Y)
    Vy += 0.05 * np.cos(5 * X)
    T = np.clip(T, 1500, 1700) 
    return X, Y, T, Vx, Vy

# ------------------------------------------------------------
# 3. MAIN FONKSİYONU
# ------------------------------------------------------------

def main():
    # --- LOGO VE MENÜ BAŞLIĞI ---
    if im:
        st.sidebar.image(im, use_container_width=True)
    else:
        st.sidebar.header("Ferrokrom AI")
        st.sidebar.error("❌ logo.jpg bulunamadı!")

    st.sidebar.markdown("---")
    
    # --- MODÜL SEÇİMİ ---
    selected_module = st.sidebar.radio(
        "📑 Modül Seçimi:",
        [
            "1️⃣ AI Bakım ve Duruş Engelleme",
            "2️⃣ AI Girdi Maliyetleri Düşürme",
            "3️⃣ Karar Destek Modülü (Process)",
            "4️⃣ Alarm, Tavsiye ve KPI'lar",
            "5️⃣ AI Enterprise Level (EBITDA)",
            "6️⃣ Scrap & Purchase Intelligence"
        ]
    )
    st.sidebar.markdown("---")

    # --- VERİ YÜKLEME ---
    try:
        df = pd.read_csv("data/BG_EAF_panelcooling_demo.csv")
    except FileNotFoundError:
        st.error("❌ Veri dosyası bulunamadı! data/BG_EAF_panelcooling_demo.csv'yi kontrol edin.")
        if selected_module != "4️⃣ Alarm, Tavsiye ve KPI'lar": 
            st.stop()

    df = feature_engineering(df)
    
    # Model hazırlığı (Kısaltılmış)
    target_col = "tap_temperature_C"
    drop_cols = ["heat_id", "tap_temperature_C", "melt_temperature_C", "panel_T_in_C", "panel_T_out_C", "panel_flow_kg_s"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target_col], errors='ignore')
    y = df[target_col]
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    trend_df = generate_dummy_trend_data()
    tonnage = 10.0 # Örnek tonaj

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
            input_data[col] = st.sidebar.slider("Oksijen (Nm3)", 100.0, 300.0, 200.0)
        elif col == 'Scrap_Quality_Index':
            input_data[col] = st.sidebar.slider("Hurda Kalitesi (0-100)", 0.0, 100.0, 70.0)
        elif col == 'tap_time_min':
            input_data[col] = st.sidebar.slider("Döküm Süresi (dk)", 40.0, 70.0, 55.0)
        elif col not in input_data: 
            input_data[col] = df[col].mean()

    # Maliyet Girdileri
    if selected_module in ["2️⃣ AI Girdi Maliyetleri Düşürme", "5️⃣ AI Enterprise Level (EBITDA)", "6️⃣ Scrap & Purchase Intelligence"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("💰 Piyasa Fiyatları")
        price_scrap = st.sidebar.number_input("Hurda Fiyatı ($/t)", 200., 600., 400.)
        price_elec = st.sidebar.number_input("Elektrik Fiyatı ($/MWh)", 30, 200, 90)
        price_oxy = st.sidebar.number_input("Oksijen Fiyatı ($/Nm³)", 0.02, 1.00, 0.08, step=0.01)
        price_electrode = st.sidebar.number_input("Elektrot Fiyatı ($/kg)", 2.0, 15.0, 4.0, step=0.5)
    else:
        price_scrap, price_elec, price_oxy, price_electrode = 400, 90, 0.08, 4.0

    # Tahmin ve KPI Hesaplamaları
    input_df = pd.DataFrame([input_data])[X.columns]
    prediction = model.predict(input_df)[0]
    panel_health_index = 100 - calculated_stress
    arc_deviation_pct = (1.0 - arc_stability_factor) * 40.0 

    # --- MODÜL İÇERİKLERİ (Tüm uygulama akışı burada) ---
    
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
            fig_health = go.Figure(go.Indicator(mode="gauge+number", value=panel_health_index, title={'text': "Sağlık"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green" if panel_health_index > 50 else "red"}}))
            fig_health.update_layout(height=250)
            st.plotly_chart(fig_health, use_container_width=True)
            if panel_health_index < 40: st.error("🚨 **KRİTİK:** Yüksek termal stres!")
            else: st.success("✅ Sistem Stabil.")

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
            df_cost = pd.DataFrame({"Kalem": ["Hurda", "Elektrik", "Oksijen", "Elektrot"], "Maliyet": [cost_scrap, cost_elec, cost_oxy, cost_electrode]})
            fig_pie = px.pie(df_cost, values='Maliyet', names='Kalem', title="Maliyet Kırılımı", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

    elif selected_module == "3️⃣ Karar Destek Modülü (Process)":
        st.title("📈 Modül 3: Karar Destek ve Dijital İkiz")
        c_left, c_right = st.columns([1, 2])
        with c_left:
            st.subheader("Sıcaklık Tahmini")
            st.plotly_chart(create_gauge_chart(prediction), use_container_width=True)
            st.metric("Ark Sapma Yüzdesi", f"%{arc_deviation_pct:.1f}", delta_color="inverse" if arc_deviation_pct > 20 else "normal")
        with c_right:
            st.subheader("Fırın İçi Akışkan Dinamiği (CFD)")
            X, Y, T, Vx, Vy = generate_cfd_fields(input_data['power_kWh'], arc_deviation_pct)
            fig_cfd, ax = plt.subplots(figsize=(8, 5))
            c = ax.contourf(X, Y, T, levels=25, cmap='inferno')
            ax.quiver(X[::4, ::4], Y[::4, ::4], Vx[::4, ::4], Vy[::4, ::4], color='white', alpha=0.6)
            fig_cfd.colorbar(c, label='Sıcaklık (°C)')
            ax.set_title(f"Havuz ve Akış (Güç: {input_data['power_kWh']} kWh)")
            st.pyplot(fig_cfd)

    elif selected_module == "4️⃣ Alarm, Tavsiye ve KPI'lar":
        st.title("🚨 Modül 4: Alarm Merkezi ve KPI")
        k1, k2, k3 = st.columns(3)
        k1.metric("Ark Stabilite Skoru", f"{arc_stability_factor*100:.1f}")
        k2.metric("Döküm Süresi", f"{input_data.get('tap_time_min', 0):.1f} dk")
        alarm = "YOK" if arc_deviation_pct < 20 else "VAR"
        k3.metric("Aktif Alarm", alarm, delta_color="inverse" if alarm=="VAR" else "normal")
        st.markdown("---")
        st.subheader("Stabilite Geçmişi")
        fig_stab = px.area(trend_df, x="Tarih", y="Arc_Stability_KPI", title="Ark Stabilizasyon Performansı")
        st.plotly_chart(fig_stab, use_container_width=True)

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
        
        revenue = sales_price * monthly_target
        var_cost_total = unit_var_cost * monthly_target
        gross = revenue - var_cost_total
        ebitda = gross - fixed_cost
        
        fig_water = go.Figure(go.Waterfall(
            name="EBITDA", orientation="v",
            measure=["relative", "relative", "total", "relative", "total"],
            x=["Ciro", "Değişken Mal.", "Brüt Kar", "Sabit Gider", "EBITDA"],
            y=[revenue, -var_cost_total, 0, -fixed_cost, 0],
            text=[f"${revenue/1e6:.1f}M", f"-${var_cost_total/1e6:.1f}M", f"${gross/1e6:.1f}M", f"-${fixed_cost/1e6:.1f}M", f"${ebitda/1e6:.1f}M"],
            connector={"line":{"color":"rgb(63, 63, 63)"}},
        ))
        st.plotly_chart(fig_water, use_container_width=True)
        st.metric("EBITDA Marjı", f"%{(ebitda/revenue)*100:.1f}")

    elif selected_module == "6️⃣ Scrap & Purchase Intelligence":
        st.title("🧠 Modül 6: Hurda ve Satınalma Zekası")
        uploaded_scrap = st.file_uploader("Hurda Verisi (CSV)", type=["csv"])
        scrap_df = pd.read_csv(uploaded_scrap) if uploaded_scrap else generate_dummy_scrap_data()
        with st.expander("Veri Önizleme"): st.dataframe(scrap_df.head(), use_container_width=True)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_scatter = px.scatter(scrap_df, x="Price_USD_t", y="Quality_Index", color="Supplier", size="Lot_tonnage", title="Tedarikçi Fiyat/Kalite Matrisi", hover_data=["Scrap_Type"])
            st.plotly_chart(fig_scatter, use_container_width=True)
        with col_s2:
            scrap_df["Energy_Cost"] = scrap_df["kWh_per_t"] * (price_elec / 1000.0)
            scrap_df["True_Cost"] = scrap_df["Price_USD_t"] + scrap_df["Energy_Cost"]
            fig_bar = px.bar(scrap_df.groupby("Supplier")[["Price_USD_t", "True_Cost"]].mean().reset_index(), x="Supplier", y=["Price_USD_t", "True_Cost"], barmode="group", title="Nominal Fiyat vs Gerçek Maliyet")
            st.plotly_chart(fig_bar, use_container_width=True)

if __name__ == "__main__":
    main()
