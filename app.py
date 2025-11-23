import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------
# 1. SAYFA VE LOGO AYARLARI
# ------------------------------------------------------------

# Logo yükleme (varsa)
logo_img = None
page_icon_img = "⚙️"
try:
    logo_img = Image.open("logo.png.png")
    page_icon_img = logo_img
except Exception:
    pass

st.set_page_config(
    page_title="BG Maden AI",
    layout="wide",
    page_icon=page_icon_img,
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# 2. YARDIMCI FONKSİYONLAR & SİMÜLASYON MOTORU
# ------------------------------------------------------------

@st.cache_data
def generate_dummy_trend_data(n_points=50):
    """Trend grafikleri için yapay zaman serisi verisi üretir."""
    dates = pd.date_range(start="2023-01-01", periods=n_points, freq="D")
    
    # Panel Sıcaklık Trendi (Artan trend = Aşınma simülasyonu)
    panel_temps = np.random.normal(35, 5, n_points) + np.linspace(0, 10, n_points)
    
    # Ark Stabilite Trendi (Dalgalı yapı)
    stability = np.random.normal(85, 5, n_points) + np.sin(np.linspace(0, 4*np.pi, n_points)) * 5
    
    return pd.DataFrame({
        "Tarih": dates,
        "Panel_Temp_Avg": panel_temps,
        "Arc_Stability_KPI": stability.clip(0, 100)
    })


def generate_dummy_scrap_data(n_suppliers: int = 4, n_lots: int = 40) -> pd.DataFrame:
    """
    Hurda & tedarikçi analizi için basit demo veri seti üretir.
    Gerçek tesiste burası ERP / hurda sahası CSV çıktısı ile değiştirilebilir.
    """
    np.random.seed(42)
    suppliers = [f"Tedarikçi {chr(65 + i)}" for i in range(n_suppliers)]
    scrap_types = ["HMS 80/20", "HMS 70/30", "Shredded", "Pig Iron", "HBI"]
    
    rows = []
    for i in range(n_lots):
        sup = np.random.choice(suppliers)
        stype = np.random.choice(scrap_types)
        
        # Baz fiyat & kalite
        base_price = np.random.uniform(280, 420)  # $/t
        quality = np.random.uniform(60, 95)       # 0-100
        
        # Verim ve tüketimler kalite ile hafif korele
        yield_pct = np.random.normal(90, 3) + (quality - 75) * 0.1
        yield_pct = np.clip(yield_pct, 82, 98)
        
        kwh_t = np.random.normal(380, 25) - (quality - 75) * 0.8
        kwh_t = np.clip(kwh_t, 320, 430)
        
        elec_kg_t = np.random.normal(1.8, 0.15)
        o2_nm3_t = np.random.normal(220, 20)
        lotsize_t = np.random.uniform(30, 90)
        
        rows.append({
            "Supplier": sup,
            "Scrap_Type": stype,
            "Lot_ID": f"LOT_{i+1:03d}",
            "Price_USD_t": round(base_price, 1),
            "Quality_Index": round(quality, 1),
            "Yield_pct": round(yield_pct, 1),
            "kWh_per_t": round(kwh_t, 1),
            "Electrode_kg_per_t": round(elec_kg_t, 2),
            "O2_Nm3_per_t": round(o2_nm3_t, 1),
            "Lot_tonnage": round(lotsize_t, 1),
        })
    
    return pd.DataFrame(rows)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Ham veriden Termal Stres ve Kalite İndekslerini türetir."""
    df = df.copy()
    
    # Termal Stres İndeksi Hesaplama
    required_thermal_cols = ["panel_T_in_C", "panel_T_out_C", "panel_flow_kg_s", "power_kWh"]
    if all(col in df.columns for col in required_thermal_cols):
        delta_T = df["panel_T_out_C"] - df["panel_T_in_C"]
        flow = df["panel_flow_kg_s"].replace(0, 1)
        power = df["power_kWh"].replace(0, 1)
        
        thermal_stress = (delta_T * power) / flow
        thermal_stress_norm = 100 * (thermal_stress - thermal_stress.min()) / (thermal_stress.max() - thermal_stress.min() + 1e-6)
        df["Thermal_Stress_Index"] = thermal_stress_norm.clip(0, 100)
    else:
        df["Thermal_Stress_Index"] = 50.0  # Varsayılan
    
    # Hurda Kalite İndeksi (demo)
    if "scrap_mix_ratio" in df.columns:
        scrap_quality = 60 + (df["scrap_mix_ratio"] * 40)
        df["Scrap_Quality_Index"] = scrap_quality.clip(0, 100)
    else:
        df["Scrap_Quality_Index"] = np.random.uniform(50, 90, len(df))
    
    # Tap Time tahmini için dummy kolon
    if "tap_time_min" not in df.columns:
        df["tap_time_min"] = np.random.uniform(45, 70, len(df))
    
    return df


def train_model(df: pd.DataFrame, target_col: str):
    """Random Forest modeli eğitir ve geri döner."""
    df = df.dropna(subset=[target_col])
    feature_cols = [c for c in df.columns if c not in [target_col, "heat_id", "date", "Tarih"]]
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, feature_cols, mae, r2


def predict_with_model(model, feature_cols, input_data: dict):
    x = np.array([input_data.get(col, 0) for col in feature_cols]).reshape(1, -1)
    return model.predict(x)[0]


# ------------------------------------------------------------
# 3. UYGULAMA AKIŞI
# ------------------------------------------------------------
def main():
    # --- SOL MENÜ: LOGO VE NAVİGASYON ---
    if logo_img:
        st.sidebar.image(logo_img, use_container_width=True)
    
    st.sidebar.title("BG Maden AI")
    st.sidebar.markdown("**Akıllı Karar Destek Sistemi**")
    st.sidebar.markdown("---")
    
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

    # --- VERİ YÜKLEME VE MODEL EĞİTİMİ ---
    try:
        df = pd.read_csv("data/BG_EAF_panelcooling_demo.csv")
    except FileNotFoundError:
        st.error("❌ Veri dosyası bulunamadı! Lütfen 'data/BG_EAF_panelcooling_demo.csv' yolunu kontrol edin.")
        st.stop()

    # Feature Engineering
    df = feature_engineering(df)
    
    # Model Eğitimi (Random Forest)
    target_col = "tap_temperature_C"
    drop_cols = ["heat_id", "date", "Tarih"]
    X_cols = [c for c in df.columns if c not in drop_cols + [target_col]]
    X = df[X_cols]
    y = df[target_col]
    
    if len(df) > 10:
        model, feature_cols, mae, r2 = train_model(df, target_col)
    else:
        model, feature_cols, mae, r2 = None, X_cols, None, None
    
    # Trend Verisi
    trend_df = generate_dummy_trend_data()
    tonnage = 10.0  # Varsayılan tonaj

    # ------------------------------------------------------------------
    # ORTAK GİRDİLER (SIDEBAR - KONTROL PANELİ)
    # ------------------------------------------------------------------
    st.sidebar.header("🎛️ Simülasyon Kontrol Paneli")
    
    input_data = {}

    # 1. Ark Stabilizasyonu (En Kritik Girdi)
    arc_stability_factor = st.sidebar.slider(
        "⚡ Ark Stabilizasyon Faktörü (0-1)", 
        0.0, 1.0, 0.90, 0.01,
        help="1.0 = Tam Merkezde/Stabil. Düşük değer = Yüksek Sapma/Risk."
    )
    
    # Ark stabilitesine göre Termal Stres ve Sapma Yüzdesi türetilir
    calculated_stress = (1.0 - arc_stability_factor) * 100
    input_data['Thermal_Stress_Index'] = calculated_stress
    
    # 2. Proses Girdileri
    for col in X.columns:
        if col == 'power_kWh':
            input_data[col] = st.sidebar.slider("Güç (kWh)", 3000.0, 5000.0, 4000.0)
        elif col == 'oxygen_Nm3':
            input_data[col] = st.sidebar.slider("Oksijen (Nm3)", 100.0, 300.0, 200.0)
        elif col == 'Scrap_Quality_Index':
            input_data[col] = st.sidebar.slider("Hurda Kalitesi (0-100)", 0.0, 100.0, 70.0)
        elif col == 'tap_time_min':
            input_data[col] = st.sidebar.slider("Tap Süresi (dk)", 40.0, 80.0, 55.0)
        else:
            # Diğer tüm numeric kolonlar için varsayılan slider
            if np.issubdtype(df[col].dtype, np.number):
                min_val = float(df[col].quantile(0.05))
                max_val = float(df[col].quantile(0.95))
                default_val = float(df[col].median())
                input_data[col] = st.sidebar.slider(col, min_val, max_val, default_val)
            else:
                # Kategorik ise ilk değeri al
                input_data[col] = df[col].iloc[0]

    # Model tahmini (tap sıcaklığı)
    prediction = None
    if model is not None:
        prediction = predict_with_model(model, feature_cols, input_data)

    # Panel Health Index
    panel_health_index = 100 - calculated_stress
    arc_deviation_pct = (1.0 - arc_stability_factor) * 100

    # Fiyat parametreleri (genel)
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Temel Fiyat Parametreleri")
    price_scrap = st.sidebar.number_input("Hurda Fiyatı ($/t)", 200, 600, 350)
    price_elec = st.sidebar.number_input("Elektrik Fiyatı ($/MWh)", 30, 200, 90)
    price_oxy = st.sidebar.number_input("Oksijen Fiyatı ($/Nm³)", 0.02, 1.00, 0.08, step=0.01)
    price_electrode = st.sidebar.number_input("Elektrot Fiyatı ($/kg)", 2.0, 15.0, 4.0, step=0.5)

    # ------------------------------------------------------------------
    # MODÜL 1: AI BAKIM VE DURUŞ ENGELLEME
    # ------------------------------------------------------------------
    if selected_module == "1️⃣ AI Bakım ve Duruş Engelleme":
        st.title("🛠️ Modül 1: AI Bakım & Duruş Engelleme")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Panel Sağlık İndeksi")
            fig1, ax1 = plt.subplots()
            ax1.barh(["Panel Health"], [panel_health_index], color="green" if panel_health_index > 60 else "red")
            ax1.set_xlim(0, 100)
            ax1.set_xlabel("Sağlık (%)")
            st.pyplot(fig1)
            
            st.markdown(f"**Termal Stres İndeksi:** {calculated_stress:.1f} / 100")
            st.markdown(f"**Ark Sapma Yüzdesi:** {arc_deviation_pct:.1f}%")

        with col2:
            st.subheader("Zaman İçinde Panel Sıcaklığı & Ark Stabilitesi")
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=trend_df["Tarih"], y=trend_df["Panel_Temp_Avg"],
                mode="lines", name="Panel Sıcaklık (°C)"
            ))
            fig_trend.add_trace(go.Scatter(
                x=trend_df["Tarih"], y=trend_df["Arc_Stability_KPI"],
                mode="lines", name="Ark Stabilite KPI"
            ))
            fig_trend.update_layout(
                xaxis_title="Tarih",
                yaxis_title="Değer",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("---")
        st.subheader("AI Tabanlı Durum Değerlendirmesi")

        if panel_health_index > 80 and arc_deviation_pct < 10:
            st.success("✅ Panel ve ark koşulları oldukça sağlıklı. Planlı bakım aralığı korunabilir.")
        elif panel_health_index > 50:
            st.warning("⚠️ Orta seviyede stres tespit edildi. Bir sonraki kampanya öncesi detaylı gözlem önerilir.")
        else:
            st.error("🚨 Yüksek termal stres ve ark sapması! Panel delinme ve ani duruş riski çok yüksek.")

        st.markdown("**Not:** Bu ekran, bakım ekibine 'nerede ne oluyor' bilgisini hızlıca göstererek, plansız duruşları azaltmayı hedefler.")

    # ------------------------------------------------------------------
    # MODÜL 2: AI GİRDİ MALİYETLERİ DÜŞÜRME
    # ------------------------------------------------------------------
    elif selected_module == "2️⃣ AI Girdi Maliyetleri Düşürme":
        st.title("💰 Modül 2: Girdi Maliyetleri Optimizasyonu")
        
        # Maliyet Hesabı
        c_scrap = tonnage * price_scrap
        c_elec = input_data['power_kWh'] * price_elec
        c_oxy = input_data['oxygen_Nm3'] * price_oxy
        c_elec_rod = tonnage * 1.8 * price_electrode
        total = c_scrap + c_elec + c_oxy + c_elec_rod
        unit_cost = total / tonnage

        col_pie, col_metric = st.columns([1, 1])
        
        with col_pie:
            st.subheader("Maliyet Kırılımı")
            df_cost = pd.DataFrame({
                "Kalem": ["Hurda", "Elektrik", "Oksijen", "Elektrot"],
                "Tutar": [c_scrap, c_elec, c_oxy, c_elec_rod]
            })
            fig_pie = px.pie(df_cost, values='Tutar', names='Kalem', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_metric:
            st.subheader("Ton Başına Toplam Maliyet")
            st.metric("Toplam Maliyet ($/t)", f"{unit_cost:.1f}")

            st.markdown("**AI Önerisi:**")
            st.markdown("- Hurda kalitesini 5 puan artırmak, enerji tüketimini ~%2 düşürebilir.")
            st.markdown("- Ark stabilizasyonunu iyileştirmek, panel kayıpları ve dolaylı duruş maliyetini azaltır.")

    # ------------------------------------------------------------------
    # MODÜL 3: KARAR DESTEK MODÜLÜ (PROCESS)
    # ------------------------------------------------------------------
    elif selected_module == "3️⃣ Karar Destek Modülü (Process)":
        st.title("📈 Modül 3: Karar Destek (Process)")
        
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("Tap Sıcaklığı Tahmini")
            if prediction is not None:
                st.metric("Tahmini Tap Sıcaklığı (°C)", f"{prediction:.1f}")
            else:
                st.info("Model henüz eğitilmedi veya yeterli veri yok.")
            
            st.markdown("Bu tahmin; güç, oksijen, hurda kalitesi ve termal stres gibi parametrelere göre üretilmiştir.")
        
        with col_right:
            st.subheader("Model Performansı")
            if mae is not None:
                st.metric("MAE (°C)", f"{mae:.1f}")
                st.metric("R² Skoru", f"{r2:.2f}")
            else:
                st.info("Model değerlendirme metriği için yeterli veri yok.")

        st.markdown("---")
        st.subheader("Operasyonel Karar Önerileri")
        if prediction and prediction < 1600:
            st.warning("Tap sıcaklığı hedefin altında. Oksijen ve güç set değerleri gözden geçirilmeli.")
        elif prediction and prediction > 1680:
            st.warning("Tap sıcaklığı yüksek. Aşırı aşınma ve enerji israfı riski var.")
        else:
            st.success("Tap sıcaklığı hedef bandında.")

    # ------------------------------------------------------------------
    # MODÜL 4: ALARM, TAVSİYE VE KPI'LAR
    # ------------------------------------------------------------------
    elif selected_module == "4️⃣ Alarm, Tavsiye ve KPI'lar":
        st.title("🚨 Modül 4: Alarm, Tavsiye ve KPI Paneli")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Ark Stabilite KPI", f"{arc_stability_factor*100:.1f}", delta="Hedef > 85")
        k2.metric("Enerji Tüketimi", f"{(input_data['power_kWh']/tonnage):.1f} kWh/t")
        k3.metric("Döküm Süresi", f"{input_data['tap_time_min']:.0f} dk")
        alarm_count = 1 if arc_deviation_pct > 20 else 0
        k4.metric("Aktif Alarm", f"{alarm_count}", delta_color="inverse")
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Stabilite Geçmişi")
            fig_stab = px.area(trend_df, x="Tarih", y="Arc_Stability_KPI", title="Ark Stabilizasyon Performansı")
            st.plotly_chart(fig_stab, use_container_width=True)
            
        with c2:
            st.subheader("📋 Operatör Aksiyon Paneli")
            if alarm_count > 0:
                st.error("🛑 **ACİL AKSİYON:** Ark sapması sınır değerin üzerinde! DC akım dengesini kontrol edin.")
            elif prediction and prediction > 1650:
                st.warning("⚠️ **UYARI:** Aşırı ısınma. Güç kademesini düşürün.")
            else:
                st.success("✅ **DURUM:** Proses optimum aralıkta çalışıyor.")

    # ------------------------------------------------------------------
    # MODÜL 6: SCRAP & PURCHASE INTELLIGENCE
    # ------------------------------------------------------------------
    elif selected_module == "6️⃣ Scrap & Purchase Intelligence":
        st.title("🧠 Modül 6: Scrap & Purchase Intelligence")
        st.markdown(
            "Bu modül, **hurda tedarikçilerini**, hurda kalitesini ve gerçek ton maliyetini (True Cost) "
            "karşılaştırarak satınalmaya **veriyle konuşma** imkânı verir."
        )

        # --------------------------------------------------------------
        # 6.1 Hurda / Tedarikçi Datası
        # --------------------------------------------------------------
        st.subheader("1️⃣ Hurda & Tedarikçi Datası")

        uploaded_scrap = st.file_uploader(
            "Hurda lot datası (CSV) yükle – yoksa demo veri kullanılacak.",
            type=["csv"],
            key="scrap_csv",
        )

        if uploaded_scrap is not None:
            scrap_df = pd.read_csv(uploaded_scrap)
            st.success(f"Yüklendi: {scrap_df.shape[0]} satır, {scrap_df.shape[1]} kolon.")
        else:
            st.info("Demo veri seti kullanılıyor. Kendi ERP / hurda sahası CSV'in ile değiştirebilirsin.")
            scrap_df = generate_dummy_scrap_data()

        with st.expander("Ham Veri Önizlemesi", expanded=False):
            st.dataframe(scrap_df.head(50), use_container_width=True)

        # Zorunlu kolon kontrolü (yoksa kullanıcıya bilgi ver)
        required_cols = {
            "Supplier",
            "Scrap_Type",
            "Price_USD_t",
            "Quality_Index",
            "Yield_pct",
            "kWh_per_t",
            "Electrode_kg_per_t",
            "O2_Nm3_per_t",
        }
        if not required_cols.issubset(set(scrap_df.columns)):
            st.warning(
                "⚠️ Demo dışı veri kullanacaksan şu kolonlara ihtiyacın var:\\n\\n"
                f"{', '.join(sorted(required_cols))}"
            )

        # --------------------------------------------------------------
        # 6.2 Fiyat ve Enerji Parametreleri
        # --------------------------------------------------------------
        st.subheader("2️⃣ Fiyat & Enerji Parametreleri")

        c1_m6, c2_m6, c3_m6, c4_m6 = st.columns(4)
        price_elec_mwh = c1_m6.number_input("Elektrik Fiyatı ($/MWh)", 30.0, 300.0, float(price_elec))
        price_electrode_m6 = c2_m6.number_input("Elektrot Fiyatı ($/kg)", 2.0, 20.0, float(price_electrode))
        price_o2_m6 = c3_m6.number_input("Oksijen Fiyatı ($/Nm³)", 0.01, 1.0, float(price_oxy))
        overhead_factor = c4_m6.number_input("Hurda dışı değişken maliyet (+%)", 0.0, 50.0, 10.0)

        df_calc = scrap_df.copy()

        # Enerji kWh/t -> $/t
        df_calc["Energy_Cost_USD_t"] = df_calc["kWh_per_t"] * (price_elec_mwh / 1000.0)
        df_calc["Electrode_Cost_USD_t"] = df_calc["Electrode_kg_per_t"] * price_electrode_m6
        df_calc["O2_Cost_USD_t"] = df_calc["O2_Nm3_per_t"] * price_o2_m6

        df_calc["Process_Cost_USD_t"] = (
            df_calc["Energy_Cost_USD_t"]
            + df_calc["Electrode_Cost_USD_t"]
            + df_calc["O2_Cost_USD_t"]
        )
        df_calc["True_Cost_USD_t"] = (df_calc["Price_USD_t"] + df_calc["Process_Cost_USD_t"]) * (
            1 + overhead_factor / 100.0
        )

        # --------------------------------------------------------------
        # 6.3 Tedarikçi Skor Kartı
        # --------------------------------------------------------------
        st.subheader("3️⃣ Tedarikçi Skor Kartı")

        agg = {
            "Price_USD_t": "mean",
            "True_Cost_USD_t": "mean",
            "Yield_pct": "mean",
            "kWh_per_t": "mean",
            "Electrode_kg_per_t": "mean",
            "O2_Nm3_per_t": "mean",
            "Lot_tonnage": "sum" if "Lot_tonnage" in df_calc.columns else "count",
            "Quality_Index": "mean",
        }
        supplier_summary = df_calc.groupby("Supplier").agg(agg).reset_index()

        # Basit skor: düşük maliyet + yüksek verim + yüksek kalite
        eps = 1e-6
        cost_norm = (supplier_summary["True_Cost_USD_t"] - supplier_summary["True_Cost_USD_t"].min()) / (
            supplier_summary["True_Cost_USD_t"].max() - supplier_summary["True_Cost_USD_t"].min() + eps
        )
        yield_norm = (supplier_summary["Yield_pct"] - supplier_summary["Yield_pct"].min()) / (
            supplier_summary["Yield_pct"].max() - supplier_summary["Yield_pct"].min() + eps
        )
        qual_norm = (supplier_summary["Quality_Index"] - supplier_summary["Quality_Index"].min()) / (
            supplier_summary["Quality_Index"].max() - supplier_summary["Quality_Index"].min() + eps
        )

        supplier_summary["Supplier_Score"] = (1 - cost_norm) * 0.5 + yield_norm * 0.25 + qual_norm * 0.25

        st.dataframe(
            supplier_summary[
                [
                    "Supplier",
                    "True_Cost_USD_t",
                    "Price_USD_t",
                    "Yield_pct",
                    "Quality_Index",
                    "Supplier_Score",
                ]
            ].sort_values("Supplier_Score", ascending=False),
            use_container_width=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            fig_tc = px.bar(
                supplier_summary,
                x="Supplier",
                y="True_Cost_USD_t",
                title="Gerçek Ton Maliyeti (True Cost $/t)",
            )
            st.plotly_chart(fig_tc, use_container_width=True)

        with col_b:
            fig_sc = px.bar(
                supplier_summary,
                x="Supplier",
                y="Supplier_Score",
                title="Toplam Tedarikçi Skoru (0-1)",
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        # --------------------------------------------------------------
        # 6.4 Senaryo Simülatörü (What-if) & Yıllık Tasarruf
        # --------------------------------------------------------------
        st.subheader("4️⃣ Senaryo Simülatörü & Yıllık Tasarruf")

        st.markdown(
            "Aşağıdan tedarikçilerin mix oranlarını belirleyerek yeni bir **hurda stratejisi** oluştur. "
            "Model, mevcut duruma göre potansiyel **yıllık tasarrufu** hesaplar."
        )

        suppliers_list = list(supplier_summary["Supplier"])
        mix_rows = []
        for sup in suppliers_list:
            share = st.slider(f"{sup} karışım oranı (%)", 0, 100, 0, key=f"mix_{sup}")
            if share > 0:
                mix_rows.append((sup, share))

        total_share = sum(s for _, s in mix_rows)

        if total_share == 0:
            st.info("En az bir tedarikçi için % oranı girerek senaryonu oluştur.")
        else:
            # Oranları 1'e normalize et
            mix_df = pd.DataFrame(
                [(sup, s / total_share) for sup, s in mix_rows],
                columns=["Supplier", "Share"],
            )
            mix_merged = mix_df.merge(supplier_summary, on="Supplier")

            mix_true_cost = (mix_merged["True_Cost_USD_t"] * mix_merged["Share"]).sum()
            mix_yield = (mix_merged["Yield_pct"] * mix_merged["Share"]).sum()

            baseline_weights = supplier_summary["Lot_tonnage"] / supplier_summary["Lot_tonnage"].sum()
            baseline_true_cost = (supplier_summary["True_Cost_USD_t"] * baseline_weights).sum()

            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Yeni Mix True Cost ($/t)", f"{mix_true_cost:.1f}")
            col_m2.metric("Yeni Mix Ortalama Verim (%)", f"{mix_yield:.1f}")

            annual_tonnage = st.number_input(
                "Yıllık hurda tonajı (t/yıl)",
                min_value=10_000.0,
                max_value=1_000_000.0,
                value=200_000.0,
                step=10_000.0,
            )

            annual_saving = (baseline_true_cost - mix_true_cost) * annual_tonnage

            if annual_saving >= 0:
                st.success(f"💰 **Yıllık Tasarruf Potansiyeli:** ${annual_saving:,.0f}")
            else:
                st.error(f"⚠️ Bu senaryo mevcut duruma göre yıllık **-${abs(annual_saving):,.0f}** ek maliyet yaratıyor.")

            with st.expander("Satınalma Sunumu İçin Özet Cümle", expanded=False):
                direction = "tasarruf" if annual_saving >= 0 else "ek maliyet"
                st.write(
                    f"Son {len(scrap_df)} ısı verisine göre, mevcut hurda karışımımızın gerçek ton maliyeti "
                    f"yaklaşık **${baseline_true_cost:,.1f}/t**. Önerilen yeni tedarikçi karışımı ile "
                    f"True Cost **${mix_true_cost:,.1f}/t** seviyesine geliyor; yıllık {annual_tonnage:,.0f} ton için "
                    f"yaklaşık **${abs(annual_saving):,.0f} {direction}** oluşuyor."
                )

    # ------------------------------------------------------------------
    # MODÜL 5: AI ENTERPRISE LEVEL (EBITDA)
    # ------------------------------------------------------------------
    elif selected_module == "5️⃣ AI Enterprise Level (EBITDA)":
        st.title("🏢 Modül 5: Kurumsal İş Zekası (EBITDA)")
        
        with st.expander("📊 Finansal Hedef Ayarları", expanded=True):
            c_e1, c_e2 = st.columns(2)
            sales_price = c_e1.number_input("Hedef Satış Fiyatı ($/ton)", 500, 2000, 900)
            monthly_target = c_e2.number_input("Aylık Hedef Tonaj", 1000, 50000, 10000, step=1000)
            
            c_e3, c_e4 = st.columns(2)
            var_cost_other = c_e3.number_input("Diğer Değişken Maliyetler ($/ton)", 0, 300, 80)
            fixed_cost = c_e4.number_input("Aylık Sabit Maliyetler ($)", 0, 5_000_000, 1_000_000, step=50_000)

        # Basit EBITDA Hesabı
        revenue = sales_price * monthly_target
        var_cost_total = (price_scrap + (price_elec * 0.4) + (price_oxy * 10) + price_electrode + var_cost_other) * monthly_target
        gross = revenue - var_cost_total
        ebitda = gross - fixed_cost
        
        # Waterfall Grafiği
        fig_water = go.Figure(go.Waterfall(
            name = "EBITDA",
            orientation = "v",
            measure = ["relative", "relative", "relative", "relative", "total"],
            x = ["Ciro", "Değişken Maliyetler", "Sabit Maliyetler", "Diğer", "EBITDA"],
            textposition = "outside",
            text = [
                f"{revenue/1e6:.1f}M",
                f"-{var_cost_total/1e6:.1f}M",
                f"-{fixed_cost/1e6:.1f}M",
                "",
                f"{ebitda/1e6:.1f}M"
            ],
            y = [revenue, -var_cost_total, -fixed_cost, 0, ebitda],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig_water.update_layout(title="Aylık Karlılık Şelalesi ($)", showlegend=False)
        st.plotly_chart(fig_water, use_container_width=True)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Aylık Ciro", f"${revenue:,.0f}")
        m2.metric("EBITDA", f"${ebitda:,.0f}", delta_color="normal" if ebitda>0 else "inverse")
        m3.metric("EBITDA Marjı", f"%{(ebitda/revenue)*100:.1f}")

if __name__ == "__main__":
    main()
