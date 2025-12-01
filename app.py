import os
import json
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import altair as alt

# ----------------------------------------------
# GENEL AYARLAR
# ----------------------------------------------
st.set_page_config(
    page_title="FeCr AI",
    page_icon="apple-touch-icon.png",
    layout="wide",
)

TZ = ZoneInfo("Europe/Istanbul")

SETUP_SAVE_PATH = "data/saved_inputs.json"
RUNTIME_SAVE_PATH = "data/runtime_data.json"
os.makedirs("data", exist_ok=True)

# ----------------------------------------------
# SETUP VERİLERİ
# ----------------------------------------------
if os.path.exists(SETUP_SAVE_PATH):
    with open(SETUP_SAVE_PATH, "r", encoding="utf-8") as f:
        saved_inputs = json.load(f)
else:
    saved_inputs = {}

if "info_state" not in st.session_state:
    st.session_state.info_state = {}

# ----------------------------------------------
# RUNTIME VERİLERİ
# ----------------------------------------------
def load_runtime_data():
    if os.path.exists(RUNTIME_SAVE_PATH):
        try:
            with open(RUNTIME_SAVE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []

def save_runtime_data(data_list):
    try:
        with open(RUNTIME_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Runtime verileri kaydedilemedi: {e}")

runtime_data = load_runtime_data()

# ----------------------------------------------
# SİMÜLASYON VERİ ÜRETİCİSİ
# ----------------------------------------------
def generate_simulation_runtime_data(n: int = 15):
    sim_list = []
    now = datetime.now(TZ)
    for i in range(n):
        ts = now - timedelta(hours=(n - 1 - i))
        heat_id = f"SIM-{i+1}"
        tap_weight = 35 + random.uniform(-3, 3)
        kwh_per_t = 420 + random.uniform(-25, 25)
        energy_kwh = tap_weight * kwh_per_t
        duration_min = 55 + random.uniform(-10, 10)
        tap_temp = 1610 + random.uniform(-15, 15)
        o2_flow = 950 + random.uniform(-150, 150)
        slag_foaming = random.randint(3, 9)
        panel_delta_t = 18 + random.uniform(-5, 8)
        electrode_cons = 1.9 + random.uniform(-0.3, 0.3)
        sim_list.append(
            {
                "timestamp": ts.isoformat(),
                "heat_id": heat_id,
                "tap_weight_t": tap_weight,
                "duration_min": duration_min,
                "energy_kwh": energy_kwh,
                "tap_temp_c": tap_temp,
                "o2_flow_nm3h": o2_flow,
                "slag_foaming_index": slag_foaming,
                "panel_delta_t_c": panel_delta_t,
                "electrode_kg_per_heat": electrode_cons,
                "kwh_per_t": kwh_per_t,
                "operator_note": "Simülasyon kaydı",
            }
        )
    return sim_list

# ----------------------------------------------
# EXCEL'DEN SORU TABLOLARI
# ----------------------------------------------
@st.cache_data
def load_sheets():
    file_name = "dc_saf_soru_tablosu.xlsx"
    try:
        xls = pd.read_excel(file_name, sheet_name=None)
        cleaned = {}
        for name, df in xls.items():
            df2 = df.dropna(how="all")
            if not df2.empty:
                cleaned[name] = df2
        return cleaned
    except Exception as e:
        st.error(f"Excel dosyası yüklenemedi: {e}")
        return {}

# ----------------------------------------------
# 1) SETUP SAYFASI
# ----------------------------------------------
def show_setup_form():
    st.markdown("## 1. Setup – Sabit Proses / Tasarım Verileri")
    st.markdown(
        "Bu form **dc_saf_soru_tablosu.xlsx** dosyasına göre hazırlanmıştır.\n\n"
        "1. Girişi sadece **Set Değeri** alanına yapınız.\n"
        "2. 🔴 Zorunlu (Önem: 1), 🟡 Faydalı (Önem: 2), ⚪ Opsiyonel (Önem: 3).\n"
        "3. Detaylı bilgi ve açıklama için ℹ️ simgesine tıklayınız."
    )

    sheets = load_sheets()
    if not sheets:
        return

    total_fields = 0
    total_filled = 0
    required_fields = 0
    required_filled = 0

    for sheet_idx, (sheet_name, df) in enumerate(sheets.items(), start=1):
        with st.expander(f"{sheet_idx}. {sheet_name}", expanded=(sheet_idx == 1)):
            df.columns = [str(c).strip() for c in df.columns]
            unit_cols = [c for c in df.columns if "set" in c.lower()]
            unit_col_name = unit_cols[0] if unit_cols else None

            for row_idx, row in df.iterrows():
                row_key = f"{sheet_idx}_{row_idx}"

                onem_raw = row.get("Önem", 3)
                try:
                    onem = int(onem_raw)
                except Exception:
                    onem = 3
                renk = {1: "🔴", 2: "🟡", 3: "⚪"}.get(onem, "⚪")

                raw_birim = row.get(unit_col_name, "") if unit_col_name else ""
                birim = ""
                if pd.notna(raw_birim):
                    birim_str = str(raw_birim).strip()
                    if birim_str.lower() not in ("", "none", "nan"):
                        birim = birim_str

                tag = row.get("Tag", "")
                val_key = f"{sheet_name}|{tag}"

                c1, c2, c3, c4, c5 = st.columns([2.2, 2.5, 4.0, 2.5, 0.7])
                c1.markdown(f"**{tag}**")
                c2.markdown(f"{renk} {row.get('Değişken', '')}")
                c3.markdown(str(row.get("Açıklama", "")))

                current_val = saved_inputs.get(val_key, "")
                with c4:
                    ic, uc = st.columns([5, 2])
                    with ic:
                        new_val = st.text_input(
                            label="",
                            value=current_val,
                            key=val_key,
                            label_visibility="collapsed",
                        )
                        if new_val != current_val:
                            saved_inputs[val_key] = new_val
                            with open(SETUP_SAVE_PATH, "w", encoding="utf-8") as f:
                                json.dump(saved_inputs, f, ensure_ascii=False, indent=2)
                    with uc:
                        if birim:
                            st.markdown(f"**{birim}**")
                        else:
                            st.markdown("")

                with c5:
                    if st.button("ℹ️", key=f"info_{row_key}"):
                        st.session_state.info_state[row_key] = not st.session_state.info_state.get(row_key, False)

                if st.session_state.info_state.get(row_key, False):
                    detaylar = []
                    da = row.get("Detaylı Açıklama")
                    if isinstance(da, str) and da.strip():
                        detaylar.append("🔷 **Detaylı Açıklama:** " + da)
                    vk = row.get("Veri Kaynağı")
                    if isinstance(vk, str) and vk.strip():
                        detaylar.append("📌 **Kaynak:** " + vk)
                    ka = row.get("Kayıt Aralığı")
                    if isinstance(ka, str) and ka.strip():
                        detaylar.append("⏱️ **Kayıt Aralığı:** " + ka)
                    if pd.notna(onem_raw):
                        detaylar.append("🔵 **Önem:** " + str(onem))
                    if detaylar:
                        st.info("\n".join(detaylar))

                total_fields += 1
                kayit_degeri = str(saved_inputs.get(val_key, "")).strip()
                if kayit_degeri:
                    total_filled += 1
                    if onem == 1:
                        required_filled += 1
                if onem == 1:
                    required_fields += 1

    st.sidebar.subheader("📊 Setup Veri Giriş Durumu")
    pct_all = round(100 * total_filled / total_fields, 1) if total_fields else 0.0
    pct_req = round(100 * required_filled / required_fields, 1) if required_fields else 0.0
    st.sidebar.metric("Toplam Giriş Oranı", f"{pct_all}%")
    st.sidebar.progress(min(pct_all / 100, 1.0))
    st.sidebar.metric("Zorunlu Veri Girişi", f"{pct_req}%")
    st.sidebar.progress(min(pct_req / 100, 1.0))
    eksik = required_fields - required_filled
    if eksik > 0:
        st.sidebar.warning(f"❗ Eksik Zorunlu Değerler: {eksik}")

# ----------------------------------------------
# 2) CANLI VERİ SAYFASI
# ----------------------------------------------
def show_runtime_page(sim_mode: bool):
    st.markdown("## 2. Canlı Veri – Şarj Bazlı Anlık Veriler")
    if sim_mode:
        st.info("🧪 **Simülasyon Modu Aktif.** Veriler simülasyon amaçlı oluşturulur ve kaydedilmez.")
    else:
        st.markdown(
            "Bu sayfada fırın işletmesi sırasında her **şarj / heat** için toplanan "
            "operasyonel veriler girilir veya otomasyon sisteminden okunur."
        )

    with st.form("runtime_form", clear_on_submit=True):
        st.markdown("### Yeni Şarj Kaydı Ekle")
        c1, c2, c3 = st.columns(3)
        with c1:
            heat_id = st.text_input("Heat ID / Şarj No", "")
        with c2:
            tap_weight = st.number_input("Tap Weight (ton)", min_value=0.0, step=0.1)
        with c3:
            duration_min = st.number_input("Toplam Süre (dk)", min_value=0.0, step=1.0)

        c4, c5, c6 = st.columns(3)
        with c4:
            energy_kwh = st.number_input("Toplam Enerji (kWh)", min_value=0.0, step=10.0)
        with c5:
            tap_temp = st.number_input("Tap Sıcaklığı (°C)", min_value=0.0, max_value=2000.0, step=1.0)
        with c6:
            o2_flow = st.number_input("Ortalama O2 Debisi (Nm³/h)", min_value=0.0, step=1.0)

        c7, c8, c9 = st.columns(3)
        with c7:
            slag_foaming = st.slider("Slag Foaming Seviyesi (0–10)", 0, 10, 5)
        with c8:
            panel_delta_t = st.number_input("Panel ΔT (°C)", min_value=0.0, step=0.1)
        with c9:
            electrode_cons = st.number_input("Elektrot Tüketimi (kg/şarj)", min_value=0.0, step=0.01)

        note = st.text_input("Operatör Notu (opsiyonel)", "")

        submitted = st.form_submit_button("Kaydet")

    if submitted:
        if not heat_id:
            st.error("Heat ID / Şarj No girilmesi zorunludur.")
        else:
            if sim_mode:
                st.warning("Simülasyon Modu açıkken yeni veri kalıcı olarak kaydedilmez.")
            else:
                now = datetime.now(TZ).isoformat()
                kwh_per_t = energy_kwh / tap_weight if tap_weight > 0 else None
                new_entry = {
                    "timestamp": now,
                    "heat_id": heat_id,
                    "tap_weight_t": tap_weight,
                    "duration_min": duration_min,
                    "energy_kwh": energy_kwh,
                    "tap_temp_c": tap_temp,
                    "o2_flow_nm3h": o2_flow,
                    "slag_foaming_index": slag_foaming,
                    "panel_delta_t_c": panel_delta_t,
                    "electrode_kg_per_heat": electrode_cons,
                    "kwh_per_t": kwh_per_t,
                    "operator_note": note,
                }
                runtime_data.append(new_entry)
                save_runtime_data(runtime_data)
                st.success(f"Şarj kaydı eklendi: {heat_id}")

    data_source = generate_simulation_runtime_data() if sim_mode else runtime_data
    if not data_source:
        st.info("Henüz canlı veri girilmemiş.")
        return

    df = pd.DataFrame(data_source)
    try:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp_dt")
    except Exception:
        df["timestamp_dt"] = df["timestamp"]

    st.markdown("### Kayıtlı Canlı Veriler")
    st.dataframe(
        df[
            [
                "timestamp_dt",
                "heat_id",
                "tap_weight_t",
                "duration_min",
                "energy_kwh",
                "kwh_per_t",
                "tap_temp_c",
                "electrode_kg_per_heat",
                "slag_foaming_index",
                "panel_delta_t_c",
            ]
        ].rename(
            columns={
                "timestamp_dt": "Zaman",
                "heat_id": "Heat ID",
                "tap_weight_t": "Tap Weight (t)",
                "duration_min": "Süre (dk)",
                "energy_kwh": "Enerji (kWh)",
                "kwh_per_t": "kWh/t",
                "tap_temp_c": "Tap T (°C)",
                "electrode_kg_per_heat": "Elektrot (kg/şarj)",
                "slag_foaming_index": "Slag Foaming",
                "panel_delta_t_c": "Panel ΔT (°C)",
            }
        ),
        use_container_width=True,
    )

    st.markdown("### Basit Trendler (Canlı Veri)")
    chart_df = df.set_index("timestamp_dt")[["kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"]]
    st.line_chart(chart_df)

# ----------------------------------------------
# 3) ARC OPTIMIZER SAYFASI
# ----------------------------------------------
def show_arc_optimizer_page(sim_mode: bool):
    st.markdown("## 3. Arc Optimizer – Trendler, KPI ve Öneriler")
    if sim_mode:
        st.info("🧪 **Simülasyon Modu Aktif.** Arc Optimizer çıktıları simüle edilen veri üzerinden hesaplanmaktadır.")
    else:
        st.markdown(
            "Bu sayfa, canlı veriler üzerinden **enerji verimliliği**, "
            "**elektrot tüketimi** ve **proses stabilitesi** ile ilgili özet KPI ve "
            "modelin önerilerini gösterir."
        )

    data_source = generate_simulation_runtime_data() if sim_mode else runtime_data
    if not data_source:
        st.info("Arc Optimizer çıktıları için henüz canlı veri yok. Önce 2. sayfadan veri ekleyin.")
        return

    df = pd.DataFrame(data_source)
    try:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp_dt")
    except Exception:
        df["timestamp_dt"] = df["timestamp"]

    last = df.iloc[-1]
    last_n = df.tail(10)

    avg_kwh_t = last_n["kwh_per_t"].dropna().mean()
    avg_electrode = last_n["electrode_kg_per_heat"].dropna().mean()
    avg_tap_temp = last_n["tap_temp_c"].dropna().mean()

    if len(df) >= 10 and df["kwh_per_t"].notna().sum() >= 10:
        first5 = df["kwh_per_t"].dropna().head(5).mean()
        last5 = df["kwh_per_t"].dropna().tail(5).mean()
        saving_potential = max(0.0, first5 - last5)
    else:
        saving_potential = 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Son Şarj kWh/t", f"{last['kwh_per_t']:.1f}" if pd.notna(last.get("kwh_per_t", None)) else "-")
    c2.metric(
        "Son Şarj Elektrot",
        f"{last['electrode_kg_per_heat']:.2f} kg/şarj" if pd.notna(last.get("electrode_kg_per_heat", None)) else "-",
    )
    c3.metric(
        "Son Tap Sıcaklığı",
        f"{last['tap_temp_c']:.0f} °C" if pd.notna(last.get("tap_temp_c", None)) else "-",
    )
    c4.metric("Son 10 Şarj Ort. kWh/t", f"{avg_kwh_t:.1f}" if avg_kwh_t and not pd.isna(avg_kwh_t) else "-")

    trend_df = df.set_index("timestamp_dt")[["kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"]]
    min_time = df["timestamp_dt"].min()
    last_time = df["timestamp_dt"].max()
    real_span = last_time - min_time
    if real_span.total_seconds() <= 0:
        real_span = timedelta(minutes=60)
    future_span = real_span * (0.4 / 0.6)
    future_end = last_time + future_span

    def _safe_base(val_avg, val_last, default):
        if val_avg is not None and not pd.isna(val_avg):
            return val_avg
        if val_last is not None and not pd.isna(val_last):
            return val_last
        return default

    base_tap_temp = _safe_base(avg_tap_temp, last.get("tap_temp_c", None), 1600.0)
    base_kwh_t = _safe_base(avg_kwh_t, last.get("kwh_per_t", None), 420.0)
    base_electrode = _safe_base(avg_electrode, last.get("electrode_kg_per_heat", None), 2.0)

    predicted_tap_temp_target = base_tap_temp + 5.0
    predicted_kwh_t_target = base_kwh_t - 5.0
    predicted_electrode_target = base_electrode

    future_points = []
    last_kwh = last.get("kwh_per_t", base_kwh_t)
    last_tap_temp = last.get("tap_temp_c", base_tap_temp)
    last_electrode = last.get("electrode_kg_per_heat", base_electrode)

    for i in range(4):
        frac = i / 3.0
        t = last_time + future_span * frac
        kwh_val = last_kwh + (predicted_kwh_t_target - last_kwh) * frac
        tap_val = last_tap_temp + (predicted_tap_temp_target - last_tap_temp) * frac
        el_val = last_electrode + (predicted_electrode_target - last_electrode) * frac
        future_points.append(
            {
                "timestamp_dt": t,
                "kwh_per_t": kwh_val,
                "tap_temp_c": tap_val,
                "electrode_kg_per_heat": el_val,
            }
        )

    future_df = pd.DataFrame(future_points)
    predicted_tap_time = future_points[-1]["timestamp_dt"]

    actual_long = (
        trend_df.reset_index()
        .melt(
            id_vars=["timestamp_dt"],
            value_vars=["kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"],
            var_name="variable",
            value_name="value",
        )
    )
    actual_long["data_type"] = "Gerçek"

    future_long = (
        future_df
        .melt(
            id_vars=["timestamp_dt"],
            value_vars=["kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"],
            var_name="variable",
            value_name="value",
        )
    )
    future_long["data_type"] = "Tahmin"

    combined = pd.concat([actual_long, future_long], ignore_index=True)
    variable_name_map = {
        "kwh_per_t": "kWh/t",
        "tap_temp_c": "Tap T (°C)",
        "electrode_kg_per_heat": "Elektrot (kg/şarj)",
    }
    combined["variable_name"] = combined["variable"].map(variable_name_map)

    st.markdown("### Proses Gidişatı – Zaman Trendi ve Tahmini Döküm Anı (AI)")

    base_chart = (
        alt.Chart(combined)
        .mark_line()
        .encode(
            x=alt.X(
                "timestamp_dt:T",
                title="Zaman",
                scale=alt.Scale(domain=[min_time, future_end]),
            ),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("variable_name:N", title="Değişken"),
            strokeDash=alt.StrokeDash(
                "data_type:N",
                title="Veri Tipi",
                scale=alt.Scale(
                    domain=["Gerçek", "Tahmin"],
                    range=[[1, 0], [6, 4]],
                ),
            ),
            tooltip=[
                alt.Tooltip("timestamp_dt:T", title="Zaman"),
                alt.Tooltip("variable_name:N", title="Değişken"),
                alt.Tooltip("value:Q", title="Değer", format=".2f"),
                alt.Tooltip("data_type:N", title="Tip"),
            ],
        )
        .properties(height=320)
    )

    tap_point_df = future_long[
        (future_long["variable"] == "tap_temp_c")
        & (future_long["timestamp_dt"] == predicted_tap_time)
    ].copy()
    tap_point_df["variable_name"] = "Tap T (°C)"

    point_chart = (
        alt.Chart(tap_point_df)
        .mark_point(size=120, filled=True)
        .encode(
            x="timestamp_dt:T",
            y="value:Q",
            color=alt.Color("variable_name:N", legend=None),
            tooltip=[
                alt.Tooltip("timestamp_dt:T", title="Tahmini Döküm Zamanı"),
                alt.Tooltip("value:Q", title="Tahmini Tap T (°C)", format=".1f"),
            ],
        )
    )

    label_df = tap_point_df.copy()
    label_df["label_top"] = label_df.apply(
        lambda r: f"Hedef Döküm Zamanı:\n{r['timestamp_dt'].strftime('%Y-%m-%d %H:%M')}",
        axis=1,
    )
    label_df["label_bottom"] = label_df.apply(
        lambda r: f"Sıcaklık: {r['value']:.0f} °C",
        axis=1,
    )

    label_top_chart = (
        alt.Chart(label_df)
        .mark_text(
            align="left",
            dx=10,
            dy=-25,
            fontSize=12,
            fontWeight="bold",
        )
        .encode(
            x="timestamp_dt:T",
            y="value:Q",
            text="label_top:N",
        )
    )

    label_bottom_chart = (
        alt.Chart(label_df)
        .mark_text(
            align="left",
            dx=10,
            dy=0,
            fontSize=11,
        )
        .encode(
            x="timestamp_dt:T",
            y="value:Q",
            text="label_bottom:N",
        )
    )

    now_df = pd.DataFrame({"timestamp_dt": [last_time]})
    now_rule = (
        alt.Chart(now_df)
        .mark_rule(strokeDash=[2, 2])
        .encode(
            x="timestamp_dt:T",
            tooltip=[alt.Tooltip("timestamp_dt:T", title="Şimdiki An / Son Ölçüm")],
        )
    )

    full_chart = (
        base_chart
        + point_chart
        + now_rule
        + label_top_chart
        + label_bottom_chart
    ).properties(padding={"right": 80})

    st.altair_chart(full_chart.interactive(), use_container_width=True)

    delta_min = (predicted_tap_time - last_time).total_seconds() / 60.0
    st.markdown(
        f"**Tahmini Döküm Anı (AI):** "
        f"{predicted_tap_time.strftime('%Y-%m-%d %H:%M')} "
        f"(yaklaşık {delta_min:.0f} dk sonra)"
    )

    # --------- PROSES KAZANÇ TABLOSU (€/t) ----------
    st.markdown("### 💰 Proses Kazanç Analizi (Ton Başına)")

    ENERGY_PRICE_EUR_PER_KWH = 0.12
    ELECTRODE_PRICE_EUR_PER_KG = 3.0
    TYPICAL_HEAT_TON = float(last.get("tap_weight_t", 35.0) or 35.0)

    rows = []
    total_gain_per_t = 0.0

    # Enerji tüketimi
    if pd.notna(last.get("kwh_per_t", None)) and avg_kwh_t and not pd.isna(avg_kwh_t):
        real_kwh_t = float(last["kwh_per_t"])
        target_kwh_t = max(avg_kwh_t - 5.0, 0.0)
        diff_kwh_t = real_kwh_t - target_kwh_t
        gain_kwh_per_t = max(0.0, diff_kwh_t) * ENERGY_PRICE_EUR_PER_KWH
        total_gain_per_t += gain_kwh_per_t
        rows.append({
            "Tag": "kwh_per_t",
            "Değişken": "Enerji tüket
