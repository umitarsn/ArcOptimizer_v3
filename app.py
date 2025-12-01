import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import altair as alt
import streamlit as st

# ----------------------------------------------
# GENEL AYARLAR
# ----------------------------------------------
st.set_page_config(
    page_title="FeCr AI",
    page_icon="apple-touch-icon.png",
    layout="wide",
)

TZ = ZoneInfo("Europe/Istanbul")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------
# SABİT FİYATLAR (Ton Başına Kazanç Hesabı İçin)
# -------------------------------------------------
ELECTRICITY_PRICE_EUR_PER_MWH = 50.0   # 50 €/MWh  => 0.05 €/kWh
ELECTRODE_PRICE_EUR_PER_KG = 3.0       # örnek: 3 €/kg


# ----------------------------------------------
# YARDIMCI FONKSİYONLAR
# ----------------------------------------------
@st.cache_data
def load_demo_data(n_rows: int = 60) -> pd.DataFrame:
    now = datetime.now(TZ)
    rows = []
    tap_weight_t = 30.0

    kwh_base = 3800
    elec_base = 55.0  # kg/heat

    for i in range(n_rows):
        ts = now - timedelta(hours=(n_rows - i))
        kwh = kwh_base + (i % 5 - 2) * 25
        electrode = elec_base + (i % 7 - 3) * 1.2

        rows.append(
            {
                "timestamp": ts,
                "heat_no": i + 1,
                "tap_weight_t": tap_weight_t,
                "kwh_per_t": kwh / tap_weight_t,
                "electrode_kg_per_heat": electrode,
            }
        )

    return pd.DataFrame(rows)


@st.cache_data
def load_uploaded_data(file) -> pd.DataFrame:
    filename = file.name.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(file)
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(file)
    else:
        raise ValueError("Sadece CSV veya Excel yükleyin.")

    df.columns = [c.strip() for c in df.columns]

    required = {"tap_weight_t", "kwh_per_t", "electrode_kg_per_heat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Eksik kolon(lar): " + ", ".join(missing))
    return df


def build_profit_rows(last: dict, avg_kwh_t: float, avg_electrode: float):
    profit_rows = []

    # --- Enerji tüketimi ---
    if pd.notna(last.get("kwh_per_t")) and not pd.isna(avg_kwh_t):
        real_kwh_t = float(last["kwh_per_t"])
        target_kwh_t = float(avg_kwh_t)
        diff = real_kwh_t - target_kwh_t

        if diff > 0:
            gain = diff * (ELECTRICITY_PRICE_EUR_PER_MWH / 1000.0)
        else:
            gain = 0.0

        profit_rows.append(
            {
                "degisken": "Enerji tüketimi",
                "aktuel": f"{real_kwh_t:.1f} kWh/t",
                "pot": f"{target_kwh_t:.1f} kWh/t",
                "fark": f"{diff:+.1f} kWh/t",
                "kazanc": gain,
                "kazanc_str": f"{gain:.2f} €/t" if gain > 0 else "✔ kalite ↑",
            }
        )

    # --- Elektrot tüketimi ---
    tap_w = float(last.get("tap_weight_t", 0.0) or 0.0)
    if tap_w > 0 and pd.notna(last.get("electrode_kg_per_heat")):
        real_pt = float(last["electrode_kg_per_heat"]) / tap_w

        if pd.notna(avg_electrode):
            avg_pt = float(avg_electrode) / tap_w
            target_pt = min(real_pt, avg_pt)
        else:
            target_pt = max(real_pt - 0.003, 0.0)

        if real_pt > target_pt:
            diff = real_pt - target_pt
            gain = diff * ELECTRODE_PRICE_EUR_PER_KG
        else:
            diff = 0.0
            gain = 0.0
            target_pt = real_pt

        profit_rows.append(
            {
                "degisken": "Elektrot tüketimi",
                "aktuel": f"{real_pt:.3f} kg/t",
                "pot": f"{target_pt:.3f} kg/t",
                "fark": f"{diff:+.3f} kg/t",
                "kazanc": gain,
                "kazanc_str": f"{gain:.2f} €/t" if gain > 0 else "✔ kalite ↑",
            }
        )

    return profit_rows


def render_profit_table(profit_rows):
    if not profit_rows:
        st.info("Kazanç analizi için yeterli veri yok.")
        return

    dfp = pd.DataFrame(profit_rows)
    total = dfp["kazanc"].sum()

    st.subheader("💶 Ton Başına Proses Kazanç Analizi")
    st.dataframe(
        dfp[["degisken", "aktuel", "pot", "fark", "kazanc_str"]],
        use_container_width=True,
        hide_index=True,
    )

    st.metric("Toplam teorik kazanç (€/t)", f"{total:.2f}")

    chart_df = dfp[dfp["kazanc"] > 0]
    if not chart_df.empty:
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x="degisken:N",
                y="kazanc:Q",
                tooltip=["degisken", "kazanc"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)


# ----------------------------------------------
# ANA UYGULAMA
# ----------------------------------------------
def main():
    st.title("🧪 FeCr AI – Proses Kazanç Analizi")

    st.sidebar.header("Veri Kaynağı")
    mode = st.sidebar.radio("Veri seç:", ["Simülasyon", "Yükle (CSV/Excel)"])

    if mode == "Simülasyon":
        df = load_demo_data()
        st.caption("Simülasyon verisi kullanılıyor.")
    else:
        file = st.sidebar.file_uploader("Dosya yükle", type=["csv", "xlsx", "xls"])
        if file is None:
            st.stop()
        df = load_uploaded_data(file)

    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range(
            end=datetime.now(TZ),
            periods=len(df),
            freq="H",
        )

    df = df.sort_values("timestamp").reset_index(drop=True)

    st.subheader("Son Proses Verileri")
    st.dataframe(df.tail(20), use_container_width=True)

    win = st.sidebar.slider("Ortalama için son N", 5, 30, 10)
    tail = df.tail(win)

    avg_kwh = tail["kwh_per_t"].mean()
    avg_elec = tail["electrode_kg_per_heat"].mean()

    last = df.iloc[-1].to_dict()

    profit = build_profit_rows(last, avg_kwh, avg_elec)
    render_profit_table(profit)

    st.subheader("⏱️ Zaman Trendi")
    cols = st.multiselect(
        "Göster:",
        ["kwh_per_t", "electrode_kg_per_heat"],
        ["kwh_per_t", "electrode_kg_per_heat"],
    )

    if cols:
        d2 = df[["timestamp"] + cols].melt("timestamp", var_name="Degisken", value_name="Deger")
        chart = (
            alt.Chart(d2)
            .mark_line()
            .encode(
                x="timestamp:T",
                y="Deger:Q",
                color="Degisken:N",
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()
