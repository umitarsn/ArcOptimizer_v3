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
    """Gerçek data yoksa kullanılacak örnek veri seti."""
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

    df = pd.DataFrame(rows)
    return df


@st.cache_data
def load_uploaded_data(file) -> pd.DataFrame:
    """Kullanıcı CSV/Excel yüklerse oku."""
    filename = file.name.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(file)
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(file)
    else:
        raise ValueError("Sadece CSV veya Excel dosyası yükleyin.")

    # Beklenen kolon isimleri yoksa kullanıcıya anlamlı bir mesaj verebilmek için normalize et
    df.columns = [c.strip() for c in df.columns]

    # Zorunlu kolonları kontrol et
    required_cols = {"tap_weight_t", "kwh_per_t", "electrode_kg_per_heat"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            "Eksik kolonlar: "
            + ", ".join(sorted(missing))
            + ". Lütfen veri setinizi kontrol edin."
        )
    return df


def build_profit_rows(last: dict, avg_kwh_t: float, avg_electrode: float):
    """Enerji ve elektrot tüketimine göre ton başına kazanç satırlarını üretir."""
    profit_rows = []

    # 1) Enerji tüketimi (kwh_per_t)
    if pd.notna(last.get("kwh_per_t", None)) and not pd.isna(avg_kwh_t):
        real_kwh_t = float(last["kwh_per_t"])
        target_kwh_t = float(avg_kwh_t)  # basit yaklaşım: son N ort.
        diff_kwh_t = real_kwh_t - target_kwh_t  # pozitif = iyileştirme alanı

        if diff_kwh_t > 0:
            gain_eur_t = diff_kwh_t * (ELECTRICITY_PRICE_EUR_PER_MWH / 1000.0)
        else:
            gain_eur_t = 0.0

        profit_rows.append(
            {
                "tag": "kwh_per_t",
                "degisken": "Enerji tüketimi",
                "aktuel": f"{real_kwh_t:.1f} kWh/t",
                "potansiyel": f"{target_kwh_t:.1f} kWh/t",
                "fark": f"{diff_kwh_t:+.1f} kWh/t",
                "kazanc_eur_t": gain_eur_t,
                "kazanc_gosterim": f"{gain_eur_t:.2f} €/t" if gain_eur_t > 0 else "✔ kalite ↑",
                "tur": "cost",
            }
        )

    # 2) Elektrot tüketimi (DAİMA daha iyi veya en kötü eşit)
    tap_w = float(last.get("tap_weight_t", 0.0) or 0.0)
    if tap_w > 0 and pd.notna(last.get("electrode_kg_per_heat", None)):
        real_elec_pt = float(last["electrode_kg_per_heat"]) / tap_w  # kg/t

        # Ortalama varsa: hedef = min(aktüel, ortalama)
        if pd.notna(avg_electrode):
            avg_elec_pt = float(avg_electrode) / tap_w
            target_elec_pt = min(real_elec_pt, avg_elec_pt)
        else:
            # Ortalama yoksa, hafif iyileştirme hedefi ama asla aktüelden kötü değil
            target_elec_pt = max(real_elec_pt - 0.003, 0.0)

        # Eğer zaten hedeften iyi ise kazanç = 0, fark = 0 göster
        if real_elec_pt > target_elec_pt:
            diff_elec_pt = real_elec_pt - target_elec_pt
            gain_elec_eur_t = diff_elec_pt * ELECTRODE_PRICE_EUR_PER_KG
        else:
            diff_elec_pt = 0.0
            gain_elec_eur_t = 0.0
            target_elec_pt = real_elec_pt

        profit_rows.append(
            {
                "tag": "electrode",
                "degisken": "Elektrot tüketimi",
                "aktuel": f"{real_elec_pt:.3f} kg/t",
                "potansiyel": f"{target_elec_pt:.3f} kg/t",
                "fark": f"{diff_elec_pt:+.3f} kg/t",
                "kazanc_eur_t": gain_elec_eur_t,
                "kazanc_gosterim": f"{gain_elec_eur_t:.2f} €/t" if gain_elec_eur_t > 0 else "✔ kalite ↑",
                "tur": "cost",
            }
        )

    return profit_rows


def render_profit_table(profit_rows):
    if not profit_rows:
        st.info("Kazanç analizi için yeterli veri yok.")
        return

    df_profit = pd.DataFrame(profit_rows)

    # Toplam kazanç (yalnızca sayısal olanlar üzerinden)
    total_gain = df_profit["kazanc_eur_t"].sum()

    st.subheader("💶 Ton Başına Proses Kazanç Analizi")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(
            df_profit[["degisken", "aktuel", "potansiyel", "fark", "kazanc_gosterim"]],
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.metric(
            label="Toplam teorik kazanç (€/t)",
            value=f"{total_gain:.2f}",
        )

    # Altair bar chart (sadece pozitif kazançları göster)
    chart_df = df_profit[df_profit["kazanc_eur_t"] > 0].copy()
    if not chart_df.empty:
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("degisken:N", title="Değişken"),
                y=alt.Y("kazanc_eur_t:Q", title="€/t"),
                tooltip=["degisken", "kazanc_eur_t"],
            )
            .properties(height=250)
        )
        st.altair_chart(chart, use_container_width=True)


# ----------------------------------------------
# ANA UYGULAMA
# ----------------------------------------------
def main():
    st.title("🧪 FeCr AI – Proses Kazanç Analizi")

    st.markdown(
        """
        Bu ekran, son ergitme verilerine ve son birkaç ergitmenin ortalamalarına bakarak
        **ton başına potansiyel kazanç** alanlarını hesaplar.
        """
    )

    st.sidebar.header("Veri Kaynağı")
    data_mode = st.sidebar.radio(
        "Veri kaynağını seçin",
        ["Demo veri (Simülasyon)", "Dos]()
