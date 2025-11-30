import os
import json
from datetime import datetime
from typing import Optional

import pandas as pd
import pytz
import streamlit as st

# ============================================================
# GENEL UYGULAMA AYARLARI
# ============================================================
st.set_page_config(
    page_title="BG-AI Arc Optimizer Pano",
    layout="wide",
)

# TR saat dilimi – tüm zaman işlemleri buna göre
TR_TZ = pytz.timezone("Europe/Istanbul")


def now_tr() -> datetime:
    """Türkiye saatine göre şimdiki zaman."""
    return datetime.now(TR_TZ)


# Veri klasörü
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Statik inputlar JSON
STATIC_INPUTS_PATH = os.path.join(DATA_DIR, "saved_inputs.json")

# Canlı veri (örnek isimler – kendi dosyana göre değiştir)
RUNTIME_EXCEL_PATH = os.path.join(DATA_DIR, "runtime_data.xlsx")
RUNTIME_SHEET_NAME = "CanliVeri"  # Excel içinde kullanacağın sayfa adı

# Tahmin / trend verisi (örnek CSV)
PREDICTIONS_CSV_PATH = os.path.join(DATA_DIR, "predictions_log.csv")


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def load_static_inputs() -> dict:
    """Müşteri / tesis tarafından girilen statik verileri JSON'dan oku."""
    if os.path.exists(STATIC_INPUTS_PATH):
        try:
            with open(STATIC_INPUTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception:
            return {}
    return {}


def save_static_inputs(data: dict):
    """Statik verileri JSON'a kaydet."""
    with open(STATIC_INPUTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_runtime_data() -> Optional[pd.DataFrame]:
    """Canlı veri Excel'ini oku. Birimlerin kaybolmaması için dtype=str."""
    if not os.path.exists(RUNTIME_EXCEL_PATH):
        return None

    try:
        df = pd.read_excel(
            RUNTIME_EXCEL_PATH,
            sheet_name=RUNTIME_SHEET_NAME,
            dtype=str  # birimlerin (ör: kWh/t, °C, min) bozulmaması için
        )
        return df
    except Exception as e:
        st.error(f"Canlı veri okunurken hata oluştu: {e}")
        return None


def load_predictions_data() -> Optional[pd.DataFrame]:
    """
    Tahmin / trend verisini CSV'den oku.
    Beklenen kolonlar (örnek):
      - timestamp: tarih/saat string
      - heat_id: döküm numarası
      - feature_1, feature_2, ... (opsiyonel)
      - predicted_tap_time: dakika veya datetime
    """
    if not os.path.exists(PREDICTIONS_CSV_PATH):
        return None

    try:
        df = pd.read_csv(PREDICTIONS_CSV_PATH)

        # timestamp'i TR saatine çevir
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            # Eğer timezone'suz geldiyse TR ile localize et
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(TR_TZ)
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert(TR_TZ)

        # predicted_tap_time zamanı ayrıysa onu da normalize edebilirsin
        if "predicted_tap_time" in df.columns:
            # Eğer bu bir tarih/saat ise:
            # df["predicted_tap_time"] = pd.to_datetime(df["predicted_tap_time"], errors="coerce")
            # if df["predicted_tap_time"].dt.tz is None:
            #     df["predicted_tap_time"] = df["predicted_tap_time"].dt.tz_localize(TR_TZ)
            # else:
            #     df["predicted_tap_time"] = df["predicted_tap_time"].dt.tz_convert(TR_TZ)
            pass

        return df
    except Exception as e:
        st.error(f"Tahmin/trend verisi okunurken hata oluştu: {e}")
        return None


# ============================================================
# 1. SAYFA – VERİ GİRİŞİ (MÜŞTERİ / TESİS INPUTLARI)
# ============================================================

def page_veri_girisi():
    st.title("1. Veri Girişi – Tesis / Proses Parametreleri")

    st.markdown(
        "Bu sayfa, müşteri tarafından doldurulan **statik parametreleri** toplar "
        "ve JSON olarak kaydeder. Diğer sayfalar bu verileri sadece okur."
    )

    saved_inputs = load_static_inputs()

    # Eski verileri form varsayılanı olarak kullan
    col1, col2, col3 = st.columns(3)

    with col1:
        plant_name = st.text_input(
            "Tesis Adı",
            value=saved_inputs.get("plant_name", "")
        )
        furnace_type = st.selectbox(
            "Ocak Tipi",
            ["EAF", "BOF", "IF", "Diğer"],
            index=["EAF", "BOF", "IF", "Diğer"].index(
                saved_inputs.get("furnace_type", "EAF")
            )
            if saved_inputs.get("furnace_type") in ["EAF", "BOF", "IF", "Diğer"]
            else 0,
        )
        tap_weight = st.number_input(
            "Nominal Tap Ağırlığı [t]",
            min_value=0.0,
            value=float(saved_inputs.get("tap_weight", 0.0)),
            step=1.0,
        )

    with col2:
        transformer_power = st.number_input(
            "Trafo Gücü [MVA]",
            min_value=0.0,
            value=float(saved_inputs.get("transformer_power", 0.0)),
            step=1.0,
        )
        hot_heel = st.number_input(
            "Hot Heel [t]",
            min_value=0.0,
            value=float(saved_inputs.get("hot_heel", 0.0)),
            step=1.0,
        )
        scrap_ratio = st.number_input(
            "Hurda Oranı [%]",
            min_value=0.0,
            max_value=100.0,
            value=float(saved_inputs.get("scrap_ratio", 0.0)),
            step=1.0,
        )

    with col3:
        operator_name = st.text_input(
            "Sorumlu Operatör / Mühendis",
            value=saved_inputs.get("operator_name", "")
        )
        project_code = st.text_input(
            "Proje Kodu",
            value=saved_inputs.get("project_code", "")
        )
        notes = st.text_area(
            "Notlar",
            value=saved_inputs.get("notes", "")
        )

    # Kaydet butonu
    if st.button("Kaydet"):
        new_data = {
            "plant_name": plant_name,
            "furnace_type": furnace_type,
            "tap_weight": tap_weight,
            "transformer_power": transformer_power,
            "hot_heel": hot_heel,
            "scrap_ratio": scrap_ratio,
            "operator_name": operator_name,
            "project_code": project_code,
            "notes": notes,
            "last_update": now_tr().isoformat(),
        }
        save_static_inputs(new_data)
        st.success("Veriler kaydedildi.")

    # Sağda özet kutusu
    with st.expander("Kayıtlı Son Veri Özeti", expanded=True):
        saved_inputs = load_static_inputs()
        if saved_inputs:
            st.json(saved_inputs, expanded=False)
        else:
            st.info("Henüz kayıtlı veri yok.")


# ============================================================
# 2. SAYFA – CANLI VERİ (RUNTIME DATA PANO)
# ============================================================

def page_canli_veri():
    st.title("2. Canlı Veri – Pano")

    st.markdown(
        "Bu sayfa, otomasyon sisteminden veya ara dosyadan gelen **anlık / geçmiş "
        "proses verilerini** gösterir."
    )

    st.caption(
        f"TR Saati (uygulama): **{now_tr().strftime('%Y-%m-%d %H:%M:%S')}**"
    )

    df = load_runtime_data()

    if df is None:
        st.warning(
            "Herhangi bir canlı veri dosyası bulunamadı.\n\n"
            f"Beklenen dosya: `{RUNTIME_EXCEL_PATH}`, sayfa: `{RUNTIME_SHEET_NAME}`"
        )
        return

    # Birimler neden kaybolmasın?
    # - Excel'de birim satırları / kolonları mutlaka ayrı ve net başlıklarla olmalı.
    # - Burada dtype=str ile okuduğumuz için 'kWh/t', '°C' vs. aynen korunur.

    st.subheader("Ham Tablo Görünümü")
    st.dataframe(df, use_container_width=True)

    # Eğer belirli kolonlarda (ör: 4, 5, 6) birim görünmüyorsa,
    # çoğunlukla başlıkların veya satırların yanlış alınmasından kaynaklanır.
    # Burada df.columns ve ilk satıra bakarak hızlı kontrol yapalım:
    with st.expander("Kolon Başlıkları ve İlk Satır Kontrolü"):
        st.write("Kolonlar:", list(df.columns))
        if len(df) > 0:
            st.write("İlk satır:", df.iloc[0].to_dict())

    # Basit filtre örneği – varsa HEAT_ID veya benzeri kolonla süzme
    filtre_kolon = None
    aday_kolonlar = [c for c in df.columns if c.lower() in ["heat_id", "heat", "cast_no"]]
    if aday_kolonlar:
        filtre_kolon = aday_kolonlar[0]

    if filtre_kolon:
        heat_list = sorted(df[filtre_kolon].dropna().unique())
        secili_heat = st.selectbox(
            f"{filtre_kolon} filtresi",
            options=["Tümü"] + list(heat_list),
            index=0,
        )
        if secili_heat != "Tümü":
            df_filtered = df[df[filtre_kolon] == secili_heat]
        else:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()

    st.subheader("Filtrelenmiş Veri")
    st.dataframe(df_filtered, use_container_width=True)


# ============================================================
# 3. SAYFA – ARC OPTIMIZER PANO (TREND + TAHMİN)
# ============================================================

def page_arc_optimizer_pano():
    st.title("3. Arc Optimizer – Zaman Trend Pano")

    st.markdown(
        "Bu sayfa, geçmiş dökümler ve **yapay zekâ tahminlerini** zaman ekseninde gösterir. "
        "Eski tahminlerin görünmemesi çoğunlukla zaman damgası ve timezone farkından kaynaklanır; "
        "bu sürümde tüm veriler **Europe/Istanbul** saatine normalize edilir."
    )

    # Simülasyon modu – demo veri gibi davranmak için
    simulation_mode = st.toggle("Simülasyon Modu", value=False)

    df_pred = load_predictions_data()

    if df_pred is None:
        st.warning(
            "Henüz tahmin / trend verisi bulunamadı.\n\n"
            f"Beklenen CSV: `{PREDICTIONS_CSV_PATH}`"
        )
        return

    # Eğer simülasyon modu açıksa, en son N satırı alıp
    # sanki 'şu an' oluyormuş gibi kaydırabilirsin.
    if simulation_mode and "timestamp" in df_pred.columns:
        df_pred = df_pred.sort_values("timestamp").copy()
        # Örnek: son 20 kaydı al
        df_pred = df_pred.tail(20)

    st.caption(
        f"TR Saati (uygulama): **{now_tr().strftime('%Y-%m-%d %H:%M:%S')}**"
    )

    # Zaman filtreleri
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Zaman Aralığı Filtresi")
        min_time = df_pred["timestamp"].min()
        max_time = df_pred["timestamp"].max()

        if pd.isna(min_time) or pd.isna(max_time):
            st.error("timestamp kolonu doğru okunamadı. CSV formatını kontrol et.")
            return

        # Slider için timezone'suz (naive) kopya
        min_time_naive = min_time.replace(tzinfo=None)
        max_time_naive = max_time.replace(tzinfo=None)

        time_range = st.slider(
            "Gösterilecek aralık",
            min_value=min_time_naive,
            max_value=max_time_naive,
            value=(min_time_naive, max_time_naive),
            format="YYYY-MM-DD HH:mm",
        )

    with col2:
        st.subheader("Diğer Seçenekler")
        show_points = st.checkbox("Veri noktalarını göster", value=True)
        show_lines = st.checkbox("Trend çizgisini göster", value=True)

    # Slider aralığını TR timezone’a geri sar
    start_tr = TR_TZ.localize(time_range[0])
    end_tr = TR_TZ.localize(time_range[1])

    mask = (df_pred["timestamp"] >= start_tr) & (df_pred["timestamp"] <= end_tr)
    df_filtered = df_pred.loc[mask].copy()

    if df_filtered.empty:
        st.info("Seçilen zaman aralığında veri bulunamadı. Zaman aralığını genişletmeyi deneyin.")
        return

    # HEAT_ID varsa seçilebilir
    heat_col_candidates = [c for c in df_filtered.columns if c.lower() in ["heat_id", "heat", "cast_no"]]
    selected_heat = None
    if heat_col_candidates:
        heat_col = heat_col_candidates[0]
        heat_values = ["Tümü"] + sorted(df_filtered[heat_col].dropna().unique().astype(str).tolist())
        selected_heat = st.selectbox(f"{heat_col} filtresi", heat_values)
        if selected_heat != "Tümü":
            df_filtered = df_filtered[df_filtered[heat_col].astype(str) == selected_heat]

    # Grafik için kullanılacak kolonlar
    # Örnek: total_energy, active_power, tap_time_min vs.
    numeric_cols = df_filtered.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        st.error("Grafik için sayısal kolon bulunamadı. CSV içeriğini kontrol et.")
        return

    y_col = st.selectbox(
        "Trend için değişken seç",
        options=numeric_cols,
        index=0
    )

    st.subheader("Zaman Serisi Grafiği")

    # Plotly veya Altair kullanılabilir; Streamlit native line_chart da olur.
    # Burada basit line_chart:
    chart_df = df_filtered.set_index("timestamp")[y_col]

    # Timestamp'i timezone'suz göstermek istersen:
    chart_df.index = chart_df.index.tz_convert(TR_TZ).tz_localize(None)

    st.line_chart(chart_df)

    # Tahmin edilen tap time noktası ayrıca gösterilebilir
    if "predicted_tap_time" in df_filtered.columns:
        st.subheader("Tahmin Edilen Döküm Anları")
        st.dataframe(
            df_filtered[["timestamp", "predicted_tap_time"] + ([heat_col] if heat_col_candidates else [])],
            use_container_width=True
        )

    with st.expander("Ham Veri (Filtrelenmiş)"):
        st.dataframe(df_filtered, use_container_width=True)


# ============================================================
# ANA ÇALIŞTIRMA BLOĞU
# ============================================================

def main():
    st.sidebar.title("BG-AI Arc Optimizer – Pano")
    page = st.sidebar.radio(
        "Sayfa Seçin",
        (
            "1. Veri Girişi",
            "2. Canlı Veri",
            "3. Arc Optimizer Pano",
        ),
    )

    if page == "1. Veri Girişi":
        page_veri_girisi()
    elif page == "2. Canlı Veri":
        page_canli_veri()
    elif page == "3. Arc Optimizer Pano":
        page_arc_optimizer_pano()


if __name__ == "__main__":
    main()
