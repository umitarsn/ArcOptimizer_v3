import os
from datetime import datetime

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# SAYFA AYARLARI
# ------------------------------------------------------------
st.set_page_config(
    page_title="Enerji Verimliliği",
    layout="wide",
    page_icon=None,
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# LOGO / MARKALAMA (Sol üst taraf)
# ------------------------------------------------------------
LOGO_FILE = "logo.png"

if os.path.exists(LOGO_FILE):
    # Sidebar navigasyonunun üstüne logo yerleştiren CSS
    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"]::before {{
                content: "";
                display: block;
                background-image: url("{LOGO_FILE}");
                background-size: contain;
                background-repeat: no-repeat;
                background-position: center;
                height: 120px;
                margin: 16px 16px 24px 16px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar içinde de logo görüntüle (fallback)
    with st.sidebar:
        st.image(LOGO_FILE, width=160)
else:
    # Logo yoksa sadece başlık göster
    st.sidebar.header("Enerji Verimliliği")


# ------------------------------------------------------------
# YARDIMCI FONKSİYON: Kolon ismi eşleştirici
# ------------------------------------------------------------
def pick_column(df: pd.DataFrame, candidates):
    """
    Verilen aday isimler arasından DataFrame'de olan ilk kolon adını döner.
    Hiçbiri yoksa None döner.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ------------------------------------------------------------
# TÜM SHEET'LERİ YÜKLEYEN FONKSİYON
# ------------------------------------------------------------
@st.cache_data
def load_all_question_sheets():
    """
    dc_saf_soru_tablosu.xlsx içindeki TÜM sheet'leri yükler.
    Her sheet bir DataFrame olarak dict içinde döner:
    { "SheetAdı": df, ... }

    Excel yapısı:
    - 1. satır: mavi başlık (ör. 'Hammadde ve Şarj girdileri')  -> atlanır
    - 2. satır: kolon başlıkları                               -> header
    - 3. satırdan itibaren: parametre satırları                -> formda kullanılacak
    """
    file_name = "dc_saf_soru_tablosu.xlsx"
    try:
        # header=1 -> 2. satır kolon başlıkları; 1. satır tamamen skip
        sheets_dict = pd.read_excel(file_name, sheet_name=None, header=1)
    except FileNotFoundError:
        st.error(f"❌ '{file_name}' dosyası bulunamadı. Lütfen proje klasörüne ekleyin.")
        return None
    except Exception as e:
        st.error(f"❌ Excel dosyası okunurken hata oluştu: {e}")
        return None

    clean_sheets = {}
    for name, df in sheets_dict.items():
        if df is not None:
            # Boş satırları (tamamen NaN) drop et
            df = df.dropna(how="all")
            if not df.empty:
                clean_sheets[name] = df
    return clean_sheets


# ------------------------------------------------------------
# FORM SAYFASI
# ------------------------------------------------------------
def show_energy_efficiency_form():
    st.title("⚡ Enerji Verimliliği")
    st.markdown(
        """
        Bu form, **DC SAF enerji verimliliği ve proses optimizasyonu** için
        ihtiyaç duyulan verileri sistematik şekilde toplamak amacıyla hazırlanmıştır.  

        Lütfen tesisinizi en doğru şekilde yansıtacak biçimde doldurunuz.
        """
    )

    sheets = load_all_question_sheets()
    if sheets is None or len(sheets) == 0:
        st.warning("Excel içinden okunacak sheet bulunamadı.")
        return

    # Sidebar’da kısa açıklama + sayaç
    total_params = 0
    for _, df_q in sheets.items():
        label_col_tmp = pick_column(
            df_q,
            ["Değer", "Deger", "Değişken", "Degisken", "Parametre", "Parameter"],
        )
        if label_col_tmp:
            total_params += df_q[label_col_tmp].dropna().shape[0]

    with st.sidebar:
        st.subheader("Form Bilgisi")
        st.info(
            f"Toplam ~{total_params} parametre, 6 başlık (sheet) altında "
            "Değer + Veri Kaynağı + Kayıt Aralığı ile doldurulabilir."
        )

    # --------------------------------------------------------
    # FORM
    # --------------------------------------------------------
    with st.form("energy_efficiency_form"):
        st.subheader("Müşteri Girdileri")
        st.write(
            "Aşağıdaki 6 ana başlık altındaki tüm parametreler için tesisinize ait "
            "**Değer**, **Veri Kaynağı** ve **Kayıt Aralığı** bilgilerini giriniz."
        )

        responses = {}

        # Her sheet (sekme) için ayrı bölüm
        for sheet_name, df_q in sheets.items():
            if df_q is None or df_q.empty:
                continue

            st.markdown(f"## {sheet_name}")

            # Kolon isimlerini bu sheet için tespit et
            label_col = pick_column(
                df_q,
                ["Değer", "Deger", "Değişken", "Degisken", "Parametre", "Parameter"],
            )
            unit_col = pick_column(
                df_q,
                [
                    "Birim",
                    "Birim/Tip",
                    "Birim /Tip",
                    "Birim / Tip",
                    "Birim/Tİp",
                    "Birim /Tip ",
                    "Birim / Tip ",
                ],
            )
            veri_kaynagi_col = pick_column(df_q, ["Veri Kaynağı", "Veri Kaynagi"])
            kayit_araligi_col = pick_column(
                df_q,
                ["Kayıt Aralığı", "Kayıt Araligi", "Kayit Araligi", "Kayit Aralığı"],
            )

            # Açıklama / Enerji etkisi gibi kolonları help text olarak birleştir
            desc_cols = []
            for c in df_q.columns:
                cname = str(c)
                if (
                    "Açıklama" in cname
                    or "Aciklama" in cname
                    or "Enerjiye Etkisi" in cname
                    or "Enerji Etkisi" in cname
                ):
                    desc_cols.append(c)

            if label_col is None:
                st.warning(
                    f"⚠ Sheet '{sheet_name}' içinde 'Değişken / Değer / Parametre' "
                    f"kolonu bulunamadı. Bu sheet atlanıyor."
                )
                continue

            # Her satır için soru alanı
            for idx, row in df_q.iterrows():
                label = row.get(label_col)

                # Boş satır ise geç
                if pd.isna(label) or str(label).strip() == "":
                    continue

                unit = row.get(unit_col) if unit_col else None

                # Açıklama / Enerji etkisi birleşik help text ve görünür açıklama
                desc_parts = []
                for col in desc_cols:
                    val = row.get(col)
                    if isinstance(val, str) and val.strip() != "":
                        desc_parts.append(val)
                desc_text = "\n\n".join(desc_parts) if desc_parts else None

                # Her sheet için key benzersiz olsun
                safe_sheet = str(sheet_name).replace(" ", "_")
                base_key = f"{safe_sheet}_q_{idx}"

                # ------------------------------
                # 1) DEĞER ALANI
                # ------------------------------
                full_label = str(label)
                if isinstance(unit, str) and unit.strip() not in ["", "–", "-"]:
                    full_label = f"{label} ({unit})"

                unit_str = str(unit) if not pd.isna(unit) else ""
                numeric_unit_tokens = [
                    "%",
                    "°C",
                    "kWh",
                    "kW",
                    "Nm³",
                    "Nm3",
                    "t",
                    "kg",
                    "MJ",
                    "m³",
                    "m3",
                    "Pa",
                    "bar",
                    "dk",
                ]

                st.markdown(f"**{full_label}**")
                if desc_text:
                    st.caption(desc_text)

                if any(sym in unit_str for sym in numeric_unit_tokens):
                    responses[base_key + "_value"] = st.number_input(
                        "Değer",
                        value=0.0,
                        step=0.1,
                        help=desc_text,
                        key=base_key + "_value",
                    )
                else:
                    responses[base_key + "_value"] = st.text_input(
                        "Değer",
                        help=desc_text,
                        key=base_key + "_value",
                    )

                # ------------------------------
                # 2) VERİ KAYNAĞI ALANI
                # ------------------------------
                default_source = ""
                if veri_kaynagi_col and veri_kaynagi_col in df_q.columns:
                    val = row.get(veri_kaynagi_col)
                    if isinstance(val, str):
                        default_source = val
                responses[base_key + "_source"] = st.text_input(
                    "Veri Kaynağı",
                    value=default_source,
                    key=base_key + "_source",
                )

                # ------------------------------
                # 3) KAYIT ARALIĞI ALANI
                # ------------------------------
                default_interval = ""
                if kayit_araligi_col and kayit_araligi_col in df_q.columns:
                    val = row.get(kayit_araligi_col)
                    if isinstance(val, str):
                        default_interval = val
                responses[base_key + "_interval"] = st.text_input(
                    "Kayıt Aralığı",
                    value=default_interval,
                    key=base_key + "_interval",
                )

                st.markdown("---")

        submitted = st.form_submit_button("Kaydet")

    # --------------------------------------------------------
    # FORM SUBMIT EDİLDİĞİNDE
    # --------------------------------------------------------
    if submitted:
        records = []
        timestamp = datetime.now().isoformat(timespec="seconds")

        for sheet_name, df_q in sheets.items():
            if df_q is None or df_q.empty:
                continue

            label_col = pick_column(
                df_q,
                ["Değer", "Deger", "Değişken", "Degisken", "Parametre", "Parameter"],
            )
            unit_col = pick_column(
                df_q,
                [
                    "Birim",
                    "Birim/Tip",
                    "Birim /Tip",
                    "Birim / Tip",
                    "Birim/Tİp",
                    "Birim /Tip ",
                    "Birim / Tip ",
                ],
            )
            if label_col is None:
                continue

            safe_sheet = str(sheet_name).replace(" ", "_")

            for idx, row in df_q.iterrows():
                label = row.get(label_col)
                if pd.isna(label) or str(label).strip() == "":
                    continue

                unit = row.get(unit_col) if unit_col else None

                base_key = f"{safe_sheet}_q_{idx}"
                value = responses.get(base_key + "_value", "")
                source = responses.get(base_key + "_source", "")
                interval = responses.get(base_key + "_interval", "")

                # Hiçbir alan doldurulmamışsa satırı atla
                if (
                    (isinstance(value, str) and value.strip() == "")
                    and (isinstance(source, str) and source.strip() == "")
                    and (isinstance(interval, str) and interval.strip() == "")
                ):
                    continue

                records.append(
                    {
                        "timestamp": timestamp,
                        "Sheet": sheet_name,
                        "Parametre": label,
                        "Birim": "" if pd.isna(unit) else str(unit),
                        "Deger": value,
                        "Veri_Kaynagi": source,
                        "Kayit_Araligi": interval,
                    }
                )

        if not records:
            st.warning("Herhangi bir cevap bulunamadı, form alanları boş olabilir.")
            return

        resp_df = pd.DataFrame(records)

        # Dosya kaydı
        os.makedirs("data", exist_ok=True)
        resp_file = os.path.join("data", "energy_efficiency_form_responses.csv")

        if os.path.exists(resp_file):
            existing = pd.read_csv(resp_file)
            combined = pd.concat([existing, resp_df], ignore_index=True)
        else:
            combined = resp_df

        combined.to_csv(resp_file, index=False)

        st.success("✅ Form yanıtlarınız kaydedildi.")
        st.subheader("Son Yanıtınız")
        st.dataframe(resp_df, use_container_width=True)

        with st.expander("Tüm Kayıtlı Yanıtlar (Yerel Dosya)"):
            st.dataframe(combined.tail(200), use_container_width=True)


# ------------------------------------------------------------
# ANA FONKSİYON
# ------------------------------------------------------------
def main():
    show_energy_efficiency_form()


if __name__ == "__main__":
    main()
