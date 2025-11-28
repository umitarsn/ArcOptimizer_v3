import os
from datetime import datetime

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# SAYFA AYARLARI
# ------------------------------------------------------------
st.set_page_config(
    page_title="Enerji Verimliligi",
    layout="wide",
    page_icon=None,
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# EXCEL OKUMA
# ------------------------------------------------------------
@st.cache_data
def load_sheets():
    """
    dc_saf_soru_tablosu.xlsx dosyasindaki tum sheet'leri okur.

    Varsayim:
    - 1. satir baslik (A,B,C,D,E,F,G)
    - A,B,C,D: ekranda gorunecek
    - D: musteri tarafindan set edilecek
    - E,F,G: detay bilgi (info icin)
    """
    file_name = "dc_saf_soru_tablosu.xlsx"

    try:
        sheets = pd.read_excel(file_name, sheet_name=None, header=0)
    except FileNotFoundError:
        st.error(
            "HATA: 'dc_saf_soru_tablosu.xlsx' bulunamadi. Dosyayi app.py ile ayni klasore koyun."
        )
        return None
    except Exception as e:
        st.error(f"Excel okunurken hata olustu: {e}")
        return None

    cleaned = {}
    for name, df in sheets.items():
        if df is None:
            continue

        # Tamamen bos satir ve kolonlari temizle
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")

        if not df.empty:
            cleaned[name] = df

    return cleaned


# ------------------------------------------------------------
# FORM SAYFASI
# ------------------------------------------------------------
def show_energy_form():
    st.title("Enerji Verimliligi")

    st.markdown(
        """
        Bu form, **dc_saf_soru_tablosu.xlsx** dosyasina gore hazirlanmistir.

        - Ekranda yalnizca **A, B, C, D** sutunlari gorunur.
        - **D sutunu** musterinin set edecegi alandir ve duzenlenebilir.
        - Her satirda bir **Info (ℹ️)** etkisi vardir; secili satir icin
          **E, F, G** sutunlarindaki detaylar altta gosterilir.
        """
    )

    sheets = load_sheets()
    if sheets is None or len(sheets) == 0:
        return

    total_rows = sum(len(df) for df in sheets.values())
    with st.sidebar:
        st.subheader("Form Bilgisi")
        st.info(f"Toplam satir sayisi: {total_rows}")

    edited_sheets = {}

    # --------------------------------------------------------
    # FORM
    # --------------------------------------------------------
    with st.form("energy_form"):
        st.subheader("Musteri Girdileri")

        for i, (sheet_name, df_full) in enumerate(sheets.items(), start=1):

            with st.expander(f"{i}. {sheet_name}", expanded=(i == 1)):

                # A,B,C,D sutunlarini al (ilk 4 sutun varsayiliyor)
                main_cols = list(df_full.columns[:4])
                # Detay sutunlari (E,F,G ve varsa sonrasi)
                detail_cols = list(df_full.columns[4:])

                # D sutunu musteri girdisi (4. kolon)
                if len(main_cols) < 4:
                    st.warning("Bu sheet icin A,B,C,D sutunlari eksik, atlaniyor.")
                    continue

                col_A = main_cols[0]
                col_B = main_cols[1]
                col_C = main_cols[2]
                col_D = main_cols[3]  # Set kolonu

                # Gorunecek tabloyu hazirla
                view_df = df_full[main_cols].copy()
                # Info kolonunu ekle (sadece ikon)
                view_df["Info"] = "ℹ️"

                st.caption("A, B, C ve D sutunlari. D sutununu set edebilirsiniz.")
                edited_view = st.data_editor(
                    view_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        col_A: st.column_config.TextColumn(disabled=True),
                        col_B: st.column_config.TextColumn(disabled=True),
                        col_C: st.column_config.TextColumn(disabled=True),
                        "Info": st.column_config.TextColumn(disabled=True),
                        # col_D sutunu default olarak duzenlenebilir kaliyor
                    },
                    key=f"sheet_{i}_view",
                )

                # Info: detay gosterme
                if detail_cols:
                    st.markdown("**Detay gormek icin satir secin (ℹ️):**")

                    # Secim icin etiket olusturalim: "Tag - Deger" gibi (A ve B)
                    labels = []
                    indices = []
                    for idx, row in edited_view.iterrows():
                        label = f"{row[col_A]} - {row[col_B]}"
                        labels.append(label)
                        indices.append(idx)

                    if labels:
                        selected_label = st.selectbox(
                            "Satir",
                            options=labels,
                            key=f"sheet_{i}_detail_select",
                        )

                        # Secilen label'in index'ini bul
                        sel_idx = indices[labels.index(selected_label)]
                        detail_row = df_full.loc[sel_idx, detail_cols]

                        # Detaylari alt alta goster
                        detail_lines = []
                        for col in detail_cols:
                            val = detail_row[col]
                            if pd.notna(val) and str(val).strip() != "":
                                detail_lines.append(f"- **{col}**: {val}")

                        if detail_lines:
                            st.info("\n".join(detail_lines))

                # Daha sonra D sutununu geri yazmak icin sakla
                edited_sheets[sheet_name] = (df_full, edited_view, col_D)

        submitted = st.form_submit_button("Kaydet")

    # --------------------------------------------------------
    # KAYDETME
    # --------------------------------------------------------
    if submitted:
        os.makedirs("data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join("data", f"energy_form_{timestamp}.xlsx")

        try:
            with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
                for name, (df_full, edited_view, col_D) in edited_sheets.items():
                    # D sutununu edited_view'den geri yaz
                    df_full[col_D] = edited_view[col_D]
                    df_full.to_excel(writer, sheet_name=name[:31], index=False)
        except Exception as e:
            st.error(f"Kayit yapilirken hata olustu: {e}")
            return

        st.success("Veriler basariyla kaydedildi.")
        st.write(f"Kaydedilen dosya: {out_file}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    show_energy_form()


if __name__ == "__main__":
    main()
