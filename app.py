import os
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import joblib

# ------------------------------------------------------------
# Genel ayarlar
# ------------------------------------------------------------
st.set_page_config(
    page_title="Arc Optimizer – Demo",
    layout="wide",
    page_icon=None,
)

DATA_PATH = "data/BG_EAF_panelcooling_demo.csv"
MODEL_PATH_KWH = "models/model_kwh_per_t.pkl"
MODEL_PATH_TAP = "models/model_tap_temp.pkl"

HEAT_TONNAGE = 10.0  # ton / heat (demo varsayımı)


# ------------------------------------------------------------
# Session state: model eğitim durumu
# ------------------------------------------------------------
if "model_status" not in st.session_state:
    st.session_state["model_status"] = "Henüz eğitilmedi."
    st.session_state["last_train_time"] = None
    st.session_state["last_train_rows"] = 0
    st.session_state["train_count"] = 0
    st.session_state["last_seen_rows"] = 0

# ------------------------------------------------------------
# Yardımcı fonksiyonlar
# ------------------------------------------------------------
@st.cache_data
def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ek kolonlar: kwh_per_t, panel_delta_t, scrap_quality
    """
    df = df.copy()

    # kWh/t
    if "power_kWh" in df.columns:
        df["kwh_per_t"] = df["power_kWh"] / HEAT_TONNAGE

    # Panel delta T
    if {"panel_T_in_C", "panel_T_out_C"}.issubset(df.columns):
        df["panel_delta_t"] = df["panel_T_out_C"] - df["panel_T_in_C"]
    else:
        df["panel_delta_t"] = 0.0

    # Scrap quality (demo amaçlı)
    for col in ["scrap_HMS80_20_pct", "scrap_HBI_pct", "scrap_Shredded_pct"]:
        if col not in df.columns:
            df[col] = 0.0

    df["scrap_quality"] = (
        df["scrap_HBI_pct"] * 1.0
        + df["scrap_Shredded_pct"] * 0.7
        + df["scrap_HMS80_20_pct"] * 0.4
    )

    return df


def get_kwh_features(df: pd.DataFrame):
    feature_cols = [
        "scrap_HMS80_20_pct",
        "scrap_HBI_pct",
        "scrap_Shredded_pct",
        "oxygen_Nm3",
        "tap_time_min",
        "scrap_quality",
        "panel_delta_t",
    ]
    target_col = "kwh_per_t"

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    if target_col not in df.columns:
        df[target_col] = np.nan

    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols, target_col


def get_tap_features(df: pd.DataFrame):
    feature_cols = [
        "power_kWh",
        "oxygen_Nm3",
        "tap_time_min",
        "scrap_quality",
        "panel_delta_t",
    ]
    target_col = "tap_temperature_C"

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    if target_col not in df.columns:
        df[target_col] = np.nan

    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols, target_col


def train_rf_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 6,
):
    """Basit RF modeli. 10 kayıttan azsa None döndürür."""
    mask = ~y.isna()
    X_valid = X[mask]
    y_valid = y[mask]

    if len(X_valid) < 10:
        return None

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X_valid, y_valid)
    return model


def save_model(path: str, model, feature_cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
        },
        path,
    )


def load_model(path: str):
    if not os.path.exists(path):
        return None, None
    data = joblib.load(path)
    return data.get("model"), data.get("feature_cols")


def train_all_models(df: pd.DataFrame, note: str = ""):
    """
    kWh/t ve tap sıcaklığı modellerini birlikte eğitir,
    eğitim durumunu session_state içinde günceller.
    """
    st.session_state["model_status"] = "Eğitiliyor..."
    with st.spinner("Modeller eğitiliyor..."):
        # kWh/t modeli
        Xk, yk, feat_kwh_new, _ = get_kwh_features(df)
        mk = train_rf_model(Xk, yk)

        # Tap sıcaklık modeli
        Xt, yt, feat_tap_new, _ = get_tap_features(df)
        mt = train_rf_model(Xt, yt)

        if mk is not None:
            save_model(MODEL_PATH_KWH, mk, feat_kwh_new)

        if mt is not None:
            save_model(MODEL_PATH_TAP, mt, feat_tap_new)

    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = len(df)

    st.session_state["model_status"] = f"Eğitildi ✅ {note}".strip()
    st.session_state["last_train_time"] = now_str
    st.session_state["last_train_rows"] = rows
    st.session_state["train_count"] += 1
    st.session_state["last_seen_rows"] = rows

    st.success(f"Modeller {rows} şarj verisiyle {now_str} tarihinde eğitildi.")


def generate_time_axis(n_points: int, start_time: dt.time = dt.time(2, 30), step_min: int = 8):
    base = dt.datetime.combine(dt.date.today(), start_time)
    times = [base + dt.timedelta(minutes=i * step_min) for i in range(n_points)]
    return [t.strftime("%H:%M") for t in times]


# ------------------------------------------------------------
# Sayfa içerikleri
# ------------------------------------------------------------
def page_veri_girisi(df: pd.DataFrame):
    st.title("Veri Girişi")
    st.write(
        "Bu sayfada demo için temel veri önizlemesi ve ileride eklenecek manuel giriş / upload fonksiyonları yer alacak."
    )
    st.subheader("Veri Önizleme (İlk 20 kayıt)")
    st.dataframe(df.head(20), use_container_width=True)


def page_canli_veri(df: pd.DataFrame):
    st.title("Canlı Veri (Demo)")
    st.write("Bu sayfada gerçek sahadan gelen canlı veriler izlenecek (şu an demo verisi gösteriliyor).")

    last_20 = df.tail(20).copy()
    last_20.reset_index(drop=True, inplace=True)
    last_20["Heat"] = np.arange(1, len(last_20) + 1)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Tap Sıcaklığı Trendi (Son 20 Şarj)")
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=last_20["Heat"],
                y=last_20["tap_temperature_C"],
                mode="lines+markers",
                name="Tap T",
            )
        )
        fig1.update_layout(
            xaxis_title="Şarj",
            yaxis_title="Tap Sıcaklığı (°C)",
            height=300,
        )
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.subheader("Enerji Tüketimi (kWh/t)")
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=last_20["Heat"],
                y=last_20["kwh_per_t"],
                name="kWh/t",
            )
        )
        fig2.update_layout(
            xaxis_title="Şarj",
            yaxis_title="kWh/t",
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Ham Veri (Son 50 Satır)"):
        st.dataframe(df.tail(50), use_container_width=True)


def page_arc_optimizer(df: pd.DataFrame):
    st.title("Arc Optimizer – Demo")

    # Modelleri yükle
    model_kwh, feat_kwh = load_model(MODEL_PATH_KWH)
    model_tap, feat_tap = load_model(MODEL_PATH_TAP)

    # Üst satır: Sol (simülasyon + KPI'lar) / Sağ (AI model durumu)
    left_col, right_col = st.columns([3, 2])

    # ------------ Sağ üst: AI model durumu ------------
    with right_col:
        st.markdown("#### 🤖 AI Modeli")

        train_mode = st.radio(
            "Eğitim modu",
            [
                "Elle (butonla)",
                "Yeni verilerle (artış olduğunda)",
                "Sürekli eğitim",
            ],
            index=0,
            key="train_mode_arc",
        )

        current_rows = len(df)

        st.write(f"**Durum:** {st.session_state['model_status']}")
        if st.session_state["last_train_time"]:
            st.caption(
                f"Son eğitim: {st.session_state['last_train_time']} · "
                f"Veri sayısı: {st.session_state['last_train_rows']} şarj · "
                f"Toplam eğitim: {st.session_state['train_count']}"
            )
        else:
            st.caption("Model henüz hiç eğitilmedi.")

        st.markdown("---")

        if train_mode == "Elle (butonla)":
            st.caption("Butona bastığında mevcut tüm verilerle modeller yeniden eğitilir.")
            if st.button("Bu verilerle modeli eğit / güncelle", key="btn_train_manual"):
                train_all_models(df, note="(Elle)")
                model_kwh, feat_kwh = load_model(MODEL_PATH_KWH)
                model_tap, feat_tap = load_model(MODEL_PATH_TAP)

        elif train_mode == "Yeni verilerle (artış olduğunda)":
            st.caption("Veri sayısı arttıysa, yeni verilerle modeller otomatik yeniden eğitilir.")
            prev_rows = st.session_state.get("last_seen_rows", 0)
            if current_rows > prev_rows:
                train_all_models(df, note="(Yeni verilerle)")
                model_kwh, feat_kwh = load_model(MODEL_PATH_KWH)
                model_tap, feat_tap = load_model(MODEL_PATH_TAP)
            else:
                st.info("🕒 Yeni veri yok, mevcut modeller kullanılıyor.")
                st.session_state["last_seen_rows"] = current_rows

        elif train_mode == "Sürekli eğitim":
            st.caption("Her yenilemede mevcut verilerle modeller tekrar eğitilir.")
            train_all_models(df, note="(Sürekli)")
            model_kwh, feat_kwh = load_model(MODEL_PATH_KWH)
            model_tap, feat_tap = load_model(MODEL_PATH_TAP)

    # ------------ Sol taraf: Simülasyon + KPI'lar ------------
    with left_col:
        sim_mode = st.checkbox("Simülasyon Modu", value=True)
        if sim_mode:
            st.markdown(
                """
                <div style="background-color:#e8f4ff;padding:8px;border-radius:8px;margin-bottom:10px;">
                ✅ <b>Simülasyon Modu Aktif.</b> Arc Optimizer çıktıları demo için simüle edilen veri üzerinden hesaplanmaktadır.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background-color:#fff4e5;padding:8px;border-radius:8px;margin-bottom:10px;">
                ℹ️ <b>Gerçek Veri Modu (Demo).</b> Şu an yalnızca örnek veriler kullanılıyor.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Özet KPI'lar (son şarj)
        last_row = df.iloc[-1]
        last_10 = df.tail(10)

        son_kwh_per_t = float(last_row.get("kwh_per_t", np.nan))
        son_electrode_kg_per_t = 1.8  # demo sabiti
        son_tap_temp = float(last_row.get("tap_temperature_C", np.nan))
        son_10_avg_kwh = float(last_10["kwh_per_t"].mean())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Son Şarj kWh/t", f"{son_kwh_per_t:.1f}" if not np.isnan(son_kwh_per_t) else "-")
        c2.metric("Son Şarj Elektrot", f"{son_electrode_kg_per_t:.2f} kg/şarj")
        c3.metric("Son Tap Sıcaklığı", f"{son_tap_temp:.0f} °C" if not np.isnan(son_tap_temp) else "-")
        c4.metric("Son 10 Şarj Ort. kWh/t", f"{son_10_avg_kwh:.1f}" if not np.isnan(son_10_avg_kwh) else "-")

    # --------------------------------------------------------
    # Proses Gidişatı – Trend + AI tahmini
    # --------------------------------------------------------
    st.markdown("### Proses Gidişatı – Zaman Trendi ve Tahmini Döküm Anı (AI)")

    trend_df = df.tail(20).copy()
    trend_df.reset_index(drop=True, inplace=True)
    trend_df["Heat"] = np.arange(1, len(trend_df) + 1)
    trend_df["time_str"] = generate_time_axis(len(trend_df))

    ai_kwh = None
    ai_tap_temp = None

    if (model_kwh is not None) and (model_tap is not None):
        Xk_full, _, _, _ = get_kwh_features(df)
        Xt_full, _, _, _ = get_tap_features(df)

        xk_last = Xk_full.iloc[[-1]].copy()
        xt_last = Xt_full.iloc[[-1]].copy()

        # Basit demo "optimizasyon": tap_time_min ve oxygen_Nm3 biraz azalt
        if "tap_time_min" in xk_last.columns:
            xk_last["tap_time_min"] = np.maximum(xk_last["tap_time_min"] - 3, 30)
        if "oxygen_Nm3" in xk_last.columns:
            xk_last["oxygen_Nm3"] = xk_last["oxygen_Nm3"] * 0.95

        if "tap_time_min" in xt_last.columns:
            xt_last["tap_time_min"] = np.maximum(xt_last["tap_time_min"] - 3, 30)
        if "oxygen_Nm3" in xt_last.columns:
            xt_last["oxygen_Nm3"] = xt_last["oxygen_Nm3"] * 0.95

        ai_kwh = float(model_kwh.predict(xk_last)[0])
        ai_tap_temp = float(model_tap.predict(xt_last)[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend_df["time_str"],
            y=trend_df["tap_temperature_C"],
            mode="lines+markers",
            name="Gerçek Tap Sıcaklığı",
            line=dict(width=2),
        )
    )

    if ai_tap_temp is not None:
        fig.add_hline(
            y=ai_tap_temp,
            line_dash="dot",
            line_color="green",
            annotation_text="AI Tahmini Tap Sıcaklığı",
            annotation_position="top left",
        )

    fig.update_layout(
        xaxis_title="Zaman (demo)",
        yaxis_title="Tap Sıcaklığı (°C)",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # Proses Kazanç Analizi (Ton Başına)
    # --------------------------------------------------------
    st.markdown("### Proses Kazanç Analizi (Ton Başına)")

    mevcut_kwh = son_10_avg_kwh
    hedef_kwh = mevcut_kwh - 10 if not np.isnan(mevcut_kwh) else np.nan
    pot_ai_kwh = ai_kwh if ai_kwh is not None else hedef_kwh

    mevcut_elec = 1.8
    hedef_elec = 1.6
    pot_ai_elec = 1.65 if ai_kwh is not None else 1.7

    hedef_tap = 1620.0
    mevcut_tap = son_tap_temp
    pot_ai_tap = ai_tap_temp if ai_tap_temp is not None else mevcut_tap

    mevcut_slop = 30.0
    hedef_slop = 10.0
    pot_ai_slop = 15.0 if ai_kwh is not None else 20.0

    table_df = pd.DataFrame(
        [
            {
                "KPI": "Enerji (kWh/t)",
                "Mevcut": round(mevcut_kwh, 1) if not np.isnan(mevcut_kwh) else None,
                "Hedef": round(hedef_kwh, 1) if not np.isnan(hedef_kwh) else None,
                "Potansiyel (AI)": round(pot_ai_kwh, 1) if pot_ai_kwh is not None else None,
            },
            {
                "KPI": "Elektrot (kg/t)",
                "Mevcut": round(mevcut_elec, 2),
                "Hedef": round(hedef_elec, 2),
                "Potansiyel (AI)": round(pot_ai_elec, 2),
            },
            {
                "KPI": "Tap Sıcaklık Kontrolü (°C)",
                "Mevcut": round(mevcut_tap, 0) if not np.isnan(mevcut_tap) else None,
                "Hedef": round(hedef_tap, 0),
                "Potansiyel (AI)": round(pot_ai_tap, 0) if pot_ai_tap is not None else None,
            },
            {
                "KPI": "Slopping Risk İndeksi",
                "Mevcut": mevcut_slop,
                "Hedef": hedef_slop,
                "Potansiyel (AI)": pot_ai_slop,
            },
        ]
    )

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
    )


# ------------------------------------------------------------
# main()
# ------------------------------------------------------------
def main():
    try:
        raw_df = load_raw_data(DATA_PATH)
    except FileNotFoundError:
        st.error("Veri dosyası bulunamadı: data/BG_EAF_panelcooling_demo.csv")
        st.stop()

    df = prepare_features(raw_df)

    page = st.sidebar.radio(
        "Sayfa Seçimi",
        ["Veri Girişi", "Canlı Veri", "Arc Optimizer"],
        index=2,  # default Arc Optimizer
    )

    if page == "Veri Girişi":
        page_veri_girisi(df)
    elif page == "Canlı Veri":
        page_canli_veri(df)
    elif page == "Arc Optimizer":
        page_arc_optimizer(df)


if __name__ == "__main__":
    main()
