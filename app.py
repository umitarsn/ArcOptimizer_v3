import streamlit as st

# Tüm uygulamayı try/except içine almak,
# import aşamasında hata olursa process'in ölmesini önler.
try:
    # --------------------------------------------------
    # IMPORTS
    # --------------------------------------------------
    import os
    import json
    import random
    from datetime import datetime, timedelta, timezone

    import pandas as pd

    # --------------------------------------------------
    # GENEL AYARLAR
    # --------------------------------------------------
    st.set_page_config(
        page_title="FeCr AI",
        page_icon="apple-touch-icon.png",
        layout="wide",
    )

    # Zaman dilimi: Europe/Istanbul bulunamazsa UTC+3
    try:
        from zoneinfo import ZoneInfo

        TZ = ZoneInfo("Europe/Istanbul")
    except Exception:
        TZ = timezone(timedelta(hours=3))

    SETUP_SAVE_PATH = "data/saved_inputs.json"
    RUNTIME_SAVE_PATH = "data/runtime_data.json"
    os.makedirs("data", exist_ok=True)

    # --------------------------------------------------
    # GLOBAL SESSION STATE
    # --------------------------------------------------
    if "info_state" not in st.session_state:
        st.session_state.info_state = {}

    if "profit_info_state" not in st.session_state:
        st.session_state.profit_info_state = {}

    if "sim_data" not in st.session_state:
        st.session_state.sim_data = None

    if "sim_mode_flag" not in st.session_state:
        st.session_state.sim_mode_flag = None

    # --------------------------------------------------
    # KAYITLI SETUP VERİLERİ
    # --------------------------------------------------
    if os.path.exists(SETUP_SAVE_PATH):
        try:
            with open(SETUP_SAVE_PATH, "r", encoding="utf-8") as f:
                saved_inputs = json.load(f)
            if not isinstance(saved_inputs, dict):
                saved_inputs = {}
        except Exception:
            saved_inputs = {}
    else:
        saved_inputs = {}

    # --------------------------------------------------
    # RUNTIME VERİLERİ
    # --------------------------------------------------
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

    # --------------------------------------------------
    # SİMÜLASYON VERİLERİ
    # --------------------------------------------------
    def generate_simulation_runtime_data(n: int = 15):
        """Simülasyon Modu için örnek şarj datası üretir."""
        sim_list = []
        now = datetime.now(TZ)

        for i in range(n):
            ts = now - timedelta(hours=(n - 1 - i))
            heat_id = f"SIM-{i+1}"

            tap_weight = 35 + random.uniform(-3, 3)          # ton
            kwh_per_t = 420 + random.uniform(-25, 25)        # kWh/t
            energy_kwh = tap_weight * kwh_per_t
            duration_min = 55 + random.uniform(-10, 10)      # dk
            tap_temp = 1610 + random.uniform(-15, 15)        # °C
            o2_flow = 950 + random.uniform(-150, 150)        # Nm³/h
            slag_foaming = random.randint(3, 9)              # 0–10
            panel_delta_t = 18 + random.uniform(-5, 8)       # °C
            electrode_cons = 1.9 + random.uniform(-0.3, 0.3) # kg/şarj

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

    # --------------------------------------------------
    # EXCEL – SETUP
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 1) SETUP SAYFASI
    # --------------------------------------------------
    def show_setup_form():
        st.markdown("## 1. Setup – Sabit Proses / Tasarım Verileri")
        st.markdown(
            "Bu form **dc_saf_soru_tablosu.xlsx** dosyasına göre hazırlanmıştır.\n\n"
            "1. Girişi sadece **Set Değeri** alanına yapınız.\n"
            "2. 🔴 Zorunlu (Önem: 1), 🟡 Faydalı (Önem: 2), ⚪ Opsiyonel (Önem: 3).\n"
            "3. Detaylı bilgi için satır sonundaki ℹ️ butonuna tıklayın."
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
                        s = str(raw_birim).strip()
                        if s.lower() not in ("", "none", "nan"):
                            birim = s

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
                                try:
                                    with open(SETUP_SAVE_PATH, "w", encoding="utf-8") as f:
                                        json.dump(saved_inputs, f, ensure_ascii=False, indent=2)
                                except Exception as e:
                                    st.error(f"Setup verileri kaydedilemedi: {e}")
                        with uc:
                            if birim:
                                st.markdown(f"**{birim}**")

                    with c5:
                        if st.button("ℹ️", key=f"info_{row_key}"):
                            st.session_state.info_state[row_key] = not st.session_state.info_state.get(
                                row_key, False
                            )

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
                        detaylar.append("🔵 **Önem:** " + str(onem))
                        st.info("\n".join(detaylar))

                    total_fields += 1
                    val = str(saved_inputs.get(val_key, "")).strip()
                    if val:
                        total_filled += 1
                        if onem == 1:
                            required_filled += 1
                    if onem == 1:
                        required_fields += 1

        st.sidebar.subheader("📊 Setup Veri Giriş Durumu")
        pct_all = round(100 * total_filled / total_fields, 1) if total_fields else 0
        pct_req = (
            round(100 * required_filled / required_fields, 1)
            if required_fields
            else 0
        )
        st.sidebar.metric("Toplam Giriş Oranı", f"{pct_all}%")
        st.sidebar.progress(min(pct_all / 100, 1.0))
        st.sidebar.metric("Zorunlu Veri Girişi", f"{pct_req}%")
        st.sidebar.progress(min(pct_req / 100, 1.0))
        eksik = required_fields - required_filled
        if eksik > 0:
            st.sidebar.warning(f"❗ Eksik Zorunlu Değerler: {eksik}")

    # --------------------------------------------------
    # 2) CANLI VERİ
    # --------------------------------------------------
    def show_runtime_page(sim_mode: bool):
        st.markdown("## 2. Canlı Veri – Şarj Bazlı Anlık Veriler")
        if sim_mode:
            st.info(
                "🧪 **Simülasyon Modu Aktif.** Aşağıdaki veriler gerçek zamanlı yerine "
                "simülasyon amaçlı oluşturulmuştur."
            )
        else:
            st.markdown(
                "Bu sayfada fırın işletmesi sırasında her **şarj / heat** için toplanan "
                "operasyonel veriler girilir veya otomasyondan okunur."
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
                tap_temp = st.number_input(
                    "Tap Sıcaklığı (°C)", min_value=0.0, max_value=2000.0, step=1.0
                )
            with c6:
                o2_flow = st.number_input("Ortalama O2 Debisi (Nm³/h)", min_value=0.0, step=1.0)

            c7, c8, c9 = st.columns(3)
            with c7:
                slag_foaming = st.slider("Slag Foaming Seviyesi (0–10)", 0, 10, 5)
            with c8:
                panel_delta_t = st.number_input("Panel ΔT (°C)", min_value=0.0, step=0.1)
            with c9:
                electrode_cons = st.number_input(
                    "Elektrot Tüketimi (kg/şarj)", min_value=0.0, step=0.01
                )

            note = st.text_input("Operatör Notu (opsiyonel)", "")

            submitted = st.form_submit_button("Kaydet")

        if submitted:
            if not heat_id:
                st.error("Heat ID / Şarj No zorunlu.")
            else:
                if sim_mode:
                    st.warning("Simülasyon modunda kayıt dosyaya yazılmaz (demo amaçlı).")
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

        if sim_mode:
            data_source = st.session_state.sim_data
        else:
            data_source = runtime_data

        if not data_source:
            st.info("Henüz canlı veri girilmedi.")
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

    # --------------------------------------------------
    # 3) ARC OPTIMIZER
    # --------------------------------------------------
    def show_arc_optimizer_page(sim_mode: bool):
        st.markdown("## 3. Arc Optimizer – Trendler, KPI ve Öneriler")
        if sim_mode:
            st.info(
                "🧪 **Simülasyon Modu Aktif.** Arc Optimizer çıktıları simüle edilen veri üzerinden hesaplanır."
            )

        if sim_mode:
            data_source = st.session_state.sim_data
        else:
            data_source = runtime_data

        if not data_source:
            st.info("Önce 2. sayfadan canlı veri ekleyin.")
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
        c1.metric(
            "Son Şarj kWh/t",
            f"{last['kwh_per_t']:.1f}" if pd.notna(last.get("kwh_per_t", None)) else "-",
        )
        c2.metric(
            "Son Şarj Elektrot",
            f"{last['electrode_kg_per_heat']:.2f} kg/şarj"
            if pd.notna(last.get("electrode_kg_per_heat", None))
            else "-",
        )
        c3.metric(
            "Son Tap Sıcaklığı",
            f"{last['tap_temp_c']:.0f} °C"
            if pd.notna(last.get("tap_temp_c", None))
            else "-",
        )
        c4.metric(
            "Son 10 Şarj Ort. kWh/t",
            f"{avg_kwh_t:.1f}" if avg_kwh_t and not pd.isna(avg_kwh_t) else "-",
        )

        # --------- BASİT ZAMAN GRAFİĞİ (Altair yok) ---------
        st.markdown("### Proses Gidişatı – Zaman Trendi")
        chart_cols = ["kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"]
        chart_df = df[["timestamp_dt"] + chart_cols].set_index("timestamp_dt")
        chart_df = chart_df.rename(
            columns={
                "kwh_per_t": "kWh/t",
                "tap_temp_c": "Tap T (°C)",
                "electrode_kg_per_heat": "Elektrot (kg/şarj)",
            }
        )
        st.line_chart(chart_df)

        # Basit bir “tahmini döküm anı” gösterimi (metinsel)
        min_time = df["timestamp_dt"].min()
        last_time = df["timestamp_dt"].max()
        real_span = last_time - min_time
        if real_span.total_seconds() <= 0:
            real_span = timedelta(hours=6)
        predicted_tap_time = last_time + real_span * 0.2
        delta_min = (predicted_tap_time - last_time).total_seconds() / 60.0

        st.markdown(
            f"**Tahmini Döküm Anı (AI – demo):** "
            f"{predicted_tap_time.strftime('%Y-%m-%d %H:%M')} "
            f"(yaklaşık {delta_min:.0f} dk sonra)"
        )

        # --------------------------------------------------
        # PROSES KAZANÇ ANALİZİ (TON BAŞINA)
        # --------------------------------------------------
        st.markdown("### 💰 Proses Kazanç Analizi (Ton Başına)")

        ENERGY_PRICE_EUR_PER_KWH = 0.12
        ELECTRODE_PRICE_EUR_PER_KG = 3.0

        rows = []
        total_gain_per_t = 0.0

        if pd.notna(last.get("kwh_per_t", None)) and avg_kwh_t and not pd.isna(avg_kwh_t):
            real = float(last["kwh_per_t"])
            target = max(avg_kwh_t - 5.0, 0.0)
            diff = real - target
            gain = abs(diff) * ENERGY_PRICE_EUR_PER_KWH
            total_gain_per_t += gain
            rows.append(
                {
                    "tag": "kwh_per_t",
                    "deg": "Enerji tüketimi",
                    "akt": f"{real:.1f} kWh/t",
                    "pot": f"{target:.1f} kWh/t",
                    "fark": f"{diff:+.1f} kWh/t",
                    "kazanc": f"{gain:.2f} €/t",
                    "type": "cost",
                }
            )

        if pd.notna(last.get("electrode_kg_per_heat", None)) and pd.notna(
            last.get("tap_weight_t", None)
        ):
            tap_w = float(last["tap_weight_t"]) or 0.0
            if tap_w > 0:
                real_pt = float(last["electrode_kg_per_heat"]) / tap_w
                if pd.notna(avg_electrode):
                    target_pt = max(avg_electrode / tap_w, 0.0)
                else:
                    target_pt = max(real_pt - 0.05, 0.0)
                diff = real_pt - target_pt
                gain = abs(diff) * ELECTRODE_PRICE_EUR_PER_KG
                total_gain_per_t += gain
                rows.append(
                    {
                        "tag": "electrode",
                        "deg": "Elektrot tüketimi",
                        "akt": f"{real_pt:.3f} kg/t",
                        "pot": f"{target_pt:.3f} kg/t",
                        "fark": f"{diff:+.3f} kg/t",
                        "kazanc": f"{gain:.2f} €/t",
                        "type": "cost",
                    }
                )

        if pd.notna(last.get("tap_temp_c", None)) and avg_tap_temp and not pd.isna(
            avg_tap_temp
        ):
            real = float(last["tap_temp_c"])
            target = float(avg_tap_temp)
            diff = real - target
            tap_gain_range = "0.03–0.10 €/t + Kalite ↑"
            rows.append(
                {
                    "tag": "tap_temp_c",
                    "deg": "Tap sıcaklığı optimizasyonu",
                    "akt": f"{real:.0f} °C",
                    "pot": f"{target:.0f} °C",
                    "fark": f"{diff:+.0f} °C",
                    "kazanc": tap_gain_range,
                    "type": "mixed",
                }
            )

        if pd.notna(last.get("panel_delta_t_c", None)):
            real = float(last["panel_delta_t_c"])
            target = 20.0
            diff = real - target
            rows.append(
                {
                    "tag": "panel_delta_t",
                    "deg": "Panel ΔT",
                    "akt": f"{real:.1f} °C",
                    "pot": f"{target:.1f} °C",
                    "fark": f"{diff:+.1f} °C",
                    "kazanc": "Kalite ↑",
                    "type": "quality",
                }
            )

        slag_val = None
        if last.get("slag_foaming_index", None) is not None:
            slag_val = float(last["slag_foaming_index"])
            target = 7.0
            diff = slag_val - target
            rows.append(
                {
                    "tag": "slag_foaming",
                    "deg": "Köpük yüksekliği / slag foaming",
                    "akt": f"{slag_val:.1f}",
                    "pot": f"{target:.1f}",
                    "fark": f"{diff:+.1f}",
                    "kazanc": "Enerji verimliliği ↑, elektrot ve refrakter tüketimi ↓",
                    "type": "quality",
                }
            )

        if pd.notna(last.get("tap_temp_c", None)) and pd.notna(
            last.get("panel_delta_t_c", None)
        ):
            t_act = float(last["tap_temp_c"])
            dT_act = float(last["panel_delta_t_c"])

            if avg_tap_temp is not None and not pd.isna(avg_tap_temp):
                dt_from_avg = t_act - float(avg_tap_temp)
            else:
                dt_from_avg = 0.0

            if dt_from_avg > 20 or dT_act > 30:
                refr_level = "Yüksek risk"
            elif dt_from_avg > 10 or dT_act > 25:
                refr_level = "Orta"
            else:
                refr_level = "Düşük"

            rows.append(
                {
                    "tag": "refractory_wear",
                    "deg": "Refrakter aşınma seviyesi",
                    "akt": refr_level,
                    "pot": "AI kontrollü optimum bölge",
                    "fark": "-",
                    "kazanc": "Refrakter ömrü ↑, planlı duruşlar dışında duruş ↓",
                    "type": "quality",
                }
            )

        if (
            pd.notna(last.get("kwh_per_t", None))
            and avg_kwh_t
            and not pd.isna(avg_kwh_t)
            and pd.notna(last.get("tap_temp_c", None))
            and avg_tap_temp
            and not pd.isna(avg_tap_temp)
        ):
            score = 0
            if slag_val is not None and slag_val >= 7.0:
                score += 1
            if abs(float(last["kwh_per_t"]) - float(avg_kwh_t)) <= 10:
                score += 1
            if abs(float(last["tap_temp_c"]) - float(avg_tap_temp)) <= 10:
                score += 1

            if score == 3:
                mix_level = "İyi"
            elif score == 2:
                mix_level = "Orta"
            else:
                mix_level = "Riskli"

            rows.append(
                {
                    "tag": "mix_quality",
                    "deg": "Karışım kalitesi (homojenlik)",
                    "akt": mix_level,
                    "pot": "AI ile stabil ve homojen bölge",
                    "fark": "-",
                    "kazanc": "Kalite ↑, iç hurda ve yeniden işleme ↓",
                    "type": "quality",
                }
            )

        widths = [1.0, 2.0, 1.3, 1.3, 1.1, 1.8, 0.5]
        hcols = st.columns(widths)
        hcols[0].markdown("**Tag**")
        hcols[1].markdown("**Değişken**")
        hcols[2].markdown("**Aktüel**")
        hcols[3].markdown("**Potansiyel (AI)**")
        hcols[4].markdown("**Fark**")
        hcols[5].markdown("**Tahmini Kazanç**")
        hcols[6].markdown("")

        profit_state = st.session_state.profit_info_state

        for row in rows:
            cols = st.columns(widths)
            cols[0].markdown(row["tag"])
            cols[1].markdown(row["deg"])
            cols[2].markdown(row["akt"])
            cols[3].markdown(row["pot"])
            cols[4].markdown(row["fark"])
            cols[5].markdown(row["kazanc"])

            btn_key = f"profit_info_btn_{row['tag']}"
            if cols[6].button("ℹ️", key=btn_key):
                profit_state[row["tag"]] = not profit_state.get(row["tag"], False)

        st.markdown(
            f"**Toplam Potansiyel Kazanç (AI tahmini, ton başına – doğrudan hesaplanabilen kalemler):** "
            f"≈ **{total_gain_per_t:,.1f} €/t**"
        )

        # Satır bazlı info kutuları (kısaltılmış açıklamalar)
        for row in rows:
            if profit_state.get(row["tag"], False):
                if row["tag"] == "kwh_per_t":
                    st.info(
                        "Enerji tüketimi farkı kWh/t bazında hesaplanır ve enerji birim fiyatı "
                        "ile çarpılarak €/t kazanç tahmini yapılır."
                    )
                elif row["tag"] == "electrode":
                    st.info(
                        "Elektrot tüketimi kg/t bazında değerlendirilir. Stabil ark ve iyi köpük, "
                        "elektrot tüketimini düşürerek maliyet kazancı sağlar."
                    )
                elif row["tag"] == "tap_temp_c":
                    st.info(
                        "Tap sıcaklığının gereksiz yüksek olması enerji kaybı ve kalite riskidir. "
                        "Optimum aralıkta tutmak hem enerji tasarrufu hem de kalite iyileşmesi sağlar."
                    )
                elif row["tag"] == "panel_delta_t":
                    st.info(
                        "Panel ΔT, duvar yükü ve ısıl profil için göstergedir. Uygun seviyede tutulması "
                        "iç hurda ve ısı kayıplarını azaltır."
                    )
                elif row["tag"] == "slag_foaming":
                    st.info(
                        "Yeterli slag foaming, arkı örtüp enerji verimliliğini artırır; aşırı veya yetersiz "
                        "seviyeler enerji ve kalite sorunlarına yol açabilir."
                    )
                elif row["tag"] == "refractory_wear":
                    st.info(
                        "Tap sıcaklığı ve panel ΔT kombinasyonu, refrakter aşınma riskinin temel göstergesidir. "
                        "AI ile optimum bölgeye yakın çalışmak refrakter ömrünü uzatır."
                    )
                elif row["tag"] == "mix_quality":
                    st.info(
                        "Karışım kalitesi; enerji girişi, sıcaklık stabilitesi ve köpük seviyesinin "
                        "birleşik sonucudur. Homojen banyo iç hurdayı azaltır."
                    )

        st.markdown("### Model Önerileri (Örnek / Demo Mantık)")
        suggestions = []

        if (
            pd.notna(last.get("kwh_per_t", None))
            and avg_kwh_t
            and not pd.isna(avg_kwh_t)
            and last["kwh_per_t"] > avg_kwh_t * 1.05
        ):
            suggestions.append(
                "🔌 Son şarjın **kWh/t** değeri son 10 şarj ortalamasına göre yüksek. "
                "Oksijen debisi ve güç profilini gözden geçirmeyi düşünün."
            )

        if (
            pd.notna(last.get("electrode_kg_per_heat", None))
            and avg_electrode
            and not pd.isna(avg_electrode)
            and last["electrode_kg_per_heat"] > avg_electrode * 1.10
        ):
            suggestions.append(
                "🧯 **Elektrot tüketimi** son şarjda yükselmiş. Ark stabilitesi (arc length, voltage) "
                "ve elektrot hareketlerini kontrol edin."
            )

        if (
            pd.notna(last.get("tap_temp_c", None))
            and avg_tap_temp
            and not pd.isna(avg_tap_temp)
            and last["tap_temp_c"] < avg_tap_temp - 10
        ):
            suggestions.append(
                "🔥 Tap sıcaklığı son şarjda düşük. Bir sonraki şarj için enerji girişini hafif artırmak "
                "veya şarj sonu bekleme süresini optimize etmek gerekebilir."
            )

        if last.get("slag_foaming_index", None) is not None and last["slag_foaming_index"] >= 8:
            suggestions.append(
                "🌋 Slag foaming seviyesi yüksek (≥8). Karbon/O₂ dengesini ve köpük kontrolünü gözden geçirin."
            )

        if last.get("panel_delta_t_c", None) is not None and last["panel_delta_t_c"] > 25:
            suggestions.append(
                "💧 Panel ΔT yüksek. Soğutma devresinde dengesizlik olabilir; panel debilerini ve "
                "tıkalı hatları kontrol edin."
            )

        if saving_potential > 0.0:
            suggestions.append(
                f"📉 kWh/t trendine göre yaklaşık **{saving_potential:.1f} kWh/t** "
                "iyileştirme potansiyeli görülüyor."
            )

        if not suggestions:
            suggestions.append(
                "✅ Model açısından belirgin bir anomali/iyileştirme alarmı yok. Mevcut ayarlar stabil görünüyor."
            )

        for s in suggestions:
            st.markdown(f"- {s}")

    # --------------------------------------------------
    # MAIN
    # --------------------------------------------------
    def main():
        with st.sidebar:
            try:
                st.image("apple-touch-icon.png", width=72)
            except Exception:
                pass
            st.markdown("### FeCr AI")

            sim_mode = st.toggle(
                "Simülasyon Modu",
                value=False,
                help="Açıkken sistem canlı veri yerine simüle edilmiş veri kullanır.",
            )

            if sim_mode:
                if (
                    st.session_state.sim_mode_flag is not True
                    or st.session_state.sim_data is None
                ):
                    st.session_state.sim_data = generate_simulation_runtime_data()
                    st.session_state.sim_mode_flag = True
            else:
                st.session_state.sim_mode_flag = False
                st.session_state.sim_data = None

            page = st.radio("Sayfa Seç", ["1. Setup", "2. Canlı Veri", "3. Arc Optimizer"])

        if page == "1. Setup":
            show_setup_form()
        elif page == "2. Canlı Veri":
            show_runtime_page(sim_mode)
        else:
            show_arc_optimizer_page(sim_mode)

    if __name__ == "__main__":
        main()

except Exception as e:
    st.error("Uygulama başlatılırken beklenmeyen bir hata oluştu.")
    st.exception(e)
