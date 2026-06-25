# ============================================================
# 🗺️  Peta Keheningan Global
# Analisis Spasio-Temporal Keterwakilan Perempuan di Parlemen
# Nasional + Proyeksi Paritas Gender (ARIMA), 1997–2025
# ============================================================
# Dibangun dengan: Streamlit + Folium + Plotly
#
# CATATAN UNTUK PENGGUNA:
# Skrip ini mengasumsikan struktur kolom berikut (sesuaikan
# nama kolom di bagian "KONFIGURASI KOLOM" bila berbeda):
#
#   panel_data_clean.csv        : iso3, country, region, year,
#                                  pct_women, quota_type
#   map_data_with_forecast.csv  : iso3, country, region, quota_type,
#                                  pct_2025, rmse, mae, model_order,
#                                  predicted_parity_year, category, note
#   countries.geojson           : Natural Earth (datasets/geo-countries),
#                                  properti nama berbeda-beda antar versi
#                                  rilis -> dideteksi otomatis di bawah.
# ============================================================

import json
import os
import time
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from branca.colormap import LinearColormap
from shapely.geometry import shape
from streamlit_folium import st_folium

# ------------------------------------------------------------
# KONFIGURASI KOLOM (ubah di sini kalau nama kolom berbeda)
# ------------------------------------------------------------
COL_ISO       = "iso3"
COL_COUNTRY   = "country"
COL_REGION    = "region"
COL_YEAR      = "year"
COL_PCT       = "pct_women"
COL_QUOTA     = "quota_type"

PARITY_TARGET   = 50.0
CAP_YEAR_GREEN  = 2063
CAP_YEAR_ORANGE = 2100

DATA_DIR = Path(__file__).parent / "data"

CATEGORY_COLORS = {
    "putih":  "#FFFFFF",
    "hijau":  "#27AE60",
    "oranye": "#F39C12",
    "merah":  "#C0392B",
}
CATEGORY_LABELS = {
    "putih":  f"Sudah ≥ {int(PARITY_TARGET)}%",
    "hijau":  f"Paritas sebelum {CAP_YEAR_GREEN}",
    "oranye": f"Paritas {CAP_YEAR_GREEN}–{CAP_YEAR_ORANGE}",
    "merah":  f"Paritas setelah {CAP_YEAR_ORANGE} / tidak konvergen",
}
REGION_ORDER = ["Amerika", "Asia-Pasifik", "Afrika", "Eropa", "MENA"]
BRAND_PURPLE = "#5B2A86"
ACTUAL_GRADIENT = ["#FFFFFF", "#E4D4ED", "#B98CCB", "#8A4FA8", "#5B2A86", "#33134D"]

# ------------------------------------------------------------
# PAGE CONFIG & STYLE
# ------------------------------------------------------------
st.set_page_config(
    page_title="Peta Keheningan Global",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #FAFAFC; }}
    h1, h2, h3 {{ color: {BRAND_PURPLE} !important; }}
    div[data-testid="stMetric"] {{
        background: #FFFFFF !important; border-radius: 10px; padding: 10px 14px;
        border: 1px solid #ECE3F3;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] * {{
        color: #1A1A2E !important;
    }}
    .legend-row {{ display:flex; align-items:center; gap:8px; margin:2px 0; font-size:0.85rem; }}
    .legend-swatch {{ width:14px; height:14px; border-radius:3px; border:1px solid #999; display:inline-block; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------
@st.cache_data(show_spinner="Memuat data panel & hasil forecasting...")
def load_data():
    df_panel = pd.read_csv(DATA_DIR / "panel_data_clean.csv")
    df_map = pd.read_csv(DATA_DIR / "map_data_with_forecast.csv")
    with open(DATA_DIR / "countries.geojson", encoding="utf-8") as f:
        geo = json.load(f)
    return df_panel, df_map, geo


def detect_iso_key(geo: dict) -> str:
    """Beberapa rilis Natural Earth/datasets-geo-countries memakai nama
    properti berbeda untuk kode ISO3 (ISO_A3, ISO3166-1-Alpha-3, dst).
    Deteksi otomatis supaya join dengan data tidak putus diam-diam."""
    candidates = ["ISO_A3", "ISO3166-1-Alpha-3", "iso_a3", "ISO_A3_EH", "ADM0_A3"]
    props = geo["features"][0]["properties"]
    for c in candidates:
        if c in props:
            return c
    # fallback: cari key apapun yang isinya 3 huruf kapital untuk feature pertama
    for k, v in props.items():
        if isinstance(v, str) and len(v) == 3 and v.isupper():
            return k
    raise KeyError(
        "Tidak menemukan kolom kode ISO3 di countries.geojson. "
        f"Properti yang tersedia: {list(props.keys())}"
    )


def detect_name_key(geo: dict, iso_key: str) -> str:
    candidates = ["ADMIN", "name", "NAME", "admin"]
    props = geo["features"][0]["properties"]
    for c in candidates:
        if c in props:
            return c
    # fallback: properti string pertama yang bukan kolom iso
    for k, v in props.items():
        if isinstance(v, str) and k != iso_key:
            return k
    return iso_key


@st.cache_data(show_spinner=False)
def compute_centroids(_geo: dict, iso_key: str) -> dict:
    """representative_point() dipakai bukan centroid biasa, supaya titik
    marker selalu jatuh DI DALAM poligon (penting untuk negara kepulauan
    / multi-part seperti Indonesia, Filipina, dll)."""
    pts = {}
    for feat in _geo["features"]:
        code = feat["properties"].get(iso_key)
        if not code or code == "-99":
            continue
        try:
            geom = shape(feat["geometry"])
            p = geom.representative_point()
            pts[code] = (p.y, p.x)  # (lat, lon)
        except Exception:
            continue
    return pts


@st.cache_data(show_spinner=False)
def compute_milestones(df_panel: pd.DataFrame) -> pd.DataFrame:
    """Tahun pertama setiap negara melewati ambang 30% dan 50%."""
    def first_cross(group, threshold):
        hit = group.loc[group[COL_PCT] >= threshold, COL_YEAR]
        return int(hit.min()) if len(hit) else None

    rows = []
    for iso, g in df_panel.groupby(COL_ISO):
        g = g.sort_values(COL_YEAR)
        rows.append({
            COL_ISO: iso,
            "first_year_30": first_cross(g, 30.0),
            "first_year_50": first_cross(g, 50.0),
        })
    return pd.DataFrame(rows)


def get_field(iso, field, df_map, df_panel_latest, default="—"):
    """Ambil field dari df_map; kalau kolomnya tidak ada / kosong,
    jatuh ke df_panel (baris tahun terakhir) sebagai cadangan."""
    row_map = df_map[df_map[COL_ISO] == iso]
    if field in df_map.columns and len(row_map) and pd.notna(row_map[field].iloc[0]):
        return row_map[field].iloc[0]
    row_panel = df_panel_latest[df_panel_latest[COL_ISO] == iso]
    if field in df_panel_latest.columns and len(row_panel) and pd.notna(row_panel[field].iloc[0]):
        return row_panel[field].iloc[0]
    return default


# ------------------------------------------------------------
# LOAD EVERYTHING
# ------------------------------------------------------------
try:
    df_panel, df_map, geo = load_data()
except FileNotFoundError as e:
    st.error(
        "⚠️ File data tidak ditemukan di folder `data/`. Pastikan "
        "`panel_data_clean.csv`, `map_data_with_forecast.csv`, dan "
        "`countries.geojson` sudah ada di sana.\n\n"
        f"Detail: {e}"
    )
    st.stop()

ISO_KEY = detect_iso_key(geo)
NAME_KEY = detect_name_key(geo, ISO_KEY)
CENTROIDS = compute_centroids(geo, ISO_KEY)
df_milestones = compute_milestones(df_panel)

YEAR_MIN, YEAR_MAX = int(df_panel[COL_YEAR].min()), int(df_panel[COL_YEAR].max())
df_panel_latest = df_panel[df_panel[COL_YEAR] == YEAR_MAX]

regions_available = [r for r in REGION_ORDER if r in df_panel[COL_REGION].unique()] or \
    sorted(df_panel[COL_REGION].dropna().unique().tolist())

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.title("🗺️ Peta Keheningan Global")
st.sidebar.caption(
    "Keterwakilan perempuan di parlemen, 1997–2025, "
    "dengan proyeksi tahun paritas (ARIMA per negara)."
)

mode = st.sidebar.radio(
    "Mode tampilan",
    ["Data historis (1997–2025)", "Proyeksi ARIMA — kapan paritas?"],
)
if mode == "Data historis (1997–2025)":
    st.sidebar.caption(
        "📍 Menampilkan % kursi perempuan **aktual** per tahun (gradien putih→ungu). "
        "Pilih mode di atas untuk lihat **proyeksi ARIMA**."
    )
else:
    st.sidebar.caption(
        "🔮 Peta beralih ke **hasil model ARIMA per negara**: warna menunjukkan "
        "perkiraan kapan negara itu mencapai paritas 50% (hijau/oranye/merah/putih, "
        "lihat legenda di bawah). Klik sebuah negara untuk lihat tahun proyeksi, RMSE & MAE-nya."
    )

st.sidebar.markdown("---")

selected_regions = st.sidebar.multiselect(
    "Kawasan", regions_available, default=regions_available
)
quota_options = sorted(df_panel[COL_QUOTA].dropna().unique().tolist()) if COL_QUOTA in df_panel.columns else []
selected_quotas = st.sidebar.multiselect(
    "Jenis kuota gender", quota_options, default=quota_options
) if quota_options else []

show_milestones = st.sidebar.checkbox(
    "📍 Tampilkan momen historis (lintas 30% / 50%)", value=False
)

selected_year = YEAR_MAX  # default kalau mode proyeksi

if mode == "Data historis (1997–2025)":
        st.sidebar.markdown("##### Tahun")
        
        # Slider manual yang sangat simpel dan anti-error
        selected_year = st.sidebar.slider(
            "Pilih Tahun", 
            min_value=YEAR_MIN, 
            max_value=YEAR_MAX, 
            value=YEAR_MAX, # Default awal di tahun terakhir
            key="year_slider_ui",
            label_visibility="collapsed"
        )

        # Tampilan Legenda — % kursi perempuan
        st.sidebar.markdown("**Legenda — % kursi perempuan**")
        grad_css = ", ".join(ACTUAL_GRADIENT)
        st.sidebar.markdown(
            f"""<div style="height:14px;border-radius:4px;
            background:linear-gradient(to right, {grad_css});
            border:1px solid #ccc;"></div>
            <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#666;">
            <span>0%</span><span>{int(PARITY_TARGET)}%+</span></div>""",
            unsafe_allow_html=True,
        )
else:
    st.sidebar.markdown("**Legenda — kategori proyeksi**")
    for cat, label in CATEGORY_LABELS.items():
        border = "1px solid #999" if cat != "putih" else "1px solid #444"
        st.sidebar.markdown(
            f'<div class="legend-row"><span class="legend-swatch" '
            f'style="background:{CATEGORY_COLORS[cat]};border:{border};"></span>{label}</div>',
            unsafe_allow_html=True,
        )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Sumber: Our World in Data Women in Parliaments· World Bank SG.GEN.PARL.ZS · IDEA Gender Quotas · "
    "Natural Earth. Model: `auto_arima` (pmdarima), fallback regresi linear."
)

# ------------------------------------------------------------
# FILTER DATA SESUAI SIDEBAR
# ------------------------------------------------------------
mask_region = df_panel[COL_REGION].isin(selected_regions) if selected_regions else True
mask_quota = df_panel[COL_QUOTA].isin(selected_quotas) if selected_quotas and COL_QUOTA in df_panel.columns else True
df_panel_f = df_panel[mask_region & mask_quota] if selected_regions or selected_quotas else df_panel

iso_allowed = set(df_panel_f[COL_ISO].unique())

# ------------------------------------------------------------
# HEADER & METRIK RINGKAS
# ------------------------------------------------------------
st.title("Peta Keheningan Global")
st.caption(
    "Analisis Spasio-Temporal Keterwakilan Perempuan di Parlemen Nasional & "
    "Proyeksi Paritas Gender Menggunakan ARIMA (1997–2025)"
)

df_now = df_panel[(df_panel[COL_YEAR] == selected_year) & (df_panel[COL_ISO].isin(iso_allowed))]
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Rata-rata global ({selected_year})", f"{df_now[COL_PCT].mean():.1f}%" if len(df_now) else "—")
m2.metric("Negara dianalisis", f"{df_panel[COL_ISO].nunique()}")
n_parity_now = int((df_now[COL_PCT] >= PARITY_TARGET).sum()) if len(df_now) else 0
m3.metric("Sudah capai paritas", f"{n_parity_now} negara")
if "rmse" in df_map.columns:
    m4.metric("RMSE rata-rata model", f"{df_map['rmse'].mean():.2f} pp")
else:
    m4.metric("Rentang data", f"{YEAR_MIN}–{YEAR_MAX}")

st.markdown("")

# ------------------------------------------------------------
# BANGUN PETA FOLIUM
# ------------------------------------------------------------
def style_actual(feature, colormap, values_by_iso):
    code = feature["properties"].get(ISO_KEY)
    val = values_by_iso.get(code)
    if code not in iso_allowed:
        return {"fillColor": "#EEEEEE", "color": "#CCCCCC", "weight": 0.4, "fillOpacity": 0.25}
    if val is None or pd.isna(val):
        return {"fillColor": "#EEEEEE", "color": "#BBBBBB", "weight": 0.5, "fillOpacity": 0.4}
    return {
        "fillColor": colormap(min(val, PARITY_TARGET)),
        "color": "#888888",
        "weight": 0.6,
        "fillOpacity": 0.88,
    }


def style_forecast(feature, category_by_iso):
    code = feature["properties"].get(ISO_KEY)
    cat = category_by_iso.get(code)
    if code not in iso_allowed:
        return {"fillColor": "#EEEEEE", "color": "#CCCCCC", "weight": 0.4, "fillOpacity": 0.25}
    if cat is None:
        return {"fillColor": "#EEEEEE", "color": "#BBBBBB", "weight": 0.5, "fillOpacity": 0.4}
    return {
        "fillColor": CATEGORY_COLORS.get(cat, "#EEEEEE"),
        "color": "#444444" if cat == "putih" else "#777777",
        "weight": 0.8 if cat == "putih" else 0.6,
        "fillOpacity": 0.88,
    }


def build_map():
    m = folium.Map(location=[15, 10], zoom_start=2, tiles="CartoDB positron", min_zoom=2)

    if mode == "Data historis (1997–2025)":
        values_by_iso = (
            df_panel[df_panel[COL_YEAR] == selected_year]
            .set_index(COL_ISO)[COL_PCT].to_dict()
        )
        colormap = LinearColormap(
            ACTUAL_GRADIENT, vmin=0, vmax=PARITY_TARGET
        )
        gj = folium.GeoJson(
            geo,
            name="Persentase kursi perempuan",
            style_function=lambda f: style_actual(f, colormap, values_by_iso),
            highlight_function=lambda f: {"weight": 2, "color": BRAND_PURPLE},
            tooltip=folium.GeoJsonTooltip(
                fields=[NAME_KEY],
                aliases=["Negara:"],
            ),
        )
        gj.add_to(m)
    else:
        category_by_iso = df_map.set_index(COL_ISO)["category"].to_dict() if "category" in df_map.columns else {}
        gj = folium.GeoJson(
            geo,
            name="Kategori proyeksi paritas",
            style_function=lambda f: style_forecast(f, category_by_iso),
            highlight_function=lambda f: {"weight": 2, "color": BRAND_PURPLE},
            tooltip=folium.GeoJsonTooltip(
                fields=[NAME_KEY],
                aliases=["Negara:"],
            ),
        )
        gj.add_to(m)

    if show_milestones:
        merged = df_milestones.merge(
            df_panel[[COL_ISO, COL_COUNTRY]].drop_duplicates(), on=COL_ISO, how="left"
        )
        for _, row in merged.iterrows():
            if row[COL_ISO] not in iso_allowed:
                continue
            if pd.isna(row["first_year_30"]) and pd.isna(row["first_year_50"]):
                continue
            latlon = CENTROIDS.get(row[COL_ISO])
            if not latlon:
                continue
            label_parts = []
            if pd.notna(row["first_year_30"]):
                label_parts.append(f"30% pada {int(row['first_year_30'])}")
            if pd.notna(row["first_year_50"]):
                label_parts.append(f"50% pada {int(row['first_year_50'])}")
            folium.CircleMarker(
                location=latlon,
                radius=4,
                color=BRAND_PURPLE,
                fill=True,
                fill_color=BRAND_PURPLE,
                fill_opacity=0.85,
                weight=1,
                tooltip=f"{row[COL_COUNTRY]}: {' · '.join(label_parts)}",
            ).add_to(m)

    return m


fmap = build_map()

# ------------------------------------------------------------
# LAYOUT: PETA (kiri) + PANEL DETAIL NEGARA (kanan)
# ------------------------------------------------------------
col_map, col_detail = st.columns([2, 1])

with col_map:
    map_state = st_folium(
        fmap,
        height=560,
        use_container_width=True,
        returned_objects=["last_active_drawing"],
        key="main_map",
    )
    st.caption("💡 Klik sebuah negara di peta untuk melihat detail trennya di panel sebelah.")

with col_detail:
    st.markdown("#### Detail negara")
    clicked = (map_state or {}).get("last_active_drawing")
    if not clicked:
        st.info("Belum ada negara dipilih. Klik salah satu wilayah pada peta.")
    else:
        iso = clicked["properties"].get(ISO_KEY)
        country_rows = df_panel[df_panel[COL_ISO] == iso].sort_values(COL_YEAR)
        if iso not in iso_allowed or country_rows.empty:
            st.warning("Negara ini tidak tercakup dalam data panel (mungkin terfilter, atau wilayah non-negara di GeoJSON).")
        else:
            name = country_rows[COL_COUNTRY].iloc[0]
            region = country_rows[COL_REGION].iloc[0]
            quota = get_field(iso, COL_QUOTA, df_map, df_panel_latest)
            current_pct = country_rows[country_rows[COL_YEAR] == selected_year][COL_PCT]
            current_pct = current_pct.iloc[0] if len(current_pct) else country_rows[COL_PCT].iloc[-1]
            rmse = get_field(iso, "rmse", df_map, df_panel_latest, default=None)
            mae = get_field(iso, "mae", df_map, df_panel_latest, default=None)
            pred_year = get_field(iso, "predicted_parity_year", df_map, df_panel_latest, default=None)
            category = get_field(iso, "category", df_map, df_panel_latest, default=None)
            note = get_field(iso, "note", df_map, df_panel_latest, default="")

            st.markdown(f"### {name}")
            st.caption(f"{region} · Kuota gender: **{quota}**")
            st.metric(f"% kursi perempuan ({selected_year})", f"{current_pct:.1f}%")

            if pred_year is not None and pd.notna(pred_year):
                cat_color = CATEGORY_COLORS.get(category, "#999999")
                st.markdown(
                    f"**Proyeksi tahun paritas:** "
                    f'<span style="background:{cat_color};padding:2px 8px;border-radius:6px;'
                    f'color:{"black" if category=="putih" else "white"};font-weight:600;">'
                    f"{int(pred_year)}</span>",
                    unsafe_allow_html=True,
                )
            elif category is not None:
                st.markdown(f"**Kategori proyeksi:** {CATEGORY_LABELS.get(category, category)}")
            if note:
                st.caption(f"Catatan model: {note}")
            if rmse is not None and pd.notna(rmse):
                st.caption(f"RMSE: {float(rmse):.2f} pp · MAE: {float(mae):.2f} pp")

            # Grafik tren mini per negara
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=country_rows[COL_YEAR], y=country_rows[COL_PCT],
                mode="lines+markers", name="Aktual",
                line=dict(color=BRAND_PURPLE, width=2), marker=dict(size=4),
            ))
            fig.add_hline(y=PARITY_TARGET, line_dash="dot", line_color="red", opacity=0.6,
                           annotation_text="Paritas 50%", annotation_font_size=10)
            ms = df_milestones[df_milestones[COL_ISO] == iso]
            if len(ms):
                if pd.notna(ms["first_year_30"].iloc[0]):
                    fig.add_vline(x=ms["first_year_30"].iloc[0], line_dash="dash",
                                  line_color="#F39C12", opacity=0.6)
                if pd.notna(ms["first_year_50"].iloc[0]):
                    fig.add_vline(x=ms["first_year_50"].iloc[0], line_dash="dash",
                                  line_color="#27AE60", opacity=0.6)
            fig.update_layout(
                height=220, margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False, plot_bgcolor="white",
                xaxis_title=None, yaxis_title="% kursi",
            )
            st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------
# GRAFIK TREN REGIONAL (5 kawasan)
# ------------------------------------------------------------
st.markdown("---")
st.markdown("#### Tren regional, 1997–2025")

df_region_trend = (
    df_panel_f.groupby([COL_REGION, COL_YEAR])[COL_PCT]
    .mean().reset_index()
)
fig_region = go.Figure()
palette = ["#5B2A86", "#27AE60", "#F39C12", "#2E86C1", "#C0392B"]
for i, region in enumerate(regions_available):
    sub = df_region_trend[df_region_trend[COL_REGION] == region]
    if sub.empty:
        continue
    fig_region.add_trace(go.Scatter(
        x=sub[COL_YEAR], y=sub[COL_PCT], mode="lines", name=region,
        line=dict(width=2.5, color=palette[i % len(palette)]),
    ))
fig_region.add_hline(y=PARITY_TARGET, line_dash="dot", line_color="red", opacity=0.5,
                      annotation_text="Target paritas 50%")
fig_region.update_layout(
    height=380, margin=dict(l=10, r=10, t=20, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    yaxis_title="Rata-rata % kursi perempuan", xaxis_title=None,
    plot_bgcolor="white",
)
st.plotly_chart(fig_region, width="stretch")

# ------------------------------------------------------------
# BONUS — KUOTA GENDER vs KECEPATAN PERTUMBUHAN (opsional)
# ------------------------------------------------------------
if COL_QUOTA in df_panel.columns:
    with st.expander("📊 Insight Tambahan — Kuota Gender vs Kecepatan Pertumbuhan"):
        def slope_of(g):
            g = g.sort_values(COL_YEAR)
            if g[COL_YEAR].nunique() < 2:
                return np.nan
            s, _ = np.polyfit(g[COL_YEAR], g[COL_PCT], 1)
            return s

        slopes = (
            df_panel.groupby(COL_ISO)
            .apply(slope_of)
            .rename("slope_pct_per_year")
            .reset_index()
        )
        quota_lookup = df_panel.drop_duplicates(COL_ISO)[[COL_ISO, COL_QUOTA]]
        slopes = slopes.merge(quota_lookup, on=COL_ISO, how="left")
        agg = slopes.groupby(COL_QUOTA)["slope_pct_per_year"].mean().sort_values(ascending=False)

        fig_bar = go.Figure(go.Bar(
            x=agg.values, y=agg.index, orientation="h",
            marker_color=BRAND_PURPLE,
        ))
        fig_bar.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Rata-rata kecepatan pertumbuhan (poin persentase / tahun)",
            yaxis_title=None, plot_bgcolor="white",
        )
        st.plotly_chart(fig_bar, width="stretch")
        st.caption(
            "Korelasi sederhana, bukan inferensi kausal — kuota legislated/reserved "
            "cenderung berasosiasi dengan kecepatan pertumbuhan lebih tinggi, "
            "tapi faktor lain (konteks politik, penegakan kuota) tidak dikontrol di sini."
        )

st.markdown("---")
st.caption(
    "Dibangun dengan Streamlit · Folium · Plotly — bagian dari proyek "
    "*Peta Keheningan Global* (UAS Visualisasi Data Spasio-Temporal)."
)
