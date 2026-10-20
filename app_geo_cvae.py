import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lasio
import io
import zipfile
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import folium
from streamlit_folium import st_folium

# ==========================================
# 1. АРХИТЕКТУРА CVAE И ЗАГРУЗКА ВЕСОВ
# ==========================================
class GeologicalCVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super(GeologicalCVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, input_dim * 2),
            nn.Tanh()
        )

    def decode(self, z):
        return self.decoder(z).view(-1, 2, self.input_dim)

@st.cache_resource
def load_cvae(weights_path="cvae_weights.pt"):
    try:
        ckpt = torch.load(weights_path, map_location=torch.device('cpu'))
        net = GeologicalCVAE(input_dim=ckpt['input_dim'], latent_dim=4)
        net.load_state_dict(ckpt['model_state'], strict=False)
        net.eval()
        return net, ckpt
    except Exception:
        return None, None

# ==========================================
# 2. СЛОВАРЬ СИНОНИМОВ И ОЧИСТКА ДАННЫХ
# ==========================================
CURVE_SYNONYMS = {
    'GK': ['GK', 'ГК', 'GR', 'GAMMA', 'GR_1', 'CGR', 'GAM', 'GR_ED'],
    'PS': ['PS', 'ПС', 'SP', 'SPC', 'SP_1', 'CPS', 'SP_ED'],
    'IK': ['IK', 'ИК', 'ILM', 'ILD', 'AIT', 'RT_IK'],
    'BK': ['BK', 'БК', 'LL3', 'LLS', 'LLD'],
    'GMZ': ['GMZ', 'ГМЗ', 'GZ', 'ГЗ'],
    'PMZ': ['PMZ', 'ПМЗ', 'PZ', 'ПЗ'],
    'NGK': ['NGK', 'НГК', 'NKT', 'НКТ', 'CNL', 'NPHI', 'TNPH'],
    'GGKP': ['GGKP', 'ГГКП', 'RHOB', 'RHOZ', 'DEN'],
    'DS': ['DS', 'ДС', 'CALI', 'CAL', 'DIAM']
}

def standardize_curve_name(raw_name):
    clean = raw_name.strip().upper()
    for std_name, aliases in CURVE_SYNONYMS.items():
        if clean in aliases:
            return std_name
        if any(clean.startswith(alias) or clean.endswith(alias) for alias in aliases):
            return std_name
    return clean

def resample_curve(depth_src, curve_src, depth_target, null_val=None, curve_name=""):
    c_arr = np.array(curve_src, dtype=float)
    d_arr = np.array(depth_src, dtype=float)
    
    valid_mask = np.isfinite(c_arr) & np.isfinite(d_arr)
    
    if null_val is not None:
        try:
            nv = float(null_val)
            valid_mask &= ~np.isclose(c_arr, nv, atol=1.0)
        except Exception:
            pass
            
    valid_mask &= (c_arr > -500.0) & (c_arr < 90000.0)
    
    c_name_up = curve_name.upper()
    if any(k in c_name_up for k in ['GK', 'ГК', 'GR', 'IK', 'ИК', 'BK', 'БК', 'GMZ', 'PMZ', 'RT']):
        valid_mask &= (c_arr >= 0.0)
        
    d_clean = d_arr[valid_mask]
    c_clean = c_arr[valid_mask]
    
    if len(d_clean) < 2:
        return np.full_like(depth_target, np.nan)
        
    if depth_target.min() > d_clean.max() or depth_target.max() < d_clean.min():
        return np.full_like(depth_target, np.nan)
        
    sort_idx = np.argsort(d_clean)
    d_clean = d_clean[sort_idx]
    c_clean = c_clean[sort_idx]
    
    d_clean, unique_idx = np.unique(d_clean, return_index=True)
    c_clean = c_clean[unique_idx]
    
    f = interp1d(d_clean, c_clean, kind='linear', bounds_error=False, fill_value=np.nan)
    return f(depth_target)

def extract_well_info(uploaded_file, default_idx=0):
    raw_bytes = uploaded_file.getvalue()
    for enc in ['utf-8', 'windows-1251', 'cp1251', 'latin-1']:
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
            
    las = lasio.read(io.StringIO(text))
    df = las.df()
    
    x, y = 0.0, 0.0
    for key in ['X', 'X_COORD', 'EASTING', 'XCOORD', 'X_LOC']:
        if key in las.well:
            try:
                x = float(str(las.well[key].value).replace(',', '.'))
                break
            except Exception: pass
            
    for key in ['Y', 'Y_COORD', 'NORTHING', 'YCOORD', 'Y_LOC']:
        if key in las.well:
            try:
                y = float(str(las.well[key].value).replace(',', '.'))
                break
            except Exception: pass
            
    if x == 0.0 and y == 0.0:
        x = float((default_idx + 1) * 450.0)
        y = float(200.0 + (default_idx % 2) * 350.0)
        
    null_val = None
    if 'NULL' in las.well:
        try:
            null_val = float(str(las.well['NULL'].value).replace(',', '.'))
        except Exception:
            pass
    
    curve_map = {}
    for col in df.columns:
        if col.strip().upper() in ['DEPT', 'DEPTH', 'ГЛУБИНА']:
            continue
        std_name = standardize_curve_name(col)
        if std_name not in curve_map:
            curve_map[std_name] = col
            
    return {
        'name': uploaded_file.name,
        'las': las,
        'df': df,
        'x': x,
        'y': y,
        'null_val': null_val,
        'curve_map': curve_map,
        'raw_curves': [c for c in df.columns if c.strip().upper() not in ['DEPT', 'DEPTH', 'ГЛУБИНА']]
    }

def compute_idw_weights(wells, target_x, target_y, smoothing=60.0):
    distances = []
    for w in wells:
        d = np.sqrt((w['x'] - target_x)**2 + (w['y'] - target_y)**2)
        distances.append(np.sqrt(d**2 + smoothing**2))
    inv_d = [1.0 / (d**1.5) for d in distances]
    total_inv = sum(inv_d)
    return [w / total_inv for w in inv_d]

# ==========================================
# 3. НАСТРОЙКА СТРАНИЦЫ
# ==========================================
# st.set_page_config(
#     page_title="SynthGeoGen",
#     layout="wide"
# )

def run_geo_app():

    if st.sidebar.button("Вернуться на главную", use_container_width=True, key="btn_back_seis"):
        st.session_state.current_page = 'home'
        st.rerun()

        
    st.markdown(
        """
        <div style="margin-bottom: 15px;">
            <h1 style="color: #1E3A8A; margin: 0; font-size: 2.2rem; font-weight: 700;">SynthGeoGen</h1>
            <p style="color: #475569; margin-top: 4px; font-size: 0.95rem;">
                Детерминированная интерполяция по координатам устьев (IDW) и стохастическое CVAE-моделирование комплекса ГИС
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "points_list" not in st.session_state:
        st.session_state.points_list = []

    if "last_registered_click" not in st.session_state:
        st.session_state.last_registered_click = None

    if "map_key_version" not in st.session_state:
        st.session_state.map_key_version = 0

    SYNTH_COLORS = [
        '#DC2626', '#EA580C', '#D97706', '#059669', '#0D9488', 
        '#0284C7', '#2563EB', '#4F46E5', '#7C3AED', '#9333EA', 
        '#C026D3', '#DB2777', '#E11D48', '#854D0E', '#4D7C0F'
    ]

    # Сайдбар: параметры и файлы
    st.sidebar.header("1. Опорные скважины")
    uploaded_files = st.sidebar.file_uploader("Загрузите файлы LAS (от 2 файлов)", type=['las'], accept_multiple_files=True)

    wells_data = []
    if uploaded_files:
        curr_hash = "_".join(sorted([f.name for f in uploaded_files]))
        if st.session_state.get("last_uploaded_hash") != curr_hash:
            st.session_state.last_uploaded_hash = curr_hash
            st.session_state.map_key_version += 1

        for idx_f, f in enumerate(uploaded_files):
            info = extract_well_info(f, default_idx=idx_f)
            wells_data.append(info)

        with st.sidebar.expander("Опорные скважины (активность)", expanded=False):
            for w in wells_data:
                w_state_key = f"las_active_{w['name']}"
                if w_state_key not in st.session_state:
                    st.session_state[w_state_key] = True
                w['active'] = st.checkbox(
                    f"{w['name']} ({int(w['x'])} м, {int(w['y'])} м)", 
                    value=st.session_state[w_state_key], 
                    key=w_state_key
                )

        with st.sidebar.expander("Сводка распознанных методов", expanded=False):
            summary_rows = []
            for w in wells_data:
                mapped_str = ", ".join([f"{k} ({v})" if k != v else k for k, v in w['curve_map'].items()])
                summary_rows.append({
                    "Скважина": w['name'],
                    "Распознано": mapped_str
                })
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    if len(wells_data) < 2:
        st.info("Необходимо загрузить минимум 2 LAS-файла с каротажными данными для выполнения расчета.")
        st.stop()

    active_wells = [w for w in wells_data if w.get('active', True)]
    if len(active_wells) < 1:
        st.warning("Включите хотя бы одну опорную скважину в боковой панели.")
        st.stop()

    all_available_methods = set()
    for w in active_wells:
        all_available_methods.update(w['curve_map'].keys())
    all_available_methods = sorted(list(all_available_methods))

    # Определение пересечений глубин
    mins_d = [float(w['df'].index.min()) for w in active_wells]
    maxs_d = [float(w['df'].index.max()) for w in active_wells]
    global_min_d = float(min(mins_d))
    global_max_d = float(max(maxs_d))

    overlap_min = float(max(mins_d))
    overlap_max = float(min(maxs_d))

    if overlap_min < overlap_max:
        def_top = float(round(overlap_min + 5.0, 1))
        def_bot = float(round(min(def_top + 35.0, overlap_max), 1))
    else:
        def_top = float(round(global_min_d, 1))
        def_bot = float(round(min(def_top + 35.0, global_max_d), 1))

    st.sidebar.markdown("---")
    st.sidebar.header("2. Интервал пласта")
    top_depth = st.sidebar.number_input("Кровля пласта (м)", value=def_top, min_value=global_min_d, max_value=global_max_d, step=1.0)
    bot_depth = st.sidebar.number_input("Подошва пласта (м)", value=def_bot, min_value=global_min_d, max_value=global_max_d, step=1.0)

    if top_depth >= bot_depth:
        st.sidebar.error("Кровля пласта должна быть меньше подошвы.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.header("3. Параметры стохастики CVAE")
    cvae_influence = st.sidebar.slider("Степень вариативности CVAE", 0.0, 1.0, 0.35, 0.05)
    sigma_val = st.sidebar.slider("Дисперсия латентного пространства (Sigma)", 0.1, 2.5, 0.8, 0.05)
    base_seed = st.sidebar.number_input("Базовый Seed", value=101, step=1)

    # Координаты карты и масштабирование
    all_xs = [w['x'] for w in wells_data]
    all_ys = [w['y'] for w in wells_data]
    min_x, max_x = float(min(all_xs)), float(max(all_xs))
    min_y, max_y = float(min(all_ys)), float(max(all_ys))

    SCALE = 0.001

    def m_to_latlon(xm, ym):
        return ym * SCALE, xm * SCALE

    def latlon_to_m(lat, lon):
        return float(round(lon / SCALE, 1)), float(round(lat / SCALE, 1))

    # ==========================================
    # 4. РАСЧЕТ ДАННЫХ И ИНФЕРЕНС CVAE
    # ==========================================
    cvae_net, cvae_meta = load_cvae("cvae_weights.pt")
    target_points = cvae_meta['input_dim'] if cvae_meta else 200
    depth_axis = np.linspace(top_depth, bot_depth, target_points)

    resampled_active = {m: {} for m in all_available_methods}
    for w in active_wells:
        df_w = w['df']
        d_raw = df_w.index.values
        for std_m, raw_col in w['curve_map'].items():
            c_raw = df_w[raw_col].values
            resampled_active[std_m][w['name']] = resample_curve(
                d_raw, c_raw, depth_axis, null_val=w.get('null_val'), curve_name=std_m
            )

    active_points_with_idx = [(idx, p) for idx, p in enumerate(st.session_state.points_list) if p.get('active', True)]
    generated_wells = []

    for idx_pt, (orig_idx, pt) in enumerate(active_points_with_idx):
        weights = compute_idw_weights(active_wells, pt['x'], pt['y'])
        rng = np.random.default_rng(int(base_seed) + idx_pt * 29)
        
        if cvae_net is not None:
            z_sample = rng.normal(0.0, sigma_val, size=(1, 4))
            with torch.no_grad():
                out_cvae = cvae_net.decode(torch.FloatTensor(z_sample)).numpy()
                
            gk_eval = (out_cvae[0, 0, :] + 1.0) / 2.0 * (cvae_meta['gk_max'] - cvae_meta['gk_min']) + cvae_meta['gk_min']
            ps_eval = (out_cvae[0, 1, :] + 1.0) / 2.0 * (cvae_meta['ps_max'] - cvae_meta['ps_min']) + cvae_meta['ps_min']
            
            gk_noise = gk_eval - np.mean(gk_eval)
            ps_noise = ps_eval - np.mean(ps_eval)
            
            gk_std = np.std(gk_noise) if np.std(gk_noise) > 1e-5 else 1.0
            cvae_pattern = gk_noise / gk_std
        else:
            ps_noise = rng.normal(0.0, 1.0, size=target_points)
            cvae_pattern = rng.normal(0.0, 1.0, size=target_points)

        max_w = max(weights)
        spatial_uncertainty = np.clip(1.2 * (1.0 - max_w), 0.2, 1.0)
        alpha = cvae_influence * spatial_uncertainty

        synth_curves = {}
        for m_name in all_available_methods:
            present_wells = [w for w in active_wells if w['name'] in resampled_active[m_name]]
            valid_wells, valid_curves = [], []
            for pw in present_wells:
                vals = resampled_active[m_name][pw['name']]
                if np.sum(~np.isnan(vals)) >= (0.3 * len(vals)):
                    valid_wells.append(pw)
                    m_mean = np.nanmean(vals)
                    valid_curves.append(np.where(np.isnan(vals), m_mean, vals))
                    
            if not valid_wells:
                continue
                
            sub_weights = compute_idw_weights(valid_wells, pt['x'], pt['y'])
            det_trend = sum(valid_curves[i] * sub_weights[i] for i in range(len(valid_wells)))
            orig_std = np.nanstd(valid_curves)
            if np.isnan(orig_std) or orig_std < 1e-3:
                orig_std = 1.0

            method_hash = sum(ord(c) for c in m_name)
            rng_m = np.random.default_rng(int(base_seed) + idx_pt * 43 + method_hash)
            m_random_pattern = rng_m.normal(0.0, 1.0, size=target_points)

            if m_name == 'PS':
                noise_comp = gaussian_filter1d(ps_noise, sigma=1.0) * alpha
            elif m_name == 'GK':
                noise_comp = gaussian_filter1d(cvae_pattern, sigma=0.4) * (orig_std * alpha * 0.45)
            else:
                noise_comp = gaussian_filter1d(m_random_pattern, sigma=0.6) * (orig_std * alpha * 0.25)

            synth_curves[m_name] = det_trend + noise_comp

        generated_wells.append({
            'id': pt['id'], 'x': pt['x'], 'y': pt['y'],
            'color': SYNTH_COLORS[orig_idx % len(SYNTH_COLORS)],
            'weights': weights, 'curves': synth_curves
        })

    # ==========================================
    # 5. СТРУКТУРИРОВАННЫЙ ИНТЕРФЕЙС (ВКЛАДКИ)
    # ==========================================
    tab_map, tab_logs, tab_export = st.tabs([
        "Карта и расстановка точек", 
        "Каротажный планшет", 
        "Сводка и экспорт LAS"
    ])

    # ------------------------------------------
    # ВКЛАДКА 1: КАРТА
    # ------------------------------------------
    with tab_map:
        col_hdr, col_btn = st.columns([3, 1])
        with col_hdr:
            st.caption("Клик по карте — установка новой скважины. Перемещение — зажатая ЛКМ. Масштаб — колесико мыши.")
        with col_btn:
            if st.button("Сфокусировать на скважинах", use_container_width=True):
                st.session_state.map_key_version += 1
                st.rerun()

        span_x = max(max_x - min_x, 600.0)
        span_y = max(max_y - min_y, 600.0)
        pad = max(span_x, span_y) * 0.25

        b_min_x, b_max_x = min_x - pad, max_x + pad
        b_min_y, b_max_y = min_y - pad, max_y + pad
        focus_bounds = [m_to_latlon(b_min_x, b_min_y), m_to_latlon(b_max_x, b_max_y)]

        m = folium.Map(
            tiles=None,
            attribution_control=False,
            zoom_control=True
        )
        m.fit_bounds(focus_bounds)

        field_pad = max(span_x, span_y) * 1.5 + 1500.0
        field_min_x = np.floor((min_x - field_pad) / 500.0) * 500.0
        field_max_x = np.ceil((max_x + field_pad) / 500.0) * 500.0
        field_min_y = np.floor((min_y - field_pad) / 500.0) * 500.0
        field_max_y = np.ceil((max_y + field_pad) / 500.0) * 500.0

        folium.Rectangle(
            bounds=[m_to_latlon(field_min_x, field_min_y), m_to_latlon(field_max_x, field_max_y)],
            color="#94A3B8",
            weight=1.5,
            fill=True,
            fill_color="#FAFAFA",
            fill_opacity=1.0
        ).add_to(m)

        grid_step = 500.0 if max(span_x, span_y) > 2000.0 else 250.0
        xs = np.arange(field_min_x, field_max_x + 1.0, grid_step)
        ys = np.arange(field_min_y, field_max_y + 1.0, grid_step)

        for gx in xs:
            folium.PolyLine(
                locations=[m_to_latlon(gx, field_min_y), m_to_latlon(gx, field_max_y)],
                color="#CBD5E1",
                weight=1.0,
                dash_array="4, 4"
            ).add_to(m)
            if (min_x - pad * 1.5) <= gx <= (max_x + pad * 1.5):
                folium.Marker(
                    location=m_to_latlon(gx, min_y - pad * 0.9),
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 9px; font-weight: 600; color: #64748B; background: rgba(255,255,255,0.85); padding: 1px 4px; border-radius: 3px; border: 1px solid #E2E8F0; white-space: nowrap; transform: translateX(-50%);">{int(gx)} м</div>'
                    )
                ).add_to(m)

        for gy in ys:
            folium.PolyLine(
                locations=[m_to_latlon(field_min_x, gy), m_to_latlon(field_max_x, gy)],
                color="#CBD5E1",
                weight=1.0,
                dash_array="4, 4"
            ).add_to(m)
            if (min_y - pad * 1.5) <= gy <= (max_y + pad * 1.5):
                folium.Marker(
                    location=m_to_latlon(min_x - pad * 0.9, gy),
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 9px; font-weight: 600; color: #64748B; background: rgba(255,255,255,0.85); padding: 1px 4px; border-radius: 3px; border: 1px solid #E2E8F0; white-space: nowrap; transform: translate(-100%, -50%);">{int(gy)} м</div>'
                    )
                ).add_to(m)

        well_colors = ['#1E293B', '#2563EB', '#059669', '#D97706', '#7C3AED', '#DB2777', '#0284C7', '#4F46E5']
        for idx, w in enumerate(wells_data):
            c = well_colors[idx % len(well_colors)]
            w_lat, w_lon = m_to_latlon(w['x'], w['y'])
            if w.get('active', True):
                folium.CircleMarker(
                    location=[w_lat, w_lon],
                    radius=9,
                    color="#0F172A",
                    weight=2,
                    fill=True,
                    fill_color=c,
                    fill_opacity=1.0,
                    tooltip=f"{w['name']} (X={w['x']:.0f} м, Y={w['y']:.0f} м)"
                ).add_to(m)
                folium.Marker(
                    location=[w_lat, w_lon],
                    icon=folium.DivIcon(html=f'<div style="font-size: 11px; font-weight: bold; color: #0F172A; margin-top: -24px; margin-left: -14px; white-space: nowrap;">{w["name"]}</div>')
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=[w_lat, w_lon],
                    radius=6,
                    color="#94A3B8",
                    weight=1,
                    fill=True,
                    fill_color="#E2E8F0",
                    fill_opacity=0.6
                ).add_to(m)

        for idx_s, s in enumerate(st.session_state.points_list):
            pt_color = SYNTH_COLORS[idx_s % len(SYNTH_COLORS)]
            s_lat, s_lon = m_to_latlon(s['x'], s['y'])
            if s.get('active', True):
                folium.Marker(
                    location=[s_lat, s_lon],
                    icon=folium.DivIcon(html=f'''
                        <div style="width:12px; height:12px; background:{pt_color}; border:1.5px solid #0F172A; transform:rotate(45deg);"></div>
                        <div style="font-size:9.5px; font-weight:bold; color:{pt_color}; margin-top:4px; margin-left:-10px; white-space: nowrap;">{s['id']}</div>
                    '''),
                    tooltip=f"{s['id']} (X={s['x']:.1f} м, Y={s['y']:.1f} м)"
                ).add_to(m)

        map_key = f"folium_map_v_{st.session_state.map_key_version}"
        map_out = st_folium(m, width=None, height=480, key=map_key)

        if map_out and map_out.get("last_clicked"):
            click_coord = map_out["last_clicked"]
            if click_coord != st.session_state.last_registered_click:
                st.session_state.last_registered_click = click_coord
                cx, cy = latlon_to_m(click_coord["lat"], click_coord["lng"])
                new_id = f"SYNTH_{len(st.session_state.points_list) + 1}"
                st.session_state.points_list.append({'id': new_id, 'x': cx, 'y': cy, 'active': True})
                st.rerun()

        c_tools1, c_tools2 = st.columns([1.5, 1.5])
        with c_tools1:
            step_val = st.number_input("Шаг авто-сетки (м)", min_value=50.0, max_value=800.0, value=300.0, step=50.0)
            if st.button("Построить регулярную сетку", use_container_width=True):
                st.session_state.points_list = []
                for cnt, (gx_v, gy_v) in enumerate([(x, y) for x in np.arange(min_x, max_x + 1.0, step_val) for y in np.arange(min_y, max_y + 1.0, step_val)], 1):
                    st.session_state.points_list.append({'id': f"GRID_{cnt}", 'x': round(gx_v, 1), 'y': round(gy_v, 1), 'active': True})
                st.rerun()
        with c_tools2:
            st.write("")
            st.write("")
            if st.button("Очистить все точки", use_container_width=True):
                st.session_state.points_list = []
                st.session_state.last_registered_click = None
                st.rerun()

        if st.session_state.points_list:
            with st.expander(f"Редактирование координат скважин ({len(st.session_state.points_list)} шт.)", expanded=False):
                idx_to_remove = None
                p_cols = st.columns(min(3, len(st.session_state.points_list)))
                for idx_p, p_item in enumerate(st.session_state.points_list):
                    col_curr = p_cols[idx_p % len(p_cols)]
                    with col_curr:
                        st.markdown(f"**{p_item['id']}**")
                        p_item['active'] = st.checkbox("Активна", value=p_item.get('active', True), key=f"act_{p_item['id']}_{idx_p}")
                        p_item['x'] = st.number_input("X (м)", value=float(p_item['x']), step=1.0, key=f"ex_{p_item['id']}_{idx_p}")
                        p_item['y'] = st.number_input("Y (м)", value=float(p_item['y']), step=1.0, key=f"ey_{p_item['id']}_{idx_p}")
                        if st.button("Удалить", key=f"del_{p_item['id']}_{idx_p}"):
                            idx_to_remove = idx_p
                if idx_to_remove is not None:
                    st.session_state.points_list.pop(idx_to_remove)
                    st.rerun()

    # ------------------------------------------
    # ВКЛАДКА 2: КАРОТАЖНЫЙ ПЛАНШЕТ
    # ------------------------------------------
    with tab_logs:
        if generated_wells:
            c_top1, c_top2 = st.columns([1.2, 2.0])
            with c_top1:
                selected_id = st.selectbox(
                    "Скважина для анализа:", 
                    options=[g['id'] for g in generated_wells],
                    format_func=lambda x: f"{x} (X={next(g['x'] for g in generated_wells if g['id'] == x):.1f} м, Y={next(g['y'] for g in generated_wells if g['id'] == x):.1f} м)"
                )
            with c_top2:
                default_display = [m for m in ['GK', 'PS', 'IK', 'BK'] if m in all_available_methods] or all_available_methods[:min(4, len(all_available_methods))]
                display_curves = st.multiselect("Отображаемые треки методов:", options=all_available_methods, default=default_display)

            curr = next(g for g in generated_wells if g['id'] == selected_id)

            # Аккуратные плашки-бейджи с процентом влияния каждой скважины
            badge_items = []
            for idx, w in enumerate(active_wells):
                c_color = well_colors[idx % len(well_colors)]
                w_val = curr['weights'][idx]
                badge_items.append(
                    f'<span style="display:inline-flex; align-items:center; background:#FFFFFF; border:1px solid #E2E8F0; '
                    f'padding:4px 10px; border-radius:16px; margin:3px 6px 3px 0; font-size:12px; color:#1E293B;">'
                    f'<span style="width:8px; height:8px; background:{c_color}; border-radius:50%; margin-right:6px; display:inline-block;"></span>'
                    f'<strong>{w["name"]}</strong>:&nbsp;<span style="color:#2563EB; font-weight:600;">{w_val:.1%}</span>'
                    f'</span>'
                )

            st.markdown(
                f'<div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:8px 12px; border-radius:8px; margin-bottom:14px;">'
                f'<span style="font-size:12px; font-weight:600; color:#64748B; margin-right:8px;">Веса влияния IDW:</span>'
                f'{"".join(badge_items)}'
                f'</div>',
                unsafe_allow_html=True
            )

            if display_curves:
                plt.style.use('default')
                n_plots = len(display_curves)
                
                fig_w = max(9.0, 2.7 * n_plots)
                fig_logs, axes = plt.subplots(1, n_plots, figsize=(fig_w, 6.2), sharey=True)
                if n_plots == 1:
                    axes = [axes]
                fig_logs.patch.set_facecolor('#FFFFFF')

                for ax_idx, c_name in enumerate(display_curves):
                    ax = axes[ax_idx]
                    ax.set_facecolor('#FFFFFF')

                    for idx, w in enumerate(active_wells):
                        if w['name'] in resampled_active[c_name]:
                            curve_v = resampled_active[c_name][w['name']]
                            if not np.all(np.isnan(curve_v)):
                                ax.plot(curve_v, depth_axis, color=well_colors[idx % len(well_colors)], lw=1.1, alpha=0.4)

                    if c_name in curr['curves']:
                        ax.plot(curr['curves'][c_name], depth_axis, color=curr['color'], lw=2.4, label=f"Синтез {curr['id']}")

                    ax.set_title(c_name, fontsize=11, fontweight='bold', color='#0F172A', pad=8)
                    if ax_idx == 0:
                        ax.set_ylabel("Глубина, м", fontsize=10, color='#0F172A')
                    ax.tick_params(colors='#0F172A', labelsize=8.5)
                    ax.grid(True, linestyle='--', color='#E2E8F0', alpha=0.8)
                    ax.legend(loc='lower right', fontsize=8, facecolor='#FFFFFF', edgecolor='#CBD5E1')

                    for spine in ax.spines.values():
                        spine.set_color('#94A3B8')
                        spine.set_linewidth(0.8)

                axes[0].invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig_logs, use_container_width=True)
        else:
            st.info("Поставьте хотя бы одну синтетическую скважину на вкладке карты.")

    # ------------------------------------------
    # ВКЛАДКА 3: ЭКСПОРТ И СВОДКА
    # ------------------------------------------
    with tab_export:
        if generated_wells:
            st.subheader("Сводная ведомость сгенерированных скважин")
            synth_summary = []
            for g in generated_wells:
                synth_summary.append({
                    "Скважина": g['id'],
                    "X (м)": g['x'],
                    "Y (м)": g['y'],
                    "Методов": len(g['curves']),
                    "Список методов": ", ".join(list(g['curves'].keys()))
                })
            st.dataframe(pd.DataFrame(synth_summary), use_container_width=True, hide_index=True)

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for w_item in generated_wells:
                    las_obj = lasio.LASFile()
                    las_obj.well.WELL = w_item['id']
                    las_obj.well['X'] = lasio.HeaderItem('X', value=str(w_item['x']), descr='X Coordinate')
                    las_obj.well['Y'] = lasio.HeaderItem('Y', value=str(w_item['y']), descr='Y Coordinate')
                    las_obj.append_curve("DEPT", depth_axis, unit="M", descr="Depth")
                    for c_name, c_vals in w_item['curves'].items():
                        las_obj.append_curve(c_name, c_vals, descr=f"Synthetic {c_name} (IDW+CVAE)")
                    str_buf = io.StringIO()
                    las_obj.write(str_buf, version=2.0)
                    zip_file.writestr(f"{w_item['id']}_X{int(w_item['x'])}_Y{int(w_item['y'])}.las", str_buf.getvalue())

            st.download_button(
                label=f"Скачать ZIP-пакет со всеми LAS-файлами ({len(generated_wells)} скважин)",
                data=zip_buf.getvalue(),
                file_name="Synthetic_Wells_Dataset.zip",
                mime="application/zip",
                use_container_width=True
            )
        else:
            st.info("Нет данных для экспорта. Добавьте точки на карте.")