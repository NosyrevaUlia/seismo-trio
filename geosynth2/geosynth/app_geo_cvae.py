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
        ckpt = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
        net = GeologicalCVAE(input_dim=ckpt['input_dim'], latent_dim=4)
        net.load_state_dict(ckpt['model_state'], strict=False)
        net.eval()
        return net, ckpt
    except Exception:
        return None, None

# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def resample_curve(depth_src, curve_src, depth_target):
    # Очистка от фиктивных значений (null values)
    mask = (~np.isnan(curve_src)) & (curve_src > -900)
    d_clean = depth_src[mask]
    c_clean = curve_src[mask]
    
    if len(d_clean) < 2:
        return np.full_like(depth_target, np.nan)
        
    f = interp1d(d_clean, c_clean, kind='linear', bounds_error=False, fill_value="extrapolate")
    return f(depth_target)

def find_curve_column(df_columns, targets):
    for col in df_columns:
        c_clean = col.strip().upper()
        if c_clean in targets:
            return col
    for col in df_columns:
        c_clean = col.strip().upper()
        if any(t in c_clean for t in targets):
            return col
    return None

def extract_well_info(uploaded_file, default_idx=0):
    raw_bytes = uploaded_file.getvalue()
    text = None
    for enc in ['utf-8', 'windows-1251', 'cp1251', 'latin-1']:
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"Не удалось декодировать файл {uploaded_file.name} (неизвестная кодировка)")

    las = lasio.read(io.StringIO(text))
    df = las.df()
    if df is None or len(df) == 0:
        raise ValueError(f"Файл {uploaded_file.name} не содержит каротажных данных")
    if not isinstance(df.index, pd.RangeIndex) and not pd.api.types.is_numeric_dtype(df.index):
        df.index = pd.to_numeric(df.index, errors='coerce')
        df = df[pd.notna(df.index)]
    
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
    
    ps_targets = ['PS', 'SP', 'SPC', 'ПС', 'SP_1', 'CPS', 'SP_ED']
    gk_targets = ['GK', 'GR', 'GAMMA', 'ГК', 'GR_1', 'CGR', 'GAM', 'GR_ED']
    
    ps_col = find_curve_column(df.columns, ps_targets)
    gk_col = find_curve_column(df.columns, gk_targets)
            
    return {
        'name': uploaded_file.name,
        'las': las,
        'df': df,
        'x': x,
        'y': y,
        'ps_col': ps_col,
        'gk_col': gk_col,
        'all_curves': list(df.columns)
    }

def compute_idw_weights(wells, target_x, target_y):
    distances = []
    for w in wells:
        d = np.sqrt((w['x'] - target_x)**2 + (w['y'] - target_y)**2)
        distances.append(max(d, 0.001))
    inv_d = [1.0 / (d**1.5) for d in distances]
    total_inv = sum(inv_d)
    return [w / total_inv for w in inv_d]

# ==========================================
# 3. НАСТРОЙКА СТРАНИЦЫ
# ==========================================
st.set_page_config(
    page_title="SynthGeoGen",
    layout="wide"
)

st.markdown(
    """
    <div style="margin-bottom: 20px;">
        <h1 style="color: #1E3A8A; margin: 0; font-size: 2.3rem; font-weight: 700;">SynthGeoGen</h1>
        <p style="color: #475569; margin-top: 4px; font-size: 0.95rem;">
            Детерминированная интерполяция по координатам устьев (IDW) и стохастическое CVAE-моделирование комплекса ГК и ПС
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

if "points_list" not in st.session_state:
    st.session_state.points_list = []

if "last_registered_click" not in st.session_state:
    st.session_state.last_registered_click = None

if "map_center" not in st.session_state:
    st.session_state.map_center = None

if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 11

SYNTH_COLORS = [
    '#DC2626', '#EA580C', '#D97706', '#059669', '#0D9488', 
    '#0284C7', '#2563EB', '#4F46E5', '#7C3AED', '#9333EA', 
    '#C026D3', '#DB2777', '#E11D48', '#854D0E', '#4D7C0F'
]

# Сайдбар: загрузка
st.sidebar.header("1. Опорные скважины")
uploaded_files = st.sidebar.file_uploader("Загрузите файлы LAS (от 2 файлов)", type=['las'], accept_multiple_files=True)

wells_data = []
if uploaded_files:
    for idx_f, f in enumerate(uploaded_files):
        try:
            info = extract_well_info(f, default_idx=idx_f)
            wells_data.append(info)
        except Exception as e:
            st.sidebar.warning(f"⚠️ Не удалось загрузить {f.name}: {str(e)}")

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

    with st.sidebar.expander("Сводка методов каротажа", expanded=False):
        summary_rows = []
        for w in wells_data:
            summary_rows.append({
                "Скважина": w['name'],
                "ПС": w['ps_col'] if w['ps_col'] else "—",
                "ГК": w['gk_col'] if w['gk_col'] else "—",
                "Кривых": len(w['all_curves'])
            })
        st.dataframe(
            pd.DataFrame(summary_rows),
            hide_index=True,
            use_container_width=True
        )

if len(wells_data) < 2:
    st.info("Необходимо загрузить минимум 2 LAS-файла с каротажными данными для выполнения расчета.")
    st.stop()

active_wells = [w for w in wells_data if w.get('active', True)]
if len(active_wells) < 1:
    st.warning("Включите хотя бы одну опорную скважину в боковой панели.")
    st.stop()

# Интервал пласта
d_index = wells_data[0]['df'].index
global_min_d, global_max_d = float(d_index.min()), float(d_index.max())
def_top = max(global_min_d, 2100.0) if global_max_d > 2100.0 else global_min_d
def_bot = min(def_top + 35.0, global_max_d)

st.sidebar.markdown("---")
st.sidebar.header("2. Интервал пласта")
top_depth = st.sidebar.number_input("Кровля пласта (м)", value=def_top, min_value=global_min_d, max_value=global_max_d, step=1.0)
bot_depth = st.sidebar.number_input("Подошва пласта (м)", value=def_bot, min_value=global_min_d, max_value=global_max_d, step=1.0)

if top_depth >= bot_depth:
    st.sidebar.error("Кровля пласта должна быть меньше подошвы.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("3. Параметры стохастики CVAE")
cvae_influence = st.sidebar.slider("Степень вариативности CVAE", 0.05, 1.0, 0.35, 0.05)
sigma_val = st.sidebar.slider("Дисперсия латентного пространства (Sigma)", 0.1, 2.0, 0.8, 0.05)
base_seed = st.sidebar.number_input("Базовый Seed", value=101, step=1)

# Координаты центра и полигона
all_xs = [w['x'] for w in wells_data]
all_ys = [w['y'] for w in wells_data]
min_x, max_x = float(min(all_xs)), float(max(all_xs))
min_y, max_y = float(min(all_ys)), float(max(all_ys))
default_center_x = (min_x + max_x) / 2.0
default_center_y = (min_y + max_y) / 2.0

SCALE = 0.001

def m_to_latlon(xm, ym):
    return ym * SCALE, xm * SCALE

def latlon_to_m(lat, lon):
    return float(round(lon / SCALE, 1)), float(round(lat / SCALE, 1))

if st.session_state.map_center is None:
    st.session_state.map_center = list(m_to_latlon(default_center_x, default_center_y))

# ==========================================
# 4. ИНТЕРАКТИВНАЯ КАРТА
# ==========================================
col_left, col_right = st.columns([1.1, 1.4])

with col_left:
    st.subheader("Схема расположения скважин")
    st.caption("Зажмите ЛКМ для перемещения. Колесико мыши — масштаб. Клик по сетке — установка скважины.")

    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        tiles=None,
        attribution_control=False,
        zoom_control=True
    )

    big_pad = 5000.0
    p_min_x, p_max_x = min_x - big_pad, max_x + big_pad
    p_min_y, p_max_y = min_y - big_pad, max_y + big_pad

    folium.Rectangle(
        bounds=[m_to_latlon(p_min_x, p_min_y), m_to_latlon(p_max_x, p_max_y)],
        color="#CBD5E1",
        weight=1.5,
        fill=True,
        fill_color="#FFFFFF",
        fill_opacity=1.0
    ).add_to(m)

    grid_step = 200.0
    grid_xs = np.arange(np.floor(p_min_x / grid_step) * grid_step, p_max_x + 1.0, grid_step)
    grid_ys = np.arange(np.floor(p_min_y / grid_step) * grid_step, p_max_y + 1.0, grid_step)

    for gx in grid_xs:
        folium.PolyLine(
            locations=[m_to_latlon(gx, p_min_y), m_to_latlon(gx, p_max_y)],
            color="#E2E8F0",
            weight=1.0,
            dash_array="3, 4"
        ).add_to(m)
        folium.Marker(
            location=m_to_latlon(gx, min_y - 80),
            icon=folium.DivIcon(html=f'<div style="font-size: 8.5px; color: #94A3B8; white-space: nowrap;">{int(gx)} м</div>')
        ).add_to(m)

    for gy in grid_ys:
        folium.PolyLine(
            locations=[m_to_latlon(p_min_x, gy), m_to_latlon(p_max_x, gy)],
            color="#E2E8F0",
            weight=1.0,
            dash_array="3, 4"
        ).add_to(m)
        folium.Marker(
            location=m_to_latlon(min_x - 120, gy),
            icon=folium.DivIcon(html=f'<div style="font-size: 8.5px; color: #94A3B8; white-space: nowrap;">{int(gy)} м</div>')
        ).add_to(m)

    well_colors = ['#1E293B', '#2563EB', '#059669', '#D97706', '#7C3AED', '#DB2777']
    for idx, w in enumerate(wells_data):
        c = well_colors[idx % len(well_colors)]
        w_lat, w_lon = m_to_latlon(w['x'], w['y'])
        if w.get('active', True):
            folium.CircleMarker(
                location=[w_lat, w_lon],
                radius=8,
                color="#0F172A",
                weight=2,
                fill=True,
                fill_color=c,
                fill_opacity=1.0,
                tooltip=f"{w['name']} (X={w['x']:.0f} м, Y={w['y']:.0f} м)"
            ).add_to(m)
            folium.Marker(
                location=[w_lat, w_lon],
                icon=folium.DivIcon(html=f'<div style="font-size: 10.5px; font-weight: bold; color: #0F172A; margin-top: -24px; margin-left: -14px; white-space: nowrap;">{w["name"]}</div>')
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[w_lat, w_lon],
                radius=6,
                color="#94A3B8",
                weight=1,
                fill=True,
                fill_color="#E2E8F0",
                fill_opacity=0.8
            ).add_to(m)

    for idx_s, s in enumerate(st.session_state.points_list):
        pt_color = SYNTH_COLORS[idx_s % len(SYNTH_COLORS)]
        s_lat, s_lon = m_to_latlon(s['x'], s['y'])
        if s.get('active', True):
            folium.Marker(
                location=[s_lat, s_lon],
                icon=folium.DivIcon(html=f'''
                    <div style="width:11px; height:11px; background:{pt_color}; border:1.5px solid #0F172A; transform:rotate(45deg);"></div>
                    <div style="font-size:8.5px; font-weight:bold; color:{pt_color}; margin-top:4px; margin-left:-10px; white-space: nowrap;">{s['id']}</div>
                '''),
                tooltip=f"{s['id']} (X={s['x']:.1f} м, Y={s['y']:.1f} м)"
            ).add_to(m)
        else:
            folium.Marker(
                location=[s_lat, s_lon],
                icon=folium.DivIcon(html='<div style="width:9px; height:9px; background:#E2E8F0; border:1px solid #94A3B8; transform:rotate(45deg);"></div>')
            ).add_to(m)

    map_out = st_folium(m, width=540, height=380, key="folium_canvas_map")

    if map_out:
        if map_out.get("center"):
            st.session_state.map_center = [map_out["center"]["lat"], map_out["center"]["lng"]]
        if map_out.get("zoom"):
            st.session_state.map_zoom = map_out["zoom"]

        if map_out.get("last_clicked"):
            click_coord = map_out["last_clicked"]
            if click_coord != st.session_state.last_registered_click:
                st.session_state.last_registered_click = click_coord
                cx, cy = latlon_to_m(click_coord["lat"], click_coord["lng"])

                new_id = f"SYNTH_{len(st.session_state.points_list) + 1}"
                st.session_state.points_list.append({'id': new_id, 'x': cx, 'y': cy, 'active': True})
                st.rerun()

    ca1, ca2 = st.columns(2)
    with ca1:
        step_val = st.number_input("Шаг авто-сетки (м)", min_value=50.0, max_value=500.0, value=200.0, step=25.0)
        if st.button("Построить регулярную сетку", use_container_width=True):
            st.session_state.points_list = []
            x_vals = np.arange(min_x, max_x + 1.0, step_val)
            y_vals = np.arange(min_y, max_y + 1.0, step_val)
            cnt = 1
            for gx_v in x_vals:
                for gy_v in y_vals:
                    st.session_state.points_list.append({
                        'id': f"GRID_{cnt}",
                        'x': float(round(gx_v, 1)),
                        'y': float(round(gy_v, 1)),
                        'active': True
                    })
                    cnt += 1
            st.rerun()
    with ca2:
        st.write("")
        st.write("")
        if st.button("Очистить все точки", use_container_width=True):
            st.session_state.points_list = []
            st.session_state.last_registered_click = None
            st.rerun()

    if st.session_state.points_list:
        st.markdown(f"**Синтетические скважины ({len(st.session_state.points_list)} шт.):**")
        
        idx_to_remove = None
        for idx_p, p_item in enumerate(st.session_state.points_list):
            expander_title = f"{p_item['id']} | X: {int(p_item['x'])} м, Y: {int(p_item['y'])} м"
            
            with st.expander(expander_title, expanded=False):
                col_act, col_del = st.columns([1.5, 1.0])
                with col_act:
                    p_item['active'] = st.checkbox(
                        "Участвует в расчете", 
                        value=p_item.get('active', True),
                        key=f"active_{p_item['id']}_{idx_p}"
                    )
                with col_del:
                    if st.button("Удалить", key=f"del_{p_item['id']}_{idx_p}", use_container_width=True):
                        idx_to_remove = idx_p
                
                col_coord_x, col_coord_y = st.columns(2)
                with col_coord_x:
                    p_item['x'] = st.number_input(
                        "Координата X (м)", 
                        value=float(p_item['x']), 
                        step=1.0, 
                        key=f"edit_x_{p_item['id']}_{idx_p}"
                    )
                with col_coord_y:
                    p_item['y'] = st.number_input(
                        "Координата Y (м)", 
                        value=float(p_item['y']), 
                        step=1.0, 
                        key=f"edit_y_{p_item['id']}_{idx_p}"
                    )

        if idx_to_remove is not None:
            st.session_state.points_list.pop(idx_to_remove)
            st.rerun()

# ==========================================
# 5. ВЫЧИСЛЕНИЯ И ИНФЕРЕНС CVAE
# ==========================================
cvae_net, cvae_meta = load_cvae("cvae_weights.pt")
target_points = int(cvae_meta['input_dim']) if cvae_meta and 'input_dim' in cvae_meta else 200

# Создаем единую шкалу глубин строго от кровли до подошвы
depth_axis = np.linspace(top_depth, bot_depth, target_points)

# Интерполируем исходные кривые опорных скважин строго на depth_axis
resampled_ps = []
resampled_gk = []

for w in active_wells:
    df_w = w['df']
    d_raw = df_w.index.values
    
    ps_raw = df_w[w['ps_col']].values if w['ps_col'] and w['ps_col'] in df_w.columns else np.full_like(d_raw, 50.0)
    gk_raw = df_w[w['gk_col']].values if w['gk_col'] and w['gk_col'] in df_w.columns else np.full_like(d_raw, 6.0)
    
    ps_grid = resample_curve(d_raw, ps_raw, depth_axis)
    gk_grid = resample_curve(d_raw, gk_raw, depth_axis)
    
    resampled_ps.append(ps_grid)
    resampled_gk.append(gk_grid)

active_points_with_idx = [(idx, p) for idx, p in enumerate(st.session_state.points_list) if p.get('active', True)]
generated_wells = []

for idx_pt, (orig_idx, pt) in enumerate(active_points_with_idx):
    weights = compute_idw_weights(active_wells, pt['x'], pt['y'])
    
    # Расчет пространственного тренда IDW (сохраняет 100% резкости границ)
    det_trend_ps = np.zeros(target_points)
    det_trend_gk = np.zeros(target_points)
    for idx_w, w_coeff in enumerate(weights):
        det_trend_ps += resampled_ps[idx_w] * w_coeff
        det_trend_gk += resampled_gk[idx_w] * w_coeff
        
    if cvae_net is not None:
        rng = np.random.default_rng(int(base_seed) + idx_pt * 17)
        z_sample = rng.normal(0.0, sigma_val, size=(1, 4))
        
        with torch.no_grad():
            out_cvae = cvae_net.decode(torch.FloatTensor(z_sample)).numpy()
            
        # Канал 0: Гамма-каротаж ГК
        # Канал 1: Потенциал самополяризации ПС
        gk_min_v = float(cvae_meta.get('gk_min', 0.0))
        gk_max_v = float(cvae_meta.get('gk_max', 150.0))
        ps_min_v = float(cvae_meta.get('ps_min', -100.0))
        ps_max_v = float(cvae_meta.get('ps_max', 50.0))
        gk_eval = (out_cvae[0, 0, :] + 1.0) / 2.0 * (gk_max_v - gk_min_v) + gk_min_v
        ps_eval = (out_cvae[0, 1, :] + 1.0) / 2.0 * (ps_max_v - ps_min_v) + ps_min_v
        
        # Центрированные флуктуации (микроструктура CVAE)
        gk_noise = gk_eval - np.mean(gk_eval)
        ps_noise = ps_eval - np.mean(ps_eval)
        
        # Фильтрация ТОЛЬКО шума, чтобы не мылить пласты
        ps_noise = gaussian_filter1d(ps_noise, sigma=1.0)
        gk_noise = gaussian_filter1d(gk_noise, sigma=0.4)
        
        # Масштабирование шума: вблизи скважины (max_w -> 1) шум деликатный,
        # вдали от скважин (max_w -> 0.3) шум более выражен
        max_w = max(weights)
        spatial_uncertainty = np.clip(1.2 * (1.0 - max_w), 0.15, 1.0)
        
        alpha = cvae_influence * spatial_uncertainty
        final_ps = det_trend_ps + ps_noise * alpha
        final_gk = det_trend_gk + gk_noise * alpha
    else:
        final_ps, final_gk = det_trend_ps, det_trend_gk
        
    generated_wells.append({
        'id': pt['id'],
        'x': pt['x'],
        'y': pt['y'],
        'color': SYNTH_COLORS[orig_idx % len(SYNTH_COLORS)],
        'weights': weights,
        'ps': final_ps,
        'gk': final_gk
    })

# ==========================================
# 6. ВЫВОД ПЛАНШЕТА И ЭКСПОРТ
# ==========================================
with col_right:
    if generated_wells:
        selected_id = st.selectbox(
            "Просмотр каротажного планшета:", 
            options=[g['id'] for g in generated_wells],
            format_func=lambda x: f"{x} (X={next(g['x'] for g in generated_wells if g['id'] == x):.1f} м, Y={next(g['y'] for g in generated_wells if g['id'] == x):.1f} м)"
        )
        curr = next(g for g in generated_wells if g['id'] == selected_id)
        
        plt.style.use('default')
        fig_logs, axes = plt.subplots(1, 2, figsize=(7.5, 5.8), sharey=True)
        fig_logs.patch.set_facecolor('#FFFFFF')
        
        # Трек ПС
        axes[0].set_facecolor('#FFFFFF')
        for idx, (p_grid, w, weight) in enumerate(zip(resampled_ps, active_wells, curr['weights'])):
            c = well_colors[idx % len(well_colors)]
            axes[0].plot(p_grid, depth_axis, color=c, lw=1.2, alpha=0.55, label=f"{w['name']} (w={weight:.2f})")
        
        axes[0].plot(curr['ps'], depth_axis, color=curr['color'], lw=2.2, label=f"Синтез {curr['id']}")
        axes[0].set_title("Потенциал ПС, мВ", fontsize=11, fontweight='bold', color='#0F172A', pad=10)
        axes[0].set_xlabel("мВ", fontsize=9, color='#0F172A')
        axes[0].set_ylabel("Глубина, м", fontsize=9, color='#0F172A')
        axes[0].tick_params(colors='#0F172A')
        axes[0].invert_yaxis()
        axes[0].grid(True, linestyle='--', color='#CBD5E1', alpha=0.8)
        axes[0].legend(loc='upper right', fontsize=8, facecolor='#FFFFFF', edgecolor='#94A3B8')
        
        # Трек ГК
        axes[1].set_facecolor('#FFFFFF')
        for idx, (g_grid, w, weight) in enumerate(zip(resampled_gk, active_wells, curr['weights'])):
            c = well_colors[idx % len(well_colors)]
            axes[1].plot(g_grid, depth_axis, color=c, lw=1.2, alpha=0.55, label=f"{w['name']} (w={weight:.2f})")
        
        axes[1].plot(curr['gk'], depth_axis, color=curr['color'], lw=2.2, label=f"Синтез {curr['id']}")
        axes[1].set_title("Гамма-каротаж ГК, мкР/ч", fontsize=11, fontweight='bold', color='#0F172A', pad=10)
        axes[1].set_xlabel("мкР/ч", fontsize=9, color='#0F172A')
        axes[1].tick_params(colors='#0F172A')
        axes[1].grid(True, linestyle='--', color='#CBD5E1', alpha=0.8)
        axes[1].legend(loc='upper right', fontsize=8, facecolor='#FFFFFF', edgecolor='#94A3B8')
        
        for ax in axes:
            for spine in ax.spines.values():
                spine.set_color('#64748B')
                spine.set_linewidth(1.0)
        
        plt.tight_layout()
        st.pyplot(fig_logs)
    else:
        st.info("Поставьте скважину кликом на схеме слева для построения планшета.")

# Экспорт пакета LAS
if generated_wells:
    st.markdown("---")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for w_item in generated_wells:
            try:
                las_obj = lasio.LASFile()
                las_obj.well.WELL = w_item['id']
                try:
                    las_obj.well['X'] = lasio.HeaderItem('X', value=str(w_item['x']), descr='X Coordinate')
                    las_obj.well['Y'] = lasio.HeaderItem('Y', value=str(w_item['y']), descr='Y Coordinate')
                except Exception:
                    if not hasattr(las_obj.well, 'X'):
                        setattr(las_obj.well, 'X', lasio.HeaderItem(mnemonic='X', value=str(w_item['x']), descr='X Coordinate'))
                    if not hasattr(las_obj.well, 'Y'):
                        setattr(las_obj.well, 'Y', lasio.HeaderItem(mnemonic='Y', value=str(w_item['y']), descr='Y Coordinate'))
                las_obj.append_curve("DEPT", depth_axis, unit="M", descr="Depth")
                las_obj.append_curve("PS", w_item['ps'], unit="MV", descr="Synthetic PS (IDW+CVAE)")
                las_obj.append_curve("GK", w_item['gk'], unit="MKR/H", descr="Synthetic GK (IDW+CVAE)")

                str_buf = io.StringIO()
                try:
                    las_obj.write(str_buf, version=2.0)
                except TypeError:
                    las_obj.write(str_buf)
                zip_file.writestr(f"{w_item['id']}_X{int(w_item['x'])}_Y{int(w_item['y'])}.las", str_buf.getvalue())
            except Exception as exp_err:
                st.warning(f"Не удалось сформировать LAS для {w_item['id']}: {str(exp_err)}")
            
    st.download_button(
        label=f"Экспорт LAS-файлов в ZIP ({len(generated_wells)} скважин)",
        data=zip_buf.getvalue(),
        file_name="Synthetic_Wells_Dataset.zip",
        mime="application/zip",
        use_container_width=True
    )