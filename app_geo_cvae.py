import os
import io
import re
import json
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import lasio
from scipy.ndimage import gaussian_filter1d
from branca.element import MacroElement, Element
from jinja2 import Template
import plotly.graph_objects as go

# ==========================================
# 1. НЕЙРОСЕТЕВАЯ МОДЕЛЬ (CVAE) — ЗАГРУЗКА ВЕСОВ
# ==========================================
# PyTorch импортируется только при открытии модуля ГИС, чтобы стартовая страница открывалась быстро.
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cvae_weights.pt")
CVAE_POINTS = 200  # число отсчетов на выходе декодера


class _Decoder:
    def __init__(self, seq, input_dim):
        self.seq = seq
        self.input_dim = input_dim

    def decode(self, z):
        import torch
        with torch.no_grad():
            out = self.seq(torch.as_tensor(z, dtype=torch.float32))
        return out.view(-1, 2, self.input_dim).numpy()


@st.cache_resource(show_spinner=False)
def load_cvae(weights_path=WEIGHTS_PATH):
    try:
        import torch
        import torch.nn as nn
        ckpt = torch.load(weights_path, map_location=torch.device('cpu'))
        dim = ckpt['input_dim']
        seq = nn.Sequential(
            nn.Linear(4, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, dim * 2),
            nn.Tanh()
        )
        state = {k[len('decoder.'):]: v for k, v in ckpt['model_state'].items() if k.startswith('decoder.')}
        seq.load_state_dict(state)
        seq.eval()
        return _Decoder(seq, dim), ckpt
    except Exception:
        return None, None


# ==========================================
# 2. МЕТОДЫ ГИС, СЛОВАРЬ СИНОНИМОВ
# ==========================================
CURVE_SYNONYMS = {
    'GK': ['GK', 'ГК', 'GR', 'GAMMA', 'CGR', 'GAM', 'GR_ED'],
    'PS': ['PS', 'ПС', 'SP', 'SPC', 'CPS', 'SP_ED'],
    'IK': ['IK', 'ИК', 'ILM', 'ILD', 'AIT', 'RT_IK'],
    'BK': ['BK', 'БК', 'LL3', 'LLS', 'LLD'],
    'GMZ': ['GMZ', 'ГМЗ', 'GZ', 'ГЗ'],
    'PMZ': ['PMZ', 'ПМЗ', 'PZ', 'ПЗ'],
    'NGK': ['NGK', 'НГК', 'NKT', 'НКТ', 'CNL', 'NPHI', 'TNPH'],
    'GGKP': ['GGKP', 'ГГКП', 'ГГК-П', 'RHOB', 'RHOZ', 'DEN'],
    'DS': ['DS', 'ДС', 'CALI', 'CAL', 'DIAM'],
    'AK': ['AK', 'АК', 'DT', 'DTP', 'DTC', 'AC', 'SONIC'],
}

# русское обозначение, цвет кривой по умолчанию, логарифмическая шкала по умолчанию
METHOD_INFO = {
    'GK': ('ГК', '#15803D', False),
    'PS': ('ПС', '#1D4ED8', False),
    'IK': ('ИК', '#7C3AED', False),
    'BK': ('БК', '#DC2626', True),
    'GMZ': ('ГЗ', '#EA580C', True),
    'PMZ': ('ПЗ', '#0891B2', True),
    'NGK': ('НГК', '#1E3A8A', False),
    'GGKP': ('ГГК-П', '#B45309', False),
    'DS': ('ДС', '#475569', False),
    'AK': ('АК', '#DB2777', False),
}
NON_NEGATIVE = {'GK', 'IK', 'BK', 'GMZ', 'PMZ', 'NGK', 'GGKP', 'DS', 'AK'}
DEPTH_NAMES = {'DEPT', 'DEPTH', 'ГЛУБИНА', 'MD'}

STEP_OPTIONS = [0.1, 0.2, 0.4]
ALT_KEYS = ['RKB', 'KB', 'EKB', 'ALT', 'ALTITUDE', 'ELEV', 'EDF', 'EGL', 'GL']
MODE_INTERVAL = "По интервалу глубин"
MODE_LAYERS = "По пластам (по отбивкам)"
TOPS_COLS = ["Скважина", "Пласт", "Кровля, м", "Подошва, м", "Насыщение"]
SATURATIONS = ["вода", "нефть", "газ"]
IDW_POWER = 1.5
IDW_SMOOTHING = 60.0
NULL_OUT = -999.25

WELL_COLORS = ['#1E293B', '#2563EB', '#059669', '#D97706', '#7C3AED', '#DB2777', '#0284C7', '#4F46E5',
               '#65A30D', '#B91C1C']
SYNTH_COLORS = ['#DC2626', '#EA580C', '#D97706', '#059669', '#0D9488', '#0284C7', '#2563EB', '#4F46E5',
                '#7C3AED', '#9333EA', '#C026D3', '#DB2777', '#E11D48', '#854D0E', '#4D7C0F']


def method_label(m):
    return METHOD_INFO.get(m, (m, '#334155', False))[0]


def method_color(m):
    return METHOD_INFO.get(m, (m, '#334155', False))[1]


def method_log(m):
    return METHOD_INFO.get(m, (m, '#334155', False))[2]


def standardize_curve_name(raw_name):
    clean = raw_name.strip().upper()
    # GR_1, ГК1, PS.2 -> GR, ГК, PS
    stripped = re.sub(r'[\s_.\-:]*\d+$', '', clean)
    for candidate in (clean, stripped):
        for std_name, aliases in CURVE_SYNONYMS.items():
            if candidate in aliases:
                return std_name
    return clean


def fmt_num(v, digits=1):
    return f"{v:,.{digits}f}".replace(",", " ").replace(".", ",")


# ==========================================
# 3. ЧТЕНИЕ LAS И ПЕРЕНОС НА СЕТКУ ГЛУБИН
# ==========================================
def _to_float(raw):
    try:
        v = float(str(raw).strip().replace(',', '.'))
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _header_number(las, keys):
    # В LAS 1.2 lasio кладет значение в поле описания (так устроен формат 1.2),
    # поэтому проверяем оба поля.
    for key in keys:
        if key in las.well:
            item = las.well[key]
            for raw in (item.value, item.descr):
                v = _to_float(raw)
                if v is not None:
                    return v
    return None


def _header_text(las, key):
    if key not in las.well:
        return ""
    item = las.well[key]
    for raw in (item.value, item.descr):
        s = str(raw).strip()
        if s and len(s) <= 30:
            return s
    return ""


def clean_curve(values, null_val, std_name):
    c = np.array(values, dtype=float)
    bad = ~np.isfinite(c)
    if null_val is not None:
        bad |= np.isclose(c, null_val, atol=1.0)
    bad |= (c <= -500.0) | (c >= 90000.0)
    if std_name in NON_NEGATIVE:
        bad |= c < 0.0
    c[bad] = np.nan
    return c


@st.cache_data(show_spinner=False)
def parse_las_bytes(file_name, raw_bytes):
    text = None
    for enc in ['utf-8', 'windows-1251', 'latin-1']:
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    las = lasio.read(io.StringIO(text))
    df = las.df()
    df.index = pd.to_numeric(df.index, errors='coerce')
    df = df[np.isfinite(df.index.values)]
    df = df[~df.index.duplicated(keep='first')].sort_index()

    x = _header_number(las, ['X', 'X_COORD', 'EASTING', 'XCOORD', 'X_LOC', 'XWELL'])
    y = _header_number(las, ['Y', 'Y_COORD', 'NORTHING', 'YCOORD', 'Y_LOC', 'YWELL'])
    coords_found = x is not None and y is not None and not (x == 0.0 and y == 0.0)

    null_val = _header_number(las, ['NULL'])

    # альтитуда: стол ротора (RKB, KB...), при отсутствии — уровень земли (GL)
    alt, alt_key = None, None
    for key in ALT_KEYS:
        v = _header_number(las, [key])
        if v is not None:
            alt, alt_key = v, key
            break

    units = {c.mnemonic: (c.unit or "") for c in las.curves}
    curve_map, curves = {}, {}
    for col in df.columns:
        if col.strip().upper() in DEPTH_NAMES:
            continue
        std_name = standardize_curve_name(col)
        if std_name in curve_map:
            continue
        curve_map[std_name] = col
        curves[std_name] = clean_curve(df[col].values, null_val, std_name)

    depth = df.index.values.astype(float)
    step = float(np.median(np.diff(depth))) if len(depth) > 1 else 0.1

    stem = os.path.splitext(file_name)[0]
    return {
        'file': file_name,
        'stem': stem,
        'well_name': _header_text(las, 'WELL') or stem,
        'x_file': x, 'y_file': y, 'coords_found': coords_found,
        'alt_file': alt, 'alt_key': alt_key,
        'depth': depth,
        'step': step,
        'curve_map': curve_map,
        'curves': curves,
        'units': {std: units.get(raw, "") for std, raw in curve_map.items()},
    }


def resample_to_grid(depth_src, values, grid, src_step, max_gap):
    """Линейная интерполяция по фактическим глубинам. Пропуски длиннее max_gap
    и глубины за пределами записи остаются пустыми (NaN)."""
    mask = np.isfinite(values) & np.isfinite(depth_src)
    d, c = depth_src[mask], values[mask]
    if len(d) < 2:
        return np.full_like(grid, np.nan)
    order = np.argsort(d)
    d, c = d[order], c[order]
    out = np.interp(grid, d, c, left=np.nan, right=np.nan)
    gap_len = np.diff(d) - src_step
    for i in np.where(gap_len > max_gap + 1e-6)[0]:
        out[(grid > d[i]) & (grid < d[i + 1])] = np.nan
    return out


def nan_runs(grid, arr):
    """Интервалы глубин, где в массиве нет значений."""
    isnan = ~np.isfinite(arr)
    runs, start = [], None
    for j, flag in enumerate(isnan):
        if flag and start is None:
            start = j
        if not flag and start is not None:
            runs.append((grid[start], grid[j - 1]))
            start = None
    if start is not None:
        runs.append((grid[start], grid[-1]))
    return runs


def runs_to_text(runs, limit=4):
    parts = [f"{a:.1f}–{b:.1f}" if b > a else f"{a:.1f}" for a, b in runs[:limit]]
    if len(runs) > limit:
        parts.append(f"и еще {len(runs) - limit}")
    return ", ".join(parts)


def build_reference(wells, depth_axis, max_gap, max_gap_ak):
    methods = sorted({m for w in wells for m in w['curve_map']})
    ref = {'methods': methods, 'curves': {}, 'gaps': [], 'uncovered': {}, 'std': {}, 'units': {},
           'gap_limits': (max_gap, max_gap_ak)}
    for m in methods:
        idx, rows = [], []
        for k, w in enumerate(wells):
            if m not in w['curves']:
                continue
            gap_limit = max_gap_ak if m == 'AK' else max_gap
            res = resample_to_grid(w.get('zdepth', w['depth']), w['curves'][m], depth_axis, w['step'], gap_limit)
            idx.append(k)
            rows.append(res)
            runs = nan_runs(depth_axis, res)
            if runs:
                ref['gaps'].append({
                    "Скважина": w['name'], "Метод": method_label(m),
                    "Нет данных, м": round(sum(b - a for a, b in runs), 1),
                    "Интервалы, м": runs_to_text(runs),
                })
            if m not in ref['units'] and w['units'].get(m):
                ref['units'][m] = w['units'][m]
        mat = np.vstack(rows)
        ref['curves'][m] = (idx, mat)
        s = np.nanstd(mat) if np.any(np.isfinite(mat)) else np.nan
        ref['std'][m] = s if np.isfinite(s) and s > 1e-3 else 1.0
        covered = np.any(np.isfinite(mat), axis=0)
        runs = nan_runs(depth_axis, np.where(covered, 1.0, np.nan))
        if runs:
            ref['uncovered'][m] = runs
    ref['r_gk'] = correlation_with_gk(ref)
    return ref


def correlation_with_gk(ref):
    r = {}
    if 'GK' not in ref['curves']:
        return r
    gk_idx, gk_mat = ref['curves']['GK']
    for m, (idx, mat) in ref['curves'].items():
        if m in ('GK', 'PS'):
            continue
        a, b = [], []
        for row_k, k in enumerate(idx):
            if k in gk_idx:
                g = gk_mat[gk_idx.index(k)]
                v = mat[row_k]
                ok = np.isfinite(g) & np.isfinite(v)
                a.append(g[ok]); b.append(v[ok])
        if a:
            a, b = np.concatenate(a), np.concatenate(b)
            if len(a) > 10 and np.std(a) > 0 and np.std(b) > 0:
                r[m] = float(np.clip(np.corrcoef(a, b)[0, 1], -0.95, 0.95))
    return r


# ==========================================
# 3а. ОТБИВКИ ПЛАСТОВ
# ==========================================
def _norm_name(s):
    s = str(s).strip()
    if re.fullmatch(r"\d+\.0+", s):  # 9.0 -> 9 (номера скважин из Excel)
        s = s.split(".")[0]
    return s.lower()


def _norm_sat(s):
    s = str(s).strip().lower()
    if s.startswith("нефт") or s.startswith("oil"):
        return "нефть"
    if s.startswith("газ") or s.startswith("gas"):
        return "газ"
    return "вода"


def empty_tops():
    return pd.DataFrame({c: pd.Series(dtype='float' if c in ("Кровля, м", "Подошва, м") else 'object')
                         for c in TOPS_COLS})


def parse_tops_table(file_name, raw):
    """Таблица отбивок из CSV или Excel: скважина, пласт, кровля, подошва, (насыщение)."""
    if file_name.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(io.BytesIO(raw))
    else:
        text = None
        for enc in ('utf-8-sig', 'cp1251', 'latin-1'):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        df = pd.read_csv(io.StringIO(text), sep=None, engine='python')
    low = {c: str(c).strip().lower() for c in df.columns}

    def find(*keys):
        return next((c for c, name in low.items() if any(k in name for k in keys)), None)

    c_well, c_layer = find('скваж', 'well'), find('пласт', 'layer', 'zone', 'горизонт', 'horizon')
    c_top, c_bot, c_sat = find('кровл', 'top'), find('подош', 'bot', 'base'), find('насыщ', 'fluid', 'sat')
    if not (c_well and c_layer and c_top and c_bot):
        raise ValueError("нужны колонки «скважина», «пласт», «кровля», «подошва»")

    def num(col):
        return pd.to_numeric(df[col].astype(str).str.replace(',', '.').str.strip(), errors='coerce')

    out = pd.DataFrame({
        "Скважина": df[c_well].map(lambda v: re.sub(r"^(\d+)\.0+$", r"\1", str(v).strip())),
        "Пласт": df[c_layer].astype(str).str.strip(),
        "Кровля, м": num(c_top),
        "Подошва, м": num(c_bot),
        "Насыщение": df[c_sat].map(_norm_sat) if c_sat else "вода",
    })
    return out.dropna(subset=["Кровля, м", "Подошва, м"]).reset_index(drop=True)


def match_tops(tops_df, wells):
    """{индекс скважины: {пласт: (кровля, подошва, насыщение)}} по глубинам по стволу;
    список названий скважин из таблицы, которых нет среди загруженных."""
    keys = {}
    for k, w in enumerate(wells):
        for name in (w['name'], w['well_name'], w['stem'], w['file']):
            keys.setdefault(_norm_name(name), k)
    result, unknown = {}, set()
    if tops_df is None or tops_df.empty:
        return result, []
    for _, row in tops_df.iterrows():
        if pd.isna(row["Кровля, м"]) or pd.isna(row["Подошва, м"]) or not str(row["Пласт"]).strip():
            continue
        k = keys.get(_norm_name(row["Скважина"]))
        if k is None:
            unknown.add(str(row["Скважина"]))
            continue
        top, bot = float(row["Кровля, м"]), float(row["Подошва, м"])
        if bot > top:
            result.setdefault(k, {})[str(row["Пласт"]).strip()] = (top, bot, _norm_sat(row.get("Насыщение", "вода")))
    return result, sorted(unknown)


def layers_in_order(tops_map, wells_shift):
    """Пласты, отсортированные по средней глубине кровли."""
    acc = {}
    for k, layers in tops_map.items():
        for name, (top, _, _) in layers.items():
            acc.setdefault(name, []).append(top - wells_shift[k])
    return sorted(acc, key=lambda n: np.mean(acc[n]))


def build_zones(tops_map, selected, wells_shift):
    """Зоны расчета: выбранные пласты и интервалы между соседними выбранными пластами.
    bounds: {индекс скважины: (кровля, подошва)} в рабочей системе глубин."""
    zones = []
    for i, name in enumerate(selected):
        bounds = {k: (lay[name][0] - wells_shift[k], lay[name][1] - wells_shift[k])
                  for k, lay in tops_map.items() if name in lay}
        zones.append({'name': name, 'kind': 'layer', 'bounds': bounds})
        if i + 1 < len(selected):
            nxt = selected[i + 1]
            between = {}
            for k, lay in tops_map.items():
                if name in lay and nxt in lay and lay[nxt][0] > lay[name][1] + 1e-6:
                    between[k] = (lay[name][1] - wells_shift[k], lay[nxt][0] - wells_shift[k])
            if between:
                zones.append({'name': f"{name} – {nxt}", 'kind': 'between', 'bounds': between})
    return [z for z in zones if z['bounds']]


def zone_bounds_at(raw_w, zone):
    """Кровля и подошва зоны в проектной точке: IDW по скважинам, где зона есть."""
    ks = list(zone['bounds'])
    w = raw_w[ks]
    tops = np.array([zone['bounds'][k][0] for k in ks])
    bots = np.array([zone['bounds'][k][1] for k in ks])
    return float((w * tops).sum() / w.sum()), float((w * bots).sum() / w.sum())


# ==========================================
# 4. ГЕНЕРАЦИЯ: IDW-ТРЕНД + СЛУЧАЙНАЯ СОСТАВЛЯЮЩАЯ
# ==========================================
def idw_raw_weights(wells, tx, ty):
    d2 = np.array([(w['x'] - tx) ** 2 + (w['y'] - ty) ** 2 for w in wells], dtype=float)
    return (d2 + IDW_SMOOTHING ** 2) ** (-IDW_POWER / 2.0)


def _noise_patterns(rng, params, n):
    """Форма изменчивости ГК и ПС от нейросети, растянутая на n отсчетов."""
    net, meta = params['net'], params['meta']
    if net is not None:
        z_sample = rng.normal(0.0, params['sigma'], size=(1, 4))
        out_cvae = net.decode(z_sample)
        gk_eval = (out_cvae[0, 0, :] + 1.0) / 2.0 * (meta['gk_max'] - meta['gk_min']) + meta['gk_min']
        ps_eval = (out_cvae[0, 1, :] + 1.0) / 2.0 * (meta['ps_max'] - meta['ps_min']) + meta['ps_min']
        gk_noise = gk_eval - np.mean(gk_eval)
        ps_noise = ps_eval - np.mean(ps_eval)
        gk_std = np.std(gk_noise) if np.std(gk_noise) > 1e-5 else 1.0
        u_src, u_dst = np.linspace(0.0, 1.0, CVAE_POINTS), np.linspace(0.0, 1.0, n)
        return np.interp(u_dst, u_src, gk_noise / gk_std), np.interp(u_dst, u_src, ps_noise)
    return rng.normal(0.0, 1.0, size=n), rng.normal(0.0, 1.0, size=n)


def _method_noise(m, n, cvae_pattern, ps_noise, s_m, alpha, seed, ref, params):
    if m == 'PS':
        return gaussian_filter1d(ps_noise, sigma=1.0) * alpha
    if m == 'GK':
        return gaussian_filter1d(cvae_pattern, sigma=0.4) * (s_m * alpha * 0.45)
    eps = np.random.default_rng(seed).normal(0.0, 1.0, size=n)
    r = ref['r_gk'].get(m)
    if params.get('correlated') and r is not None:
        eps = r * cvae_pattern + np.sqrt(1.0 - r * r) * eps
    return gaussian_filter1d(eps, sigma=0.6) * (s_m * alpha * 0.25)


def _weighted_trend(rows, wts):
    """IDW-среднее по строкам с учетом пропусков: на каждой глубине — только скважины с замером."""
    mat = np.vstack(rows)
    w_m = np.asarray(wts, dtype=float)[:, None]
    finite = np.isfinite(mat)
    den = (w_m * finite).sum(axis=0)
    num = (w_m * np.where(finite, mat, 0.0)).sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _point_common(pt, wells, params):
    raw_w = idw_raw_weights(wells, pt['x'], pt['y'])
    weights = raw_w / raw_w.sum()
    alpha = params['influence'] * np.clip(1.2 * (1.0 - weights.max()), 0.2, 1.0)
    alts = params.get('alts')
    alt = float((weights * alts).sum()) if alts is not None else 0.0
    return raw_w, weights, alpha, alt


def _finalize(m, values):
    if m in NON_NEGATIVE:
        return np.where(np.isfinite(values), np.maximum(values, 0.0), values)
    return values


def synthesize(points, wells, ref, n_depth, params, seed_offset=0):
    """Режим «по интервалу глубин»: значения складываются с одинаковой глубины."""
    base_seed = int(params['seed']) + int(seed_offset)
    result = []
    for idx_pt, pt in enumerate(points):
        raw_w, weights, alpha, alt = _point_common(pt, wells, params)
        rng = np.random.default_rng(base_seed + idx_pt * 29)
        cvae_pattern, ps_noise = _noise_patterns(rng, params, n_depth)

        curves = {}
        for m in ref['methods']:
            idx, mat = ref['curves'][m]
            trend = _weighted_trend(list(mat), raw_w[idx])
            seed = base_seed + idx_pt * 43 + sum(ord(c) for c in m)
            noise = _method_noise(m, n_depth, cvae_pattern, ps_noise, ref['std'][m], alpha, seed, ref, params)
            curves[m] = _finalize(m, trend + noise)

        result.append({
            'id': pt['id'], 'x': pt['x'], 'y': pt['y'], 'color': pt['color'],
            'weights': weights, 'alpha': alpha, 'alt': alt, 'curves': curves, 'tops': [],
        })
    return result


def synthesize_layers(points, wells, ref, zones, depth_axis, params, seed_offset=0):
    """Режим «по пластам»: кровля и подошва каждой зоны в точке считаются IDW, внутри зоны значения
    складываются по одинаковому относительному положению (кровля — 0, подошва — 1)."""
    base_seed = int(params['seed']) + int(seed_offset)
    max_gap, max_gap_ak = ref['gap_limits']
    n_all = len(depth_axis)
    result = []
    for idx_pt, pt in enumerate(points):
        raw_w, weights, alpha, alt = _point_common(pt, wells, params)
        rng = np.random.default_rng(base_seed + idx_pt * 29)
        curves = {m: np.full(n_all, np.nan) for m in ref['methods']}
        tops = []
        for zi, zone in enumerate(zones):
            top_s, bot_s = zone_bounds_at(raw_w, zone)
            tops.append({'name': zone['name'], 'kind': zone['kind'], 'top': top_s, 'bot': bot_s})
            mask = (depth_axis >= top_s - 1e-9) & (depth_axis <= bot_s + 1e-9)
            n = int(mask.sum())
            if n == 0 or bot_s <= top_s:
                continue
            u = (depth_axis[mask] - top_s) / (bot_s - top_s)
            cvae_pattern, ps_noise = _noise_patterns(rng, params, n)
            for m in ref['methods']:
                rows, wts = [], []
                for k, (zt, zb) in zone['bounds'].items():
                    w = wells[k]
                    if m not in w['curves']:
                        continue
                    gap_limit = max_gap_ak if m == 'AK' else max_gap
                    rows.append(resample_to_grid(w['zdepth'], w['curves'][m], zt + u * (zb - zt), w['step'], gap_limit))
                    wts.append(raw_w[k])
                if not rows:
                    continue
                trend = _weighted_trend(rows, wts)
                seed = base_seed + idx_pt * 43 + sum(ord(c) for c in m) + zi * 7919
                noise = _method_noise(m, n, cvae_pattern, ps_noise, ref['std'][m], alpha, seed, ref, params)
                curves[m][mask] = _finalize(m, trend + noise)
        result.append({
            'id': pt['id'], 'x': pt['x'], 'y': pt['y'], 'color': pt['color'],
            'weights': weights, 'alpha': alpha, 'alt': alt, 'curves': curves, 'tops': tops,
        })
    return result


# ==========================================
# 5. ЭКСПОРТ LAS И ПРОЕКТ
# ==========================================
def make_las_text(gw, depth_axis, units, methods=None):
    """depth_axis — рабочая шкала (по стволу или абсолютная); в файл пишется глубина по стволу
    синтетической скважины: рабочая глубина + ее альтитуда."""
    alt = gw.get('alt', 0.0) or 0.0
    names = [c for c in gw['curves'] if methods is None or c in methods]
    sl = slice(None)
    if gw.get('tops'):
        # в режиме «по пластам» отрезаем пустые края выше и ниже пластов
        finite = np.zeros(len(depth_axis), dtype=bool)
        for c in names:
            finite |= np.isfinite(gw['curves'][c])
        if finite.any():
            idx = np.where(finite)[0]
            sl = slice(idx[0], idx[-1] + 1)
    md = np.round(depth_axis[sl] + alt, 4)

    las = lasio.LASFile()
    las.well['WELL'].value = gw['id']
    las.well['NULL'].value = NULL_OUT
    las.well['STRT'].value = float(md[0])
    las.well['STOP'].value = float(md[-1])
    las.well['STEP'].value = float(round(md[1] - md[0], 4)) if len(md) > 1 else 0.0
    las.well['X'] = lasio.HeaderItem('X', value=str(gw['x']), descr='X Coordinate')
    las.well['Y'] = lasio.HeaderItem('Y', value=str(gw['y']), descr='Y Coordinate')
    if alt:
        las.well['RKB'] = lasio.HeaderItem('RKB', unit='M', value=str(round(alt, 2)), descr='Altitude (IDW)')
    if gw.get('tops'):
        lines = ["Tops of synthetic well (MD, m): zone; top; bottom"]
        lines += [f"{t['name']}; {t['top'] + alt:.2f}; {t['bot'] + alt:.2f}" for t in gw['tops']]
        las.other = "\n".join(lines)
    las.append_curve("DEPT", md, unit="M", descr="Depth")
    for c_name in names:
        vals = gw['curves'][c_name][sl]
        las.append_curve(c_name, np.where(np.isfinite(vals), vals, NULL_OUT), unit=units.get(c_name, ""),
                         descr=f"Synthetic {c_name} (IDW+CVAE)")
    buf = io.StringIO()
    las.write(buf, version=2.0)
    return buf.getvalue()


def tops_table_for(generated, use_abs):
    rows = []
    for g in generated:
        alt = g.get('alt', 0.0) or 0.0
        for t in g.get('tops', []):
            row = {"Скважина": g['id'], "Зона": t['name'],
                   "Кровля, м": round(t['top'] + alt, 2), "Подошва, м": round(t['bot'] + alt, 2),
                   "Мощность, м": round(t['bot'] - t['top'], 2)}
            if use_abs:
                row["Кровля, абс. м"] = round(-t['top'], 2)
                row["Подошва, абс. м"] = round(-t['bot'], 2)
            rows.append(row)
    return pd.DataFrame(rows)


def las_file_name(gw):
    return f"{gw['id']}_X{int(gw['x'])}_Y{int(gw['y'])}.las"


SETTING_KEYS = ['geo_top', 'geo_bot', 'geo_step', 'geo_infl', 'geo_seed', 'geo_sigma', 'geo_gap', 'geo_gap_ak',
                'geo_corr', 'geo_mode', 'geo_abs', 'geo_layers', 'geo_margin']


def build_project_zip(sources, overrides, points, tops_df=None):
    meta = {
        'version': 2,
        'settings': {k: st.session_state.get(k) for k in SETTING_KEYS},
        'wells': {name: overrides.get(name) for name, _ in sources},
        'points': points,
        'tops': [] if tops_df is None else json.loads(tops_df.to_json(orient='records', force_ascii=False)),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", json.dumps(meta, ensure_ascii=False, indent=1))
        for name, data in sources:
            zf.writestr(f"las/{name}", data)
    return buf.getvalue()


def read_project_zip(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        meta = json.loads(zf.read("project.json").decode("utf-8"))
        files = [(os.path.basename(n), zf.read(n)) for n in zf.namelist()
                 if n.startswith("las/") and n.lower().endswith(".las")]
    return meta, files


# ==========================================
# 6. ПЛАНШЕТЫ (PLOTLY)
# ==========================================
def _axis_def(m, color, units, is_log, rng, overlay=None, domain=None, side='top', grid=True):
    unit = units.get(m, "")
    ax = dict(
        title=dict(text=f"<b>{method_label(m)}</b>" + (f", {unit}" if unit else ""),
                   font=dict(color=color, size=12), standoff=4),
        type='log' if is_log else 'linear', side=side, anchor='y',
        tickfont=dict(color=color, size=10), showgrid=grid, gridcolor='#E2E8F0', zeroline=False,
        linecolor=color, showline=True,
    )
    if overlay:
        ax['overlaying'] = overlay
    if domain is not None:
        ax['domain'] = domain
    lo, hi = rng
    if lo is not None and hi is not None and hi > lo:
        if is_log:
            if lo > 0:
                ax['range'] = [np.log10(lo), np.log10(hi)]
        else:
            ax['range'] = [lo, hi]
    return ax


LAYER_COLOR = '#F59E0B'


def _layer_shapes(tops, x0=0.0, x1=1.0, label=True):
    """Полоса пласта и линии кровли и подошвы в пределах колонки [x0, x1] (доли ширины рисунка)."""
    shapes, notes = [], []
    for t in tops or []:
        if t['kind'] != 'layer':
            continue
        shapes.append(dict(type='rect', xref='paper', yref='y', x0=x0, x1=x1, y0=t['top'], y1=t['bot'],
                           fillcolor='rgba(245,158,11,0.07)', line=dict(width=0), layer='below'))
        for yv in (t['top'], t['bot']):
            shapes.append(dict(type='line', xref='paper', yref='y', x0=x0, x1=x1, y0=yv, y1=yv,
                               line=dict(color=LAYER_COLOR, width=1.4, dash='dash')))
        if label:
            notes.append(dict(text=f"<b>{t['name']}</b> · {fmt_num(t['bot'] - t['top'])} м", x=x0, y=t['top'],
                              xref='paper', yref='y', xanchor='left', yanchor='bottom', showarrow=False,
                              font=dict(size=11, color='#B45309'), bgcolor='rgba(255,255,255,0.8)'))
    return shapes, notes


def _finish_layout(fig, layout, top_margin=70, y_title='Глубина, м', shapes=None, notes=None):
    layout['yaxis'] = dict(autorange='reversed', title=dict(text=y_title), showgrid=True,
                           gridcolor='#E2E8F0', zeroline=False, ticksuffix=' ')
    if shapes:
        layout['shapes'] = shapes
    if notes:
        layout['annotations'] = list(layout.get('annotations', [])) + notes
    fig.update_layout(
        **layout,
        height=760, plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
        margin=dict(l=60, r=20, t=top_margin, b=70),
        hovermode='y unified',
        legend=dict(orientation='h', yanchor='top', y=-0.06, x=0, font=dict(size=11)),
        dragmode='zoom',
    )
    return fig


def build_planchette(depth, tracks, wells, ref, synth, synth_id, show_ref=True, compare=None,
                     y_title='Глубина, м', tops=None):
    n = len(tracks)
    gap = 0.03 if n > 1 else 0.0
    width = (1.0 - gap * (n - 1)) / n
    fig = go.Figure()
    layout = {}
    legend_names = set()
    compare = compare or []

    for t, cfg in enumerate(tracks):
        k = t + 1
        ax = 'x' if k == 1 else f'x{k}'
        domain = [t * (width + gap), t * (width + gap) + width]
        for pos, m, color in (('top', cfg['m1'], cfg['c1']), ('bottom', cfg['m2'], cfg['c2'])):
            if m is None:
                continue
            if pos == 'top':
                axis_ref, axis_key = ax, ('xaxis' if k == 1 else f'xaxis{k}')
                layout[axis_key] = _axis_def(m, color, ref['units'], cfg['log'], cfg['range'], domain=domain)
            else:
                axis_ref, axis_key = f'x{n + k}', f'xaxis{n + k}'
                layout[axis_key] = _axis_def(m, color, ref['units'], cfg.get('log2', False), (None, None),
                                             overlay=ax, side='bottom', grid=False)
            dash = 'solid' if pos == 'top' else 'dash'

            if show_ref and m in ref['curves']:
                idx, mat = ref['curves'][m]
                for row_k, wk in enumerate(idx):
                    w = wells[wk]
                    show = w['name'] not in legend_names
                    legend_names.add(w['name'])
                    fig.add_trace(go.Scattergl(
                        x=mat[row_k], y=depth, xaxis=axis_ref, yaxis='y', mode='lines',
                        line=dict(color=w['color'], width=1, dash='solid' if pos == 'top' else 'dot'),
                        opacity=0.5, name=w['name'], legendgroup=w['name'], showlegend=show,
                        hovertemplate=f"{w['name']} · {method_label(m)}: %{{x:.3f}}<extra></extra>",
                    ))
            for comp in compare:
                if m in comp['curves']:
                    show = comp['id'] not in legend_names
                    legend_names.add(comp['id'])
                    fig.add_trace(go.Scattergl(
                        x=comp['curves'][m], y=depth, xaxis=axis_ref, yaxis='y', mode='lines',
                        line=dict(color=comp['color'], width=1.6, dash=dash), opacity=0.9,
                        name=comp['id'], legendgroup=comp['id'], showlegend=show,
                        hovertemplate=f"{comp['id']} · {method_label(m)}: %{{x:.3f}}<extra></extra>",
                    ))
            if m in synth:
                fig.add_trace(go.Scattergl(
                    x=synth[m], y=depth, xaxis=axis_ref, yaxis='y', mode='lines',
                    line=dict(color=color, width=2.6, dash=dash),
                    name=f"{synth_id} · {method_label(m)}",
                    hovertemplate=f"<b>{synth_id} · {method_label(m)}: %{{x:.3f}}</b><extra></extra>",
                ))
    shapes, notes = _layer_shapes(tops)
    return _finish_layout(fig, layout, y_title=y_title, shapes=shapes, notes=notes)


def build_correlation(depth, columns, m1, m2, c1, c2, log1, log2, units, y_title='Глубина, м'):
    """Схема корреляции: скважины рядом на одной шкале глубин; одноименные кровли и подошвы
    пластов соседних скважин соединены линиями."""
    n = len(columns)
    gap = 0.02 if n > 1 else 0.0
    width = (1.0 - gap * (n - 1)) / n
    fig = go.Figure()
    layout, annotations = {}, []
    for t, col in enumerate(columns):
        k = t + 1
        ax = 'x' if k == 1 else f'x{k}'
        domain = [t * (width + gap), t * (width + gap) + width]
        layout['xaxis' if k == 1 else f'xaxis{k}'] = _axis_def(m1, c1, units, log1, (None, None), domain=domain)
        if m1 in col['curves']:
            fig.add_trace(go.Scattergl(
                x=col['curves'][m1], y=depth, xaxis=ax, yaxis='y', mode='lines',
                line=dict(color=c1, width=1.8), showlegend=False,
                hovertemplate=f"{col['name']} · {method_label(m1)}: %{{x:.3f}}<extra></extra>"))
        if m2:
            ax2 = f'x{n + k}'
            layout[f'xaxis{n + k}'] = _axis_def(m2, c2, units, log2, (None, None), overlay=ax, side='bottom', grid=False)
            if m2 in col['curves']:
                fig.add_trace(go.Scattergl(
                    x=col['curves'][m2], y=depth, xaxis=ax2, yaxis='y', mode='lines',
                    line=dict(color=c2, width=1.6, dash='dash'), showlegend=False,
                    hovertemplate=f"{col['name']} · {method_label(m2)}: %{{x:.3f}}<extra></extra>"))
        annotations.append(dict(
            text=f"<b>{col['name']}</b>" + ("<br><span style='font-size:10px;color:#64748B'>опорная</span>"
                                             if col['is_ref'] else ""),
            x=(domain[0] + domain[1]) / 2, y=1.0, xref='paper', yref='paper', yshift=58,
            showarrow=False, font=dict(size=12, color=col['color'])))
    layout['annotations'] = annotations

    shapes, notes = [], []
    for t, col in enumerate(columns):
        domain = [t * (width + gap), t * (width + gap) + width]
        s, a = _layer_shapes(col.get('tops'), domain[0], domain[1], label=(t == 0))
        shapes += s
        notes += a
        if t + 1 < n:
            nxt = {z['name']: z for z in columns[t + 1].get('tops') or [] if z['kind'] == 'layer'}
            x_next = (t + 1) * (width + gap)
            for z in col.get('tops') or []:
                if z['kind'] == 'layer' and z['name'] in nxt:
                    for key in ('top', 'bot'):
                        shapes.append(dict(type='line', xref='paper', yref='y', x0=domain[1], x1=x_next,
                                           y0=z[key], y1=nxt[z['name']][key],
                                           line=dict(color=LAYER_COLOR, width=1.4, dash='dot')))
    return _finish_layout(fig, layout, top_margin=110, y_title=y_title, shapes=shapes, notes=notes)


# ==========================================
# 7. КАРТА: СЕТКА КООРДИНАТ И СТИЛИ
# ==========================================
class Graticule(MacroElement):
    """Сетка координат с подписями у краев карты; шаг меняется вместе с масштабом."""
    _template = Template(r"""
    {% macro script(this, kwargs) %}
    (function() {
        var map = {{ this._parent.get_name() }};
        var unit = {{ this.unit }}, x0 = {{ this.x0 }}, y0 = {{ this.y0 }};
        var grid = L.layerGroup().addTo(map);
        var info = L.control({position: 'bottomright'});
        info.onAdd = function() { this._div = L.DomUtil.create('div', 'geo-scale'); return this._div; };
        info.addTo(map);
        function nice(v) {
            var p = Math.pow(10, Math.floor(Math.log(v) / Math.LN10)), f = v / p;
            return (f < 1.5 ? 1 : (f < 3.5 ? 2 : (f < 7.5 ? 5 : 10))) * p;
        }
        function fmt(v) { return Math.round(v).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' '); }
        function lbl(html) { return L.divIcon({className: 'geo-ax-icon', iconSize: [0, 0], html: html}); }
        function draw() {
            grid.clearLayers();
            var b = map.getBounds(), s = b.getSouth(), n = b.getNorth(), w = b.getWest(), e = b.getEast();
            var xmin = w * unit + x0, xmax = e * unit + x0, ymin = s * unit + y0, ymax = n * unit + y0;
            var step = nice(Math.max(xmax - xmin, ymax - ymin) / 7);
            var opt = {color: '#CBD5E1', weight: 1, dashArray: '4,4', interactive: false};
            var v, c;
            for (v = Math.ceil(xmin / step) * step; v <= xmax; v += step) {
                c = (v - x0) / unit;
                L.polyline([[s, c], [n, c]], opt).addTo(grid);
                L.marker([s, c], {interactive: false, keyboard: false,
                    icon: lbl('<div class="geo-ax geo-ax-x">X ' + fmt(v) + '</div>')}).addTo(grid);
            }
            for (v = Math.ceil(ymin / step) * step; v <= ymax; v += step) {
                c = (v - y0) / unit;
                L.polyline([[c, w], [c, e]], opt).addTo(grid);
                L.marker([c, w], {interactive: false, keyboard: false,
                    icon: lbl('<div class="geo-ax geo-ax-y">Y ' + fmt(v) + '</div>')}).addTo(grid);
            }
            info._div.innerHTML = 'Шаг сетки: ' + fmt(step) + ' м';
        }
        map.on('moveend zoomend resize', draw);
        setTimeout(draw, 0);
    })();
    {% endmacro %}
    """)

    def __init__(self, unit, x0, y0):
        super().__init__()
        self._name = 'Graticule'
        self.unit, self.x0, self.y0 = float(unit), float(x0), float(y0)


MAP_CSS = """
<style>
.leaflet-container {background: #FAFAFA !important; cursor: crosshair;}
.geo-ax {font-size: 10px; font-weight: 600; color: #64748B; background: rgba(255,255,255,0.9);
         padding: 1px 5px; border-radius: 3px; border: 1px solid #E2E8F0; white-space: nowrap;}
.geo-ax-x {transform: translate(-50%, -135%); display: inline-block;}
.geo-ax-y {transform: translate(4px, -50%); display: inline-block;}
.geo-scale {background: rgba(255,255,255,0.92); padding: 2px 8px; border: 1px solid #E2E8F0;
            border-radius: 4px; font-size: 11px; color: #475569;}
.geo-diamond {width: 12px; height: 12px; border: 1.5px solid #0F172A; transform: rotate(45deg); box-sizing: border-box;}
.geo-lbl {font-size: 10px; font-weight: 700; margin-top: 4px; margin-left: -10px; white-space: nowrap;
          text-shadow: 0 0 2px #fff, 0 0 2px #fff, 0 0 2px #fff;}
.geo-sel-wrap {position: relative; width: 16px; height: 16px;}
.geo-sel-wrap .geo-diamond {position: absolute; left: 0; top: 0; width: 16px; height: 16px; border-width: 2px;
                            box-shadow: 0 0 0 2px #fff; animation: geo-bounce 0.9s ease-in-out infinite;}
.geo-ring {position: absolute; left: -8px; top: -8px; width: 32px; height: 32px; border-radius: 50%;
           border: 2px solid; animation: geo-pulse 1.3s ease-out infinite;}
.geo-well-ring {position: absolute; left: -15px; top: -15px; width: 30px; height: 30px; border-radius: 50%;
                border: 3px solid #F59E0B; animation: geo-pulse 1.3s ease-out infinite;}
.geo-wlbl {transform: translate(-50%, -50%); display: inline-block; background: #fff; border: 1px solid #E2E8F0;
           border-radius: 10px; padding: 0 6px; font-size: 10px; font-weight: 700; white-space: nowrap;}
@keyframes geo-bounce {0%, 100% {transform: translateY(0) rotate(45deg);} 50% {transform: translateY(-5px) rotate(45deg);}}
@keyframes geo-pulse {0% {transform: scale(0.5); opacity: 0.9;} 100% {transform: scale(1.6); opacity: 0;}}
</style>
"""


# ==========================================
# 8. СПРАВКА
# ==========================================
HELP_MD = """
#### Как пользоваться
1. **Загрузите LAS-файлы** опорных скважин (не меньше двух). Координаты устья берутся из шапки файла (X, Y), в том числе из LAS 1.2.
   Если координат в файле нет, скважина ставится условно, задайте X и Y вручную (кнопка «Задать координаты» слева).
2. **Выберите режим расчета.**
   - «По пластам (по отбивкам)» — основной режим, если есть таблица отбивок (раздел «Отбивки пластов» слева).
     Пласт выбирается по названию, глубины вводить не нужно.
   - «По интервалу глубин» — окно глубин для просмотра, общее для всех скважин. Начальные значения программа
     подставляет сама (начало общего для всех файлов интервала + 5 м, длина 35 м), с геологией они не связаны:
     укажите глубины интересующего пласта.

   Шаг по умолчанию равен самому мелкому шагу записи среди загруженных файлов.
3. **Поставьте синтетические скважины** на вкладке «Карта и расстановка точек»: щелчком по карте, вводом координат,
   регулярной сеткой или профилем по линии.
4. **Посмотрите результат** на вкладке «Каротажный планшет»: одну скважину или схему корреляции из нескольких.
5. **Скачайте LAS-файлы** на вкладке «Сводка и экспорт LAS», там же можно подготовить сразу несколько вариантов для группы
   студентов и сохранить проект, чтобы открыть его в следующий раз.

#### Работа с картой
- **Щелчок по пустому месту** ставит новую синтетическую скважину (если включено «Добавлять точку по щелчку»).
- **Щелчок по точке или опорной скважине** выбирает ее: она подсвечивается, справа открывается карточка.
  В карточке можно поменять название и координаты, удалить точку или переместить ее: нажмите «Переместить» и щелкните новое место.
- Линии от выбранной точки к опорным скважинам показывают веса IDW: чем толще линия, тем больше влияние скважины.
- «Отменить действие» возвращает точки к состоянию до последнего добавления, удаления, перемещения, сетки или профиля.
- Подписи координат всегда у краев карты, шаг сетки указан в правом нижнем углу.

#### Расчет по пластам
- **Таблица отбивок**: скважина, пласт, кровля, подошва (глубины по стволу, как в LAS), насыщение. Загружается
  из CSV или Excel или заполняется прямо в приложении; кнопка «Скачать шаблон» дает таблицу с названиями скважин.
  Таблица сохраняется вместе с проектом.
- **Кровля и подошва пласта** в синтетической скважине считаются тем же IDW, что и кривые. Метод работает и с двумя
  скважинами; скважина, в которой пласта нет, для этого пласта не участвует.
- **Внутри пласта** значения складываются не по одинаковой глубине, а по одинаковому положению: кровля с кровлей,
  середина с серединой, подошва с подошвой. Поэтому пласт не обрезается, если в скважинах он на разной глубине
  и разной мощности. Если выбрано несколько пластов, интервалы между ними считаются так же.
- Границы пластов видны на планшете и на схеме корреляции (одноименные кровли и подошвы соединены линиями),
  рассчитанные отбивки синтетических скважин выгружаются вместе с LAS.

#### Альтитуда
- Скважины считаются вертикальными. Если альтитуда (RKB, KB) есть у всех скважин, скважины сопоставляются по
  абсолютным отметкам: абсолютная глубина = глубина по стволу − альтитуда. Альтитуду можно ввести вручную в таблице
  «Опорные скважины». Альтитуда синтетической скважины считается IDW, в LAS записывается глубина по стволу.

#### Как считается синтетика
- Кривые не растягиваются и не сжимаются: в режиме по интервалу складываются только значения с одинаковых глубин. Разный шаг записи приводится к выбранному шагу.
- Короткие пропуски в записи (до 0,5 м, для АК до 10 м) заполняются по соседним замерам. На глубинах с длинными пропусками
  скважина в расчете не участвует, приложение показывает такие интервалы. Если данных нет ни в одной скважине, синтетика там не строится.
- **Тренд** считается методом обратных взвешенных расстояний (IDW): чем ближе опорная скважина, тем больше ее вес.
- **Случайная составляющая** делает кривые неидентичными. Для ГК и ПС ее форму дает нейросетевая модель, для остальных методов
  используется сглаженный шум. Рядом с опорной скважиной случайная составляющая минимальна, между скважинами она растет.

#### Настройки
- **Степень вариативности**: насколько сильно к тренду добавляется случайная составляющая. 0 означает чистую интерполяцию.
- **Номер варианта (seed)**: при одном и том же номере результат повторяется. Сменили номер — получили новый вариант.
- **Sigma** (в «Дополнительно»): насколько необычной может быть форма изменчивости от нейросети.
- **Пороги пропусков** (в «Дополнительно»): до какой длины пропуски заполняются.
- **Согласовать шум с ГК** (в «Дополнительно»): случайная составляющая остальных методов связывается с ГК
  с тем же коэффициентом корреляции, что и в опорных скважинах.

#### Планшет
- Цвет кривой, второй метод на том же треке, линейная или логарифмическая шкала и пределы шкалы задаются в блоке «Настройка треков».
- Можно скрыть опорные скважины и наложить для сравнения другие синтетические скважины.
- «Схема корреляции» показывает несколько скважин рядом на одной шкале глубин, удобно вместе с профилем по линии.
- Выделите область мышью, чтобы увеличить ее; двойной щелчок возвращает исходный масштаб. При наведении видны значения на этой глубине.
"""


# ==========================================
# 9. ИНТЕРФЕЙС
# ==========================================
class _MemFile:
    """Файл из памяти (из проекта или с диска) в том же виде, что и загруженный."""

    def __init__(self, name, data):
        self.name = name
        self._b = data

    def getvalue(self):
        return self._b


CSS = """
<style>
.geo-chips {display:flex; flex-wrap:wrap; gap:6px; margin:2px 0 12px 0;}
.geo-chip {background:#F1F5F9; border:1px solid #E2E8F0; border-radius:14px; padding:3px 11px;
           font-size:12.5px; color:#1E293B;}
.geo-chip b {color:#1E3A8A;}
.geo-badges {background:#F8FAFC; border:1px solid #E2E8F0; padding:8px 12px; border-radius:8px; margin-bottom:10px;}
.geo-badge {display:inline-flex; align-items:center; background:#FFFFFF; border:1px solid #E2E8F0;
            padding:3px 10px; border-radius:14px; margin:3px 6px 3px 0; font-size:12px; color:#1E293B;}
.geo-dot {width:9px; height:9px; border-radius:50%; margin-right:6px; display:inline-block;}
.geo-card-title {font-size:17px; font-weight:700; color:#0F172A; margin:2px 0 6px 0; display:flex; align-items:center; gap:8px;}
.geo-card-sub {font-size:12.5px; color:#475569; margin-bottom:8px;}
.geo-step {background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:14px 16px; height:100%;}
.geo-step b {color:#1E3A8A; font-size:15px;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {padding: 8px 16px; background: #F1F5F9; border-radius: 8px 8px 0 0;}
.stTabs [data-baseweb="tab"] p {font-size: 15px; font-weight: 600;}
.stTabs [aria-selected="true"] {background: #E0E7FF;}
</style>
"""


def _nice_step(span, target=6):
    for s in [10, 25, 50, 100, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 50000, 100000]:
        if span / s <= target:
            return float(s)
    return float(10 ** np.ceil(np.log10(span / target)))


def _free_name(points, prefix):
    names = {p['id'] for p in points}
    n = 1
    while f"{prefix}_{n}" in names:
        n += 1
    return f"{prefix}_{n}", n


def _make_point(x, y, name, n):
    return {'id': name, 'x': float(round(x, 1)), 'y': float(round(y, 1)), 'active': True,
            'color': SYNTH_COLORS[(n - 1) % len(SYNTH_COLORS)]}


def _set_points(points):
    st.session_state.points_list = points
    st.session_state.points_base = [dict(p) for p in points]
    st.session_state.points_ver += 1


def _commit_points(points):
    """Изменение точек с возможностью отмены."""
    st.session_state.history.append([dict(p) for p in st.session_state.points_list])
    st.session_state.history = st.session_state.history[-30:]
    _set_points(points)


def _badges(items):
    return "".join(
        f'<span class="geo-badge"><span class="geo-dot" style="background:{c};"></span>'
        f'<strong>{name}</strong>:&nbsp;<span style="color:#2563EB; font-weight:600;">{val:.1%}</span></span>'
        for name, c, val in items)


def _init_state():
    defaults = {
        "points_list": [], "points_base": [], "points_ver": 0, "history": [],
        "selected": None, "move_mode": False,
        "last_click": None, "last_obj_count": None,
        "map_key_version": 0, "ref_overrides": {}, "ref_ver": 0, "ref_expanded": False,
        "variants_zip": None, "project_zip": None,
        "project_files": [], "proj_ver": 0, "skip_defaults_once": False,
        "click_mode": True, "show_labels": True, "labels_off_pending": False,
        "geo_infl": 0.35, "geo_seed": 101, "geo_sigma": 0.8, "geo_gap": 0.5, "geo_gap_ak": 10.0, "geo_corr": False,
        "geo_mode": MODE_INTERVAL, "geo_abs": True,
        "tops_df": None, "tops_current": None, "tops_ver": 0, "tops_autoloaded": False, "last_abs": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
    if st.session_state.labels_off_pending:
        st.session_state.show_labels = False
        st.session_state.labels_off_pending = False


def _select(kind, obj_id):
    st.session_state.selected = {'type': kind, 'id': obj_id}
    st.session_state.move_mode = False
    if kind == 'point':
        st.session_state.planchette_sel = obj_id


def _sanitize(key, options, multi=False):
    """Убирает из состояния виджета значения, которых больше нет среди вариантов."""
    if key not in st.session_state:
        return
    val = st.session_state[key]
    bad = any(v not in options for v in (val or [])) if multi else val not in options
    if bad:
        del st.session_state[key]  # виджет создастся заново со значением по умолчанию


EDITOR_FIELDS = {"Активна": "active", "Скважина": "id", "X, м": "x", "Y, м": "y"}


def _apply_points_editor():
    """Применяет правки из таблицы точек до отрисовки карты, чтобы карта не отставала на один шаг."""
    edits = st.session_state.get(f"pts_editor_{st.session_state.points_ver}")
    if not edits or not edits.get("edited_rows"):
        return
    new = [dict(p) for p in st.session_state.points_base]
    for row, changes in edits["edited_rows"].items():
        r = int(row)
        if r >= len(new):
            continue
        for col, val in changes.items():
            field = EDITOR_FIELDS.get(col)
            if field is None or val is None:
                continue
            if field in ('x', 'y'):
                new[r][field] = float(val)
            elif field == 'active':
                new[r][field] = bool(val)
            else:
                new[r][field] = str(val).strip() or new[r]['id']
    seen = set()
    for p in new:
        while p['id'] in seen:
            p['id'] += "_2"
        seen.add(p['id'])
    st.session_state.points_list = new


def run_geo_app():
    # карта нужна только в модуле ГИС: импорт здесь ускоряет открытие стартовой страницы
    import folium
    from streamlit_folium import st_folium

    if st.sidebar.button("Вернуться на главную", use_container_width=True, key="btn_back_seis"):
        st.session_state.current_page = 'home'
        st.rerun()

    _init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin-bottom: 6px;">
            <h1 style="color: #1E3A8A; margin: 0; font-size: 2.2rem; font-weight: 700;">SynthGeoGen</h1>
            <p style="color: #475569; margin-top: 4px; font-size: 0.95rem;">
                Генерация синтетических данных ГИС: интерполяция по координатам устьев (IDW) и стохастическая составляющая (CVAE)
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ------------------------------------------
    # САЙДБАР: ОПОРНЫЕ СКВАЖИНЫ И ПРОЕКТ
    # ------------------------------------------
    st.sidebar.header("1. Опорные скважины")
    uploaded_files = st.sidebar.file_uploader(
        "Загрузите файлы LAS (от 2 файлов)", type=['las'], accept_multiple_files=True,
        help="LAS 1.2 и 2.0. Координаты устья берутся из полей X и Y в разделе ~WELL.") or []

    with st.sidebar.expander("Открыть сохраненный проект", expanded=False):
        proj_file = st.file_uploader("Файл проекта (.zip)", type=['zip'], key=f"proj_up_{st.session_state.proj_ver}",
                                     help="Проект сохраняется на вкладке «Сводка и экспорт LAS».")
        if proj_file is not None:
            try:
                meta, files = read_project_zip(proj_file.getvalue())
                st.session_state.project_files = files
                st.session_state.ref_overrides = {k: v for k, v in meta.get('wells', {}).items() if v}
                st.session_state.ref_ver += 1
                st.session_state.history = []
                st.session_state.selected = None
                pts = []
                for i, p in enumerate(meta.get('points', []), 1):
                    pts.append({'id': str(p.get('id', f"SYNTH_{i}")), 'x': float(p['x']), 'y': float(p['y']),
                                'active': bool(p.get('active', True)),
                                'color': p.get('color') or SYNTH_COLORS[(i - 1) % len(SYNTH_COLORS)]})
                _set_points(pts)
                tops_rec = meta.get('tops') or []
                st.session_state.tops_df = (pd.DataFrame(tops_rec).reindex(columns=TOPS_COLS) if tops_rec else None)
                st.session_state.tops_ver += 1
                for k, v in meta.get('settings', {}).items():
                    if k in SETTING_KEYS and v is not None:
                        st.session_state['geo_margin_init' if k == 'geo_margin' else k] = v
                st.session_state.skip_defaults_once = True
                st.session_state.proj_ver += 1  # очищает поле загрузки, чтобы проект не открывался повторно
                st.rerun()
            except Exception as exc:
                st.error(f"Не удалось открыть проект: {exc}")

    files_all = [_MemFile(n, b) for n, b in st.session_state.project_files]
    names_proj = {f.name for f in files_all}
    files_all += [f for f in uploaded_files if f.name not in names_proj]
    if not files_all and os.environ.get("GEO_AUTOLOAD_DIR"):
        auto_dir = os.environ["GEO_AUTOLOAD_DIR"]
        files_all = [_MemFile(f, open(os.path.join(auto_dir, f), 'rb').read()) for f in sorted(os.listdir(auto_dir))
                     if f.lower().endswith('.las')]
    if st.session_state.project_files:
        st.sidebar.caption("Из проекта: " + ", ".join(sorted(names_proj)))
        if st.sidebar.button("Закрыть проект", use_container_width=True):
            st.session_state.project_files = []
            st.rerun()

    wells_data = []
    curr_hash = "_".join(sorted(f.name for f in files_all))
    if files_all:
        if st.session_state.get("last_uploaded_hash") != curr_hash:
            st.session_state.last_uploaded_hash = curr_hash
            st.session_state.map_key_version += 1
            if not st.session_state.skip_defaults_once:
                for k in ('geo_top', 'geo_bot', 'geo_step'):
                    st.session_state.pop(k, None)
        st.session_state.skip_defaults_once = False

        for idx_f, f in enumerate(files_all):
            info = dict(parse_las_bytes(f.name, f.getvalue()))
            info['color'] = WELL_COLORS[idx_f % len(WELL_COLORS)]
            info['raw'] = f.getvalue()
            wells_data.append(info)

        # отображаемое имя: имя скважины из шапки, при совпадении — имя файла
        names = [w['well_name'] for w in wells_data]
        for w in wells_data:
            w['name'] = w['well_name'] if names.count(w['well_name']) == 1 else w['stem']

        overrides = st.session_state.ref_overrides
        for idx_f, w in enumerate(wells_data):
            if w['file'] not in overrides:
                if w['coords_found']:
                    x0, y0 = w['x_file'], w['y_file']
                else:
                    x0, y0 = float((idx_f + 1) * 450.0), float(200.0 + (idx_f % 2) * 350.0)
                overrides[w['file']] = {'active': True, 'x': float(x0), 'y': float(y0), 'alt': w['alt_file']}
            elif 'alt' not in overrides[w['file']]:
                overrides[w['file']]['alt'] = w['alt_file']

        no_coords = [w['name'] for w in wells_data if not w['coords_found']]
        if no_coords:
            st.sidebar.warning("Нет координат в файле: " + ", ".join(no_coords) +
                               ". Скважины поставлены условно.")
            if st.sidebar.button("Задать координаты", use_container_width=True, key="open_ref"):
                st.session_state.ref_expanded = True
                st.rerun()

        ref_key = f"{curr_hash}_{st.session_state.ref_ver}"
        if st.session_state.get("ref_base_key") != ref_key:
            st.session_state.ref_base_key = ref_key
            st.session_state.ref_base = pd.DataFrame([{
                "Активна": overrides[w['file']]['active'],
                "Скважина": w['name'],
                "X, м": overrides[w['file']]['x'],
                "Y, м": overrides[w['file']]['y'],
                "Альтитуда, м": overrides[w['file']].get('alt'),
                "Координаты": "из файла" if w['coords_found'] else "условные",
            } for w in wells_data]).astype({"Альтитуда, м": float})

        with st.sidebar.expander("Опорные скважины: активность, координаты, альтитуда",
                                 expanded=st.session_state.ref_expanded):
            st.caption("Снимите флажок, чтобы временно исключить скважину из расчета. "
                       "Координаты и альтитуду можно исправить прямо в таблице.")
            edited_ref = st.data_editor(
                st.session_state.ref_base, key=f"ref_editor_{ref_key}", hide_index=True,
                use_container_width=True, num_rows="fixed",
                disabled=["Скважина", "Координаты"],
                column_config={
                    "X, м": st.column_config.NumberColumn(format="%.1f"),
                    "Y, м": st.column_config.NumberColumn(format="%.1f"),
                    "Альтитуда, м": st.column_config.NumberColumn(
                        format="%.1f", help="Альтитуда стола ротора (RKB, KB) из шапки LAS. Пусто — нет данных."),
                })
        for w, (_, row) in zip(wells_data, edited_ref.iterrows()):
            prev = overrides[w['file']]
            overrides[w['file']] = {
                'active': bool(row["Активна"]),
                'x': prev['x'] if pd.isna(row["X, м"]) else float(row["X, м"]),
                'y': prev['y'] if pd.isna(row["Y, м"]) else float(row["Y, м"]),
                'alt': None if pd.isna(row["Альтитуда, м"]) else float(row["Альтитуда, м"]),
            }
            w['active'] = overrides[w['file']]['active']
            w['x'], w['y'] = overrides[w['file']]['x'], overrides[w['file']]['y']
            w['alt'] = overrides[w['file']]['alt']

        with st.sidebar.expander("Распознанные методы", expanded=False):
            summary_rows = []
            for w in wells_data:
                mapped_str = ", ".join([f"{method_label(k)} ({v})" if method_label(k) != v else method_label(k)
                                        for k, v in w['curve_map'].items()])
                summary_rows.append({"Скважина": w['name'], "Шаг, м": round(w['step'], 3), "Методы": mapped_str})
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

        # отбивки пластов
        if (st.session_state.tops_df is None and not st.session_state.tops_autoloaded
                and os.environ.get("GEO_AUTOLOAD_DIR")):
            st.session_state.tops_autoloaded = True
            auto_dir = os.environ["GEO_AUTOLOAD_DIR"]
            for f in sorted(os.listdir(auto_dir)):
                if f.lower().endswith('.csv'):
                    try:
                        st.session_state.tops_df = parse_tops_table(f, open(os.path.join(auto_dir, f), 'rb').read())
                        st.session_state.tops_ver += 1
                    except Exception:
                        pass
                    break
        tops_base = st.session_state.tops_df if st.session_state.tops_df is not None else empty_tops()
        with st.sidebar.expander(f"Отбивки пластов ({len(tops_base)} строк)", expanded=False):
            st.caption("Скважина, пласт, кровля и подошва — глубины по стволу, как в LAS. Нужны для режима "
                       "«По пластам». Колонка «Насыщение» будет учитываться в расчете насыщенности (готовится).")
            tops_file = st.file_uploader("Загрузить таблицу (CSV или Excel)", type=['csv', 'txt', 'xlsx', 'xls'],
                                         key=f"tops_up_{st.session_state.tops_ver}")
            if tops_file is not None:
                try:
                    st.session_state.tops_df = parse_tops_table(tops_file.name, tops_file.getvalue())
                    st.session_state.tops_ver += 1
                    st.rerun()
                except ImportError:
                    st.error("Для чтения Excel нужна библиотека openpyxl. Сохраните таблицу как CSV.")
                except Exception as exc:
                    st.error(f"Не удалось прочитать таблицу: {exc}")
            tops_edit = st.data_editor(
                tops_base, key=f"tops_editor_{st.session_state.tops_ver}", num_rows="dynamic", hide_index=True,
                use_container_width=True,
                column_config={
                    "Кровля, м": st.column_config.NumberColumn(format="%.1f"),
                    "Подошва, м": st.column_config.NumberColumn(format="%.1f"),
                    "Насыщение": st.column_config.SelectboxColumn(options=SATURATIONS, default="вода"),
                })
            st.session_state.tops_current = tops_edit
            template = pd.DataFrame({"Скважина": [w['name'] for w in wells_data], "Пласт": "",
                                     "Кровля, м": np.nan, "Подошва, м": np.nan, "Насыщение": "вода"})
            st.download_button("Скачать шаблон таблицы", data=template.to_csv(sep=';', index=False).encode('utf-8-sig'),
                               file_name="Отбивки_шаблон.csv", mime="text/csv", use_container_width=True)
            _, unknown = match_tops(tops_edit, wells_data)
            if unknown:
                st.warning("Нет среди загруженных скважин: " + ", ".join(unknown) +
                           ". Названия должны совпадать с именем скважины или файла.")

    if len(wells_data) < 2:
        st.markdown("#### С чего начать")
        s1, s2, s3 = st.columns(3)
        s1.markdown('<div class="geo-step"><b>1. Загрузите LAS</b><br>Файлы опорных скважин (от двух) в панели слева. '
                    'Координаты устья возьмутся из шапки файла.</div>', unsafe_allow_html=True)
        s2.markdown('<div class="geo-step"><b>2. Расставьте скважины</b><br>Щелчком по карте, регулярной сеткой или '
                    'профилем по линии. Результат сразу виден на планшете.</div>', unsafe_allow_html=True)
        s3.markdown('<div class="geo-step"><b>3. Скачайте LAS</b><br>Архив со всеми синтетическими скважинами или '
                    'сразу несколько вариантов для группы студентов.</div>', unsafe_allow_html=True)
        st.write("")
        with st.expander("Справка", expanded=False):
            st.markdown(HELP_MD)
        st.stop()

    active_wells = [w for w in wells_data if w.get('active', True)]
    if len(active_wells) < 1:
        st.warning("Включите хотя бы одну опорную скважину в таблице «Опорные скважины».")
        st.stop()

    # ------------------------------------------
    # САЙДБАР: ИНТЕРВАЛ, ШАГ, ИЗМЕНЧИВОСТЬ
    # ------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("2. Режим расчета и глубины")
    mode = st.sidebar.radio(
        "Режим расчета", [MODE_INTERVAL, MODE_LAYERS], key='geo_mode',
        help="«По интервалу глубин»: значения складываются с одинаковой глубины в заданном окне. "
             "«По пластам»: по таблице отбивок кровля и подошва пласта в каждой точке считаются IDW, "
             "а внутри пласта значения складываются по одинаковому положению (кровля с кровлей, середина "
             "с серединой). Так пласт не обрезается, даже если в скважинах он на разной глубине и разной мощности.")

    no_alt = [w['name'] for w in active_wells if w.get('alt') is None]
    abs_available = not no_alt
    use_abs = st.sidebar.toggle(
        "Учитывать альтитуду (абсолютные отметки)", key='geo_abs', disabled=not abs_available,
        help="Скважины считаются вертикальными: абсолютная глубина = глубина по стволу − альтитуда. "
             "Сопоставление скважин и интерполяция кровли и подошвы идут по абсолютным отметкам. "
             "В выгружаемых LAS остается глубина по стволу.") and abs_available
    if not abs_available:
        st.sidebar.caption(f"Альтитуда не задана: {', '.join(no_alt)}. Глубины считаются по стволу. "
                           "Альтитуду можно ввести в таблице «Опорные скважины».")
    if st.session_state.last_abs is not None and st.session_state.last_abs != use_abs:
        for k in ('geo_top', 'geo_bot'):
            st.session_state.pop(k, None)
    st.session_state.last_abs = use_abs
    for w in active_wells:
        w['shift'] = float(w['alt']) if use_abs else 0.0
        w['zdepth'] = w['depth'] - w['shift']
    shifts = {k: w['shift'] for k, w in enumerate(active_wells)}
    unit_txt = "м абс." if use_abs else "м"
    y_title = "Абсолютная глубина (ниже уровня моря), м" if use_abs else "Глубина по стволу, м"

    min_src_step = min(w['step'] for w in active_wells)
    def_step = STEP_OPTIONS[int(np.argmin([abs(s - min_src_step) for s in STEP_OPTIONS]))]
    if st.session_state.get('geo_step') not in STEP_OPTIONS:
        st.session_state.geo_step = def_step

    layer_mode, zones = False, []
    tops_map, _ = match_tops(st.session_state.tops_current, active_wells)
    if mode == MODE_LAYERS:
        layers_all = layers_in_order(tops_map, shifts)
        if not layers_all:
            st.sidebar.warning("Для режима «По пластам» нужна таблица отбивок: загрузите ее в разделе "
                               "«Отбивки пластов» выше. Пока расчет идет по интервалу глубин.")
        else:
            layer_mode = True
            _sanitize('geo_layers', layers_all, multi=True)
            if not st.session_state.get('geo_layers'):
                st.session_state.geo_layers = [layers_all[0]]
            sel_layers = st.sidebar.multiselect(
                "Пласты", layers_all, key='geo_layers',
                help="Можно выбрать несколько пластов: интервалы между соседними выбранными пластами "
                     "тоже рассчитываются пропорционально.")
            sel_layers = [n for n in layers_all if n in sel_layers]
            if not sel_layers:
                st.sidebar.error("Выберите хотя бы один пласт.")
                st.stop()
            counts = {n: sum(1 for lay in tops_map.values() if n in lay) for n in sel_layers}
            st.sidebar.caption("Скважин с отбивками: " + ", ".join(f"{n} — {c}" for n, c in counts.items()))
            margin = st.sidebar.number_input("Показывать выше и ниже пласта, м", min_value=0.0, max_value=200.0,
                                             value=float(st.session_state.get('geo_margin_init', 5.0)),
                                             step=1.0, key='geo_margin')
            zones = build_zones(tops_map, sel_layers, shifts)

    depth_step = None
    if layer_mode:
        depth_step = st.sidebar.selectbox(
            "Шаг по глубине (м)", STEP_OPTIONS, key='geo_step', format_func=lambda s: f"{s:.1f}".replace(".", ","),
            help="По умолчанию самый мелкий шаг записи среди загруженных файлов. Этот же шаг будет в выгружаемых LAS.")
        z_lo = min(b[0] for z in zones for b in z['bounds'].values()) - margin
        z_hi = max(b[1] for z in zones for b in z['bounds'].values()) + margin
        top_depth = float(np.floor(z_lo / depth_step) * depth_step)
        bot_depth = float(np.ceil(z_hi / depth_step) * depth_step)
    else:
        mins_d = [float(w['zdepth'].min()) for w in active_wells]
        maxs_d = [float(w['zdepth'].max()) for w in active_wells]
        global_min_d, global_max_d = float(min(mins_d)), float(max(maxs_d))
        overlap_min, overlap_max = float(max(mins_d)), float(min(maxs_d))
        if overlap_min < overlap_max:
            def_top = float(round(overlap_min + 5.0, 1))
            def_bot = float(round(min(def_top + 35.0, overlap_max), 1))
        else:
            def_top = float(round(global_min_d, 1))
            def_bot = float(round(min(def_top + 35.0, global_max_d), 1))
        for key, default in (('geo_top', def_top), ('geo_bot', def_bot)):
            val = st.session_state.get(key, default)
            st.session_state[key] = float(min(max(float(val), global_min_d), global_max_d))

        st.sidebar.caption("Это окно глубин, в котором строится синтетика: укажите глубины интересующего пласта. "
                           "Начальные значения программа подставляет сама (начало общего для всех файлов интервала "
                           "+ 5 м, длина 35 м), с геологией они не связаны. Если есть отбивки, удобнее режим "
                           "«По пластам».")
        top_depth = st.sidebar.number_input(
            f"Кровля интервала ({unit_txt})", min_value=global_min_d, max_value=global_max_d, step=1.0,
            key='geo_top', help="Интервал общий для всех скважин и не зависит от положения точки на карте.")
        bot_depth = st.sidebar.number_input(
            f"Подошва интервала ({unit_txt})", min_value=global_min_d, max_value=global_max_d, step=1.0,
            key='geo_bot')
        depth_step = st.sidebar.selectbox(
            "Шаг по глубине (м)", STEP_OPTIONS, key='geo_step', format_func=lambda s: f"{s:.1f}".replace(".", ","),
            help="По умолчанию самый мелкий шаг записи среди загруженных файлов. Этот же шаг будет в выгружаемых LAS.")
        if top_depth >= bot_depth:
            st.sidebar.error("Кровля интервала должна быть меньше подошвы.")
            st.stop()
    n_depth = int(round((bot_depth - top_depth) / depth_step)) + 1
    if n_depth > 60000:
        st.sidebar.error("Слишком длинный интервал для выбранного шага. Увеличьте шаг или сократите интервал.")
        st.stop()
    depth_axis = np.round(top_depth + np.arange(n_depth) * depth_step, 4)

    st.sidebar.markdown("---")
    st.sidebar.header("3. Изменчивость кривых")
    cvae_influence = st.sidebar.slider(
        "Степень вариативности", 0.0, 1.0, step=0.05, key='geo_infl',
        help="Насколько сильно к интерполированной кривой добавляется случайная составляющая. 0 — чистая интерполяция.")
    base_seed = st.sidebar.number_input(
        "Номер варианта (seed)", step=1, key='geo_seed',
        help="При одном и том же номере результат повторяется. Другой номер — новый вариант кривых.")
    with st.sidebar.expander("Дополнительно", expanded=False):
        sigma_val = st.slider(
            "Sigma латентного пространства", 0.1, 2.5, step=0.05, key='geo_sigma',
            help="Чем больше, тем менее типична форма изменчивости, которую дает нейросеть (ГК и ПС).")
        max_gap = st.number_input(
            "Заполнять пропуски до, м", min_value=0.0, max_value=20.0, step=0.1, key='geo_gap',
            help="Более длинные пропуски не заполняются: на этих глубинах скважина не участвует в расчете.")
        max_gap_ak = st.number_input("Для АК — до, м", min_value=0.0, max_value=50.0, step=0.5, key='geo_gap_ak')
        correlated = st.toggle(
            "Согласовать шум с ГК", key='geo_corr',
            help="Случайная составляющая остальных методов связывается с ГК с тем же коэффициентом корреляции, "
                 "что и в опорных скважинах на интервале.")

    # ------------------------------------------
    # ОПОРНЫЕ ДАННЫЕ НА СЕТКЕ, МОДЕЛЬ
    # ------------------------------------------
    ref = build_reference(active_wells, depth_axis, max_gap, max_gap_ak)
    all_available_methods = ref['methods']
    if st.session_state.get("cvae_ready"):
        cvae_net, cvae_meta = load_cvae()
    else:
        with st.spinner("Загружаем нейросетевую модель (при первом запуске 10–20 секунд)..."):
            cvae_net, cvae_meta = load_cvae()
        st.session_state.cvae_ready = True
    params = {'net': cvae_net, 'meta': cvae_meta, 'seed': base_seed, 'sigma': sigma_val,
              'influence': cvae_influence, 'correlated': correlated,
              'alts': np.array([w['alt'] for w in active_wells], dtype=float) if use_abs else None}

    def run_synth(points_, seed_offset=0):
        if layer_mode:
            return synthesize_layers(points_, active_wells, ref, zones, depth_axis, params, seed_offset)
        return synthesize(points_, active_wells, ref, n_depth, params, seed_offset)

    def ref_tops(k):
        """Отбивки опорной скважины (индекс в active_wells) в рабочей системе глубин."""
        return [{'name': z['name'], 'kind': z['kind'], 'top': z['bounds'][k][0], 'bot': z['bounds'][k][1]}
                for z in zones if k in z['bounds']]

    def point_tops(px, py):
        """Кровля и подошва зон и альтитуда в проектной точке (без расчета кривых)."""
        raw_w = idw_raw_weights(active_wells, px, py)
        alt_p = float((raw_w / raw_w.sum() * params['alts']).sum()) if use_abs else 0.0
        return [(z['name'], z['kind']) + zone_bounds_at(raw_w, z) for z in zones], alt_p

    # выбранный объект должен существовать
    sel = st.session_state.selected
    if sel:
        exists = (any(p['id'] == sel['id'] for p in st.session_state.points_list) if sel['type'] == 'point'
                  else any(w['file'] == sel['id'] for w in wells_data))
        if not exists:
            st.session_state.selected = None
            st.session_state.move_mode = False

    # координаты карты: плоская система, единица карты = map_unit метров
    all_xs = [w['x'] for w in wells_data]
    all_ys = [w['y'] for w in wells_data]
    min_x, max_x = float(min(all_xs)), float(max(all_xs))
    min_y, max_y = float(min(all_ys)), float(max(all_ys))
    span = max(max_x - min_x, max_y - min_y, 600.0)
    origin_x, origin_y = min_x, min_y
    map_unit = span / 1000.0

    def m_to_map(xm, ym):
        return [(ym - origin_y) / map_unit, (xm - origin_x) / map_unit]

    def map_to_m(lat, lng):
        return float(lng * map_unit + origin_x), float(lat * map_unit + origin_y)

    chips = [
        f"Опорных скважин: <b>{len(active_wells)}</b> из {len(wells_data)}",
        f"Методов: <b>{len(all_available_methods)}</b>",
        ("Режим: <b>по пластам</b> (" + ", ".join(z['name'] for z in zones if z['kind'] == 'layer') + ")"
         if layer_mode else "Режим: <b>по интервалу глубин</b>"),
        f"Глубины: <b>{'абсолютные' if use_abs else 'по стволу'}</b>",
        f"Интервал: <b>{fmt_num(top_depth)}–{fmt_num(bot_depth)} {unit_txt}</b>",
        f"Шаг: <b>{fmt_num(depth_step)} м</b> ({n_depth} точек)",
        f"Синтетических скважин: <b>{sum(1 for p in st.session_state.points_list if p.get('active', True))}</b>",
    ]
    st.markdown('<div class="geo-chips">' + "".join(f'<span class="geo-chip">{c}</span>' for c in chips) + '</div>',
                unsafe_allow_html=True)

    tab_map, tab_logs, tab_export, tab_help = st.tabs([
        "Карта и расстановка точек", "Каротажный планшет", "Сводка и экспорт LAS", "Справка"
    ])

    # ------------------------------------------
    # ВКЛАДКА 1: КАРТА
    # ------------------------------------------
    with tab_map:
        _apply_points_editor()
        points = st.session_state.points_list
        t1, t2, t3, t4, t5 = st.columns([2.4, 1.25, 1.0, 1.15, 1.25])
        with t1:
            st.caption("Щелчок по пустому месту ставит скважину, щелчок по точке или опорной скважине выбирает ее. "
                       "Перемещение карты — зажатая левая кнопка, масштаб — колесико.")
        with t2:
            click_mode = st.toggle("Добавлять точку по щелчку", key="click_mode",
                                   help="Выключите, чтобы случайные щелчки по карте не добавляли точки.")
        with t3:
            show_labels = st.toggle("Подписи точек", key="show_labels")
        with t4:
            if st.button("Отменить действие", use_container_width=True, disabled=not st.session_state.history,
                         help="Возвращает точки к состоянию до последнего добавления, удаления, перемещения, "
                              "сетки или профиля."):
                _set_points(st.session_state.history.pop())
                st.rerun()
        with t5:
            if st.button("Сфокусировать на скважинах", use_container_width=True):
                st.session_state.map_key_version += 1
                st.rerun()

        sel = st.session_state.selected
        if st.session_state.move_mode and sel:
            st.info(f"Щелкните на карте новое место для {sel['id']}. Чтобы отменить, нажмите «Отменить перемещение» "
                    "в карточке справа.")

        col_map, col_card = st.columns([3.2, 1.3])

        # базовая карта: сетка и опорные скважины (не перерисовывается при изменении точек)
        pad = span * 0.25
        m = folium.Map(crs='Simple', tiles=None, attribution_control=False, zoom_control=True)
        # для плоской системы координат folium не передает пределы масштаба, задаем их напрямую
        m.options.update({'minZoom': -4, 'maxZoom': 8, 'zoomSnap': 0.25, 'zoomDelta': 0.5})
        m.get_root().header.add_child(Element(MAP_CSS))
        m.fit_bounds([m_to_map(min_x - pad, min_y - pad), m_to_map(max_x + pad, max_y + pad)])
        Graticule(map_unit, origin_x, origin_y).add_to(m)

        for w in wells_data:
            loc = m_to_map(w['x'], w['y'])
            if w.get('active', True):
                folium.CircleMarker(location=loc, radius=9, color="#0F172A", weight=2, fill=True,
                                    fill_color=w['color'], fill_opacity=1.0,
                                    tooltip=f"{w['name']} (X={w['x']:.0f} м, Y={w['y']:.0f} м)").add_to(m)
                folium.Marker(location=loc, icon=folium.DivIcon(
                    html=f'<div style="font-size: 11px; font-weight: bold; color: #0F172A; margin-top: -24px; '
                         f'margin-left: -14px; white-space: nowrap; text-shadow: 0 0 2px #fff;">{w["name"]}</div>')
                ).add_to(m)
            else:
                folium.CircleMarker(location=loc, radius=6, color="#94A3B8", weight=1, fill=True,
                                    fill_color="#E2E8F0", fill_opacity=0.6,
                                    tooltip=f"{w['name']} (исключена из расчета)").add_to(m)

        # слой точек и выделения: при изменениях карта не перерисовывается и не сбрасывает масштаб
        layer = folium.FeatureGroup(name="Синтетические скважины")
        sel_point = next((p for p in points if sel and sel['type'] == 'point' and p['id'] == sel['id']), None)
        sel_well = next((w for w in wells_data if sel and sel['type'] == 'well' and w['file'] == sel['id']), None)

        if sel_point is not None:
            raw_w = idw_raw_weights(active_wells, sel_point['x'], sel_point['y'])
            for w, wv in zip(active_wells, raw_w / raw_w.sum()):
                a, b = m_to_map(sel_point['x'], sel_point['y']), m_to_map(w['x'], w['y'])
                folium.PolyLine([a, b], color=w['color'], weight=1.5 + 7 * wv, opacity=0.65,
                                dash_array="6, 5", interactive=False).add_to(layer)
                mid = [a[0] + (b[0] - a[0]) * 0.55, a[1] + (b[1] - a[1]) * 0.55]
                folium.Marker(location=mid, interactive=False, icon=folium.DivIcon(
                    icon_size=(0, 0), html=f'<div class="geo-wlbl" style="color:{w["color"]}">{wv:.0%}</div>')
                ).add_to(layer)
        if sel_well is not None:
            folium.Marker(location=m_to_map(sel_well['x'], sel_well['y']), interactive=False,
                          icon=folium.DivIcon(icon_size=(0, 0), html='<div class="geo-well-ring"></div>')
                          ).add_to(layer)

        for s in points:
            if not s.get('active', True):
                continue
            is_sel = sel_point is not None and s['id'] == sel_point['id']
            label = f'<div class="geo-lbl" style="color:{s["color"]};">{s["id"]}</div>' if (show_labels or is_sel) else ""
            if is_sel:
                icon = folium.DivIcon(icon_size=(16, 16), html=(
                    f'<div class="geo-sel-wrap"><div class="geo-ring" style="border-color:{s["color"]};"></div>'
                    f'<div class="geo-diamond" style="background:{s["color"]};"></div></div>{label}'))
            else:
                icon = folium.DivIcon(html=f'<div class="geo-diamond" style="background:{s["color"]};"></div>{label}')
            folium.Marker(location=m_to_map(s['x'], s['y']), icon=icon,
                          tooltip=f"{s['id']} (X={s['x']:.1f} м, Y={s['y']:.1f} м)").add_to(layer)

        with col_map:
            map_key = f"folium_map_v_{st.session_state.map_key_version}"
            map_out = st_folium(m, height=600, use_container_width=True, key=map_key,
                                feature_group_to_add=layer,
                                returned_objects=["last_clicked", "last_object_clicked", "last_object_clicked_count"])

        # обработка щелчков
        map_out = map_out or {}
        obj, obj_cnt, clk = map_out.get("last_object_clicked"), map_out.get("last_object_clicked_count"), \
            map_out.get("last_clicked")
        new_obj = obj is not None and obj_cnt != st.session_state.last_obj_count
        new_clk = clk is not None and clk != st.session_state.last_click
        st.session_state.last_obj_count = obj_cnt
        st.session_state.last_click = clk

        def _hit(o):
            if not o:
                return None
            for s in points:
                if s.get('active', True):
                    lat, lng = m_to_map(s['x'], s['y'])
                    if abs(lat - o['lat']) < 1e-6 * (1 + abs(lat)) and abs(lng - o['lng']) < 1e-6 * (1 + abs(lng)):
                        return ('point', s['id'])
            for w in wells_data:
                lat, lng = m_to_map(w['x'], w['y'])
                if abs(lat - o['lat']) < 1e-6 * (1 + abs(lat)) and abs(lng - o['lng']) < 1e-6 * (1 + abs(lng)):
                    return ('well', w['file'])
            return None

        hit = _hit(obj) if new_obj else None
        if hit:
            _select(*hit)
            st.rerun()
        elif new_clk:
            cx, cy = map_to_m(clk["lat"], clk["lng"])
            tol = max(span * 0.01, 3.0)
            near_pt = next((p for p in points if p.get('active', True)
                            and np.hypot(p['x'] - cx, p['y'] - cy) < tol), None)
            near_well = next((w for w in wells_data if np.hypot(w['x'] - cx, w['y'] - cy) < tol), None)
            if st.session_state.move_mode and sel_point is not None:
                moved = [dict(p, x=float(round(cx, 1)), y=float(round(cy, 1))) if p['id'] == sel_point['id'] else p
                         for p in points]
                st.session_state.move_mode = False
                _commit_points(moved)
                st.rerun()
            elif near_pt is not None:
                _select('point', near_pt['id'])
                st.rerun()
            elif near_well is not None:
                _select('well', near_well['file'])
                st.rerun()
            elif click_mode:
                name, n = _free_name(points, "SYNTH")
                _commit_points(points + [_make_point(cx, cy, name, n)])
                _select('point', name)
                st.rerun()
            elif sel:
                st.session_state.selected = None
                st.rerun()

        # карточка выбранного объекта
        with col_card:
            if sel_point is not None:
                p = sel_point
                st.markdown(f'<div class="geo-card-title"><span class="geo-dot" style="background:{p["color"]};'
                            f'width:12px;height:12px;"></span>{p["id"]}</div>'
                            f'<div class="geo-card-sub">Синтетическая скважина</div>', unsafe_allow_html=True)
                with st.form(f"card_point_{p['id']}_{st.session_state.points_ver}"):
                    new_name = st.text_input("Название", value=p['id'])
                    fx = st.number_input("X, м", value=float(p['x']), step=10.0, format="%.1f")
                    fy = st.number_input("Y, м", value=float(p['y']), step=10.0, format="%.1f")
                    f_active = st.checkbox("Участвует в расчете", value=p.get('active', True))
                    if st.form_submit_button("Применить", use_container_width=True, type="primary"):
                        new_name = new_name.strip() or p['id']
                        if new_name != p['id'] and any(q['id'] == new_name for q in points):
                            st.error("Точка с таким названием уже есть.")
                        else:
                            _commit_points([dict(q, id=new_name, x=float(fx), y=float(fy), active=f_active)
                                            if q['id'] == p['id'] else q for q in points])
                            _select('point', new_name)
                            st.rerun()
                b1, b2 = st.columns(2)
                with b1:
                    if st.session_state.move_mode:
                        if st.button("Отменить перемещение", use_container_width=True):
                            st.session_state.move_mode = False
                            st.rerun()
                    elif st.button("Переместить", use_container_width=True,
                                   help="После нажатия щелкните на карте новое место точки."):
                        st.session_state.move_mode = True
                        st.rerun()
                with b2:
                    if st.button("Удалить", use_container_width=True):
                        _commit_points([q for q in points if q['id'] != p['id']])
                        st.session_state.selected = None
                        st.rerun()
                if st.button("Снять выбор", use_container_width=True):
                    st.session_state.selected = None
                    st.session_state.move_mode = False
                    st.rerun()
                raw_w = idw_raw_weights(active_wells, p['x'], p['y'])
                st.markdown('<div style="font-size:12px;color:#64748B;margin-top:6px;">Веса IDW:</div>' +
                            _badges([(w['name'], w['color'], v) for w, v in zip(active_wells, raw_w / raw_w.sum())]),
                            unsafe_allow_html=True)
                if layer_mode or use_abs:
                    p_tops, p_alt = point_tops(p['x'], p['y'])
                    lines = []
                    if use_abs:
                        lines.append(f"- Альтитуда (IDW): **{fmt_num(p_alt)} м**")
                    for name, kind, zt, zb in p_tops:
                        if kind == 'layer':
                            lines.append(f"- {name}: **{fmt_num(zt + p_alt)}–{fmt_num(zb + p_alt)} м** "
                                         f"по стволу, мощность {fmt_num(zb - zt)} м")
                    if lines:
                        st.markdown("\n".join(lines))
                st.caption("Кривые этой скважины открыты на вкладке «Каротажный планшет».")
            elif sel_well is not None:
                w = sel_well
                st.markdown(f'<div class="geo-card-title"><span class="geo-dot" style="background:{w["color"]};'
                            f'width:12px;height:12px;"></span>{w["name"]}</div>'
                            f'<div class="geo-card-sub">Опорная скважина · файл {w["file"]}</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    f"- Запись: **{fmt_num(w['depth'].min())}–{fmt_num(w['depth'].max())} м**, шаг {w['step']:.2f} м\n"
                    f"- Методы: {', '.join(method_label(k) for k in w['curve_map'])}\n"
                    f"- Координаты: {'из файла' if w['coords_found'] else '**условные** (в файле нет)'}\n"
                    f"- Альтитуда: " + (f"**{fmt_num(w['alt'])} м**" + (f" ({w['alt_key']})" if w.get('alt_key') else "")
                                        if w.get('alt') is not None else "не задана"))
                k_w = next((k for k, aw in enumerate(active_wells) if aw['file'] == w['file']), None)
                w_tops = tops_map.get(k_w, {}) if k_w is not None else {}
                if w_tops:
                    st.markdown("**Отбивки:** " + "; ".join(f"{n} {fmt_num(t)}–{fmt_num(b)} м"
                                                            for n, (t, b, _) in w_tops.items()))
                gaps_w = [g for g in ref['gaps'] if g['Скважина'] == w['name']]
                if gaps_w and w.get('active', True):
                    st.markdown("**Нет данных на интервале:**")
                    for g in gaps_w:
                        st.markdown(f"- {g['Метод']}: {g['Интервалы, м']} м")
                with st.form(f"card_well_{w['file']}_{st.session_state.ref_ver}"):
                    fx = st.number_input("X, м", value=float(w['x']), step=10.0, format="%.1f")
                    fy = st.number_input("Y, м", value=float(w['y']), step=10.0, format="%.1f")
                    falt = st.number_input("Альтитуда, м", value=None if w.get('alt') is None else float(w['alt']),
                                           step=0.1, format="%.1f", placeholder="не задана")
                    f_active = st.checkbox("Участвует в расчете", value=w.get('active', True))
                    if st.form_submit_button("Применить", use_container_width=True, type="primary"):
                        st.session_state.ref_overrides[w['file']] = {
                            'active': f_active, 'x': float(fx), 'y': float(fy),
                            'alt': None if falt is None else float(falt)}
                        st.session_state.ref_ver += 1
                        st.rerun()
                if st.button("Снять выбор", use_container_width=True):
                    st.session_state.selected = None
                    st.rerun()
            else:
                st.markdown('<div class="geo-card-title">Ничего не выбрано</div>', unsafe_allow_html=True)
                st.caption("Щелкните по синтетической точке или опорной скважине на карте: откроется ее карточка. "
                           "В ней можно изменить название и координаты, удалить точку или переместить ее.")
                st.caption("Линии от выбранной точки к опорным скважинам показывают веса IDW.")

        # управление точками
        st.markdown("")
        c1, c2, c3 = st.columns([1.3, 1.0, 1.0])
        with c1:
            def_grid = float(np.clip(_nice_step(span, target=4), 50.0, 800.0))
            step_val = st.number_input("Шаг регулярной сетки (м)", min_value=50.0, max_value=800.0,
                                       value=def_grid, step=50.0)
        with c2:
            st.write("")
            st.write("")
            if st.button("Построить регулярную сетку", use_container_width=True,
                         help="Заменяет текущие точки сеткой в прямоугольнике, охватывающем опорные скважины."):
                coords = [(gx, gy) for gx in np.arange(min_x, max_x + 1.0, step_val)
                          for gy in np.arange(min_y, max_y + 1.0, step_val)]
                if len(coords) > 400:
                    st.warning(f"Получится {len(coords)} точек, это слишком много. Увеличьте шаг сетки.")
                else:
                    _commit_points([_make_point(gx, gy, f"GRID_{i}", i) for i, (gx, gy) in enumerate(coords, 1)])
                    st.session_state.selected = None
                    if len(coords) > 30:
                        st.session_state.labels_off_pending = True
                    st.rerun()
        with c3:
            st.write("")
            st.write("")
            if st.button("Очистить все точки", use_container_width=True, disabled=not points):
                _commit_points([])
                st.session_state.selected = None
                st.rerun()

        with st.expander("Профиль скважин по линии (для схемы корреляции)", expanded=False):
            st.caption("Синтетические скважины ставятся с равным шагом на отрезке между двумя объектами или координатами. "
                       "Удобно для заданий на корреляцию: на планшете выберите «Схема корреляции».")
            objects = [(f"скв. {w['name']} (опорная)", w['x'], w['y']) for w in active_wells] + \
                      [(f"{p['id']}", p['x'], p['y']) for p in points] + [("координаты вручную", None, None)]
            labels = [o[0] for o in objects]
            pa, pb = st.columns(2)
            ends = []
            for col, title, idx_def, key in ((pa, "Начало профиля", 0, "prof_a"),
                                             (pb, "Конец профиля", min(1, len(labels) - 2), "prof_b")):
                with col:
                    _sanitize(key, labels)
                    choice = st.selectbox(title, labels, index=idx_def, key=key)
                    o = objects[labels.index(choice)]
                    if o[1] is None:
                        ex = st.number_input("X, м", value=float(min_x), step=10.0, format="%.1f", key=f"{key}_x")
                        ey = st.number_input("Y, м", value=float(min_y), step=10.0, format="%.1f", key=f"{key}_y")
                        ends.append((ex, ey))
                    else:
                        ends.append((o[1], o[2]))
            pn, pe = st.columns(2)
            with pn:
                n_prof = st.number_input("Число скважин на профиле", min_value=2, max_value=50, value=8, step=1)
            with pe:
                st.write("")
                with_ends = st.checkbox("Ставить точки и на концах профиля", value=False)
            if st.button("Добавить профиль", use_container_width=True, type="primary"):
                (xa, ya), (xb, yb) = ends
                if np.hypot(xb - xa, yb - ya) < 1.0:
                    st.warning("Начало и конец профиля совпадают.")
                else:
                    ts = np.linspace(0, 1, int(n_prof)) if with_ends else np.linspace(0, 1, int(n_prof) + 2)[1:-1]
                    new_pts, cur = [], list(points)
                    for t in ts:
                        name, n = _free_name(cur, "PROF")
                        pt = _make_point(xa + (xb - xa) * t, ya + (yb - ya) * t, name, n)
                        cur.append(pt)
                        new_pts.append(pt)
                    _commit_points(points + new_pts)
                    st.rerun()

        if points:
            st.markdown(f"**Синтетические скважины ({len(points)})**")
            st.caption("Координаты и название можно исправить прямо в таблице. Флажок «Активна» временно исключает точку.")
            base_df = pd.DataFrame([{"Активна": p['active'], "Скважина": p['id'], "X, м": p['x'], "Y, м": p['y']}
                                    for p in st.session_state.points_base])
            edited_pts = st.data_editor(
                base_df, key=f"pts_editor_{st.session_state.points_ver}", hide_index=True,
                use_container_width=True, num_rows="fixed",
                column_config={"X, м": st.column_config.NumberColumn(format="%.1f"),
                               "Y, м": st.column_config.NumberColumn(format="%.1f")})
            new_list, seen = [], set()
            for base_p, (_, row) in zip(st.session_state.points_base, edited_pts.iterrows()):
                name = str(row["Скважина"]).strip() or base_p['id']
                while name in seen:
                    name += "_2"
                seen.add(name)
                new_list.append({'id': name,
                                 'x': base_p['x'] if pd.isna(row["X, м"]) else float(row["X, м"]),
                                 'y': base_p['y'] if pd.isna(row["Y, м"]) else float(row["Y, м"]),
                                 'active': bool(row["Активна"]), 'color': base_p['color']})
            if sel and sel['type'] == 'point':
                for base_p, new_p in zip(st.session_state.points_base, new_list):
                    if base_p['id'] == sel['id'] and new_p['id'] != base_p['id']:
                        st.session_state.selected = {'type': 'point', 'id': new_p['id']}
            st.session_state.points_list = new_list

            d1, d2 = st.columns([3, 1])
            with d1:
                to_delete = st.multiselect("Удалить точки", [p['id'] for p in new_list],
                                           placeholder="Выберите точки для удаления",
                                           key=f"del_sel_{st.session_state.points_ver}")
            with d2:
                st.write("")
                st.write("")
                if st.button("Удалить выбранные", use_container_width=True, disabled=not to_delete):
                    _commit_points([p for p in new_list if p['id'] not in to_delete])
                    st.rerun()

    # ------------------------------------------
    # РАСЧЕТ СИНТЕТИКИ
    # ------------------------------------------
    active_points = [p for p in st.session_state.points_list if p.get('active', True)]
    generated_wells = run_synth(active_points)

    # ------------------------------------------
    # ВКЛАДКА 2: КАРОТАЖНЫЙ ПЛАНШЕТ
    # ------------------------------------------
    with tab_logs:
        if not generated_wells:
            st.info("Поставьте хотя бы одну синтетическую скважину на вкладке «Карта и расстановка точек».")
        else:
            ids = [g['id'] for g in generated_wells]
            gen_by_id = {g['id']: g for g in generated_wells}
            view = st.radio("Вид", ["Одна скважина", "Схема корреляции"], horizontal=True,
                            label_visibility="collapsed", key="plan_view")

            if ref['gaps'] or ref['uncovered']:
                msg = []
                if ref['gaps']:
                    msg.append(f"в {len({g['Скважина'] for g in ref['gaps']})} скв. есть интервалы без данных — "
                               "на этих глубинах синтетика считается по остальным скважинам")
                if ref['uncovered']:
                    msg.append("по некоторым методам данных нет ни в одной скважине — там синтетика не строится")
                with st.expander("Внимание: " + "; ".join(msg) + ".", expanded=False):
                    if ref['gaps']:
                        st.dataframe(pd.DataFrame(ref['gaps']), hide_index=True, use_container_width=True)
                    for mm, runs in ref['uncovered'].items():
                        st.markdown(f"- **{method_label(mm)}**: синтетика не строится на {runs_to_text(runs)} м")
                    st.caption("Если пропусков слишком много, замените LAS-файл или исключите скважину в таблице слева.")

            if view == "Одна скважина":
                if st.session_state.get('planchette_sel') not in ids:
                    st.session_state.planchette_sel = ids[0]
                c_top1, c_top2 = st.columns([1.2, 2.0])
                with c_top1:
                    selected_id = st.selectbox(
                        "Синтетическая скважина:", options=ids, key="planchette_sel",
                        format_func=lambda i: f"{i} (X={gen_by_id[i]['x']:.1f} м, Y={gen_by_id[i]['y']:.1f} м)")
                with c_top2:
                    default_display = [mm for mm in ['GK', 'PS', 'IK', 'BK'] if mm in all_available_methods] \
                        or all_available_methods[:min(4, len(all_available_methods))]
                    display_curves = st.multiselect("Треки (методы):", options=all_available_methods,
                                                    default=default_display, format_func=method_label)
                curr = gen_by_id[selected_id]

                o1, o2 = st.columns([1, 2.2])
                with o1:
                    show_ref = st.toggle("Показывать опорные скважины", value=True)
                with o2:
                    compare_ids = st.multiselect("Сравнить с другими синтетическими (до 3):",
                                                 [i for i in ids if i != selected_id], max_selections=3)

                st.markdown('<div class="geo-badges"><span style="font-size:12px; font-weight:600; color:#64748B; '
                            'margin-right:8px;" title="На глубинах, где у части скважин нет данных, веса '
                            'пересчитываются между остальными скважинами.">Веса IDW:</span>' +
                            _badges([(w['name'], w['color'], v) for w, v in zip(active_wells, curr['weights'])]) +
                            '</div>', unsafe_allow_html=True)

                if display_curves:
                    tracks = []
                    with st.expander("Настройка треков: цвет, второй метод, шкала", expanded=False):
                        hdr = st.columns([1.0, 1.4, 0.7, 0.7, 1.1, 1.0, 1.0])
                        for col, txt in zip(hdr, ["Трек", "Второй метод", "Цвет", "Цвет 2", "Шкала", "Мин.", "Макс."]):
                            col.caption(txt)
                        for mm in display_curves:
                            cols = st.columns([1.0, 1.4, 0.7, 0.7, 1.1, 1.0, 1.0])
                            cols[0].markdown(f"**{method_label(mm)}**")
                            others = [None] + [o for o in all_available_methods if o != mm]
                            _sanitize(f"trk2_{mm}", others)
                            m2 = cols[1].selectbox("Второй метод", others, key=f"trk2_{mm}",
                                                   format_func=lambda o: "—" if o is None else method_label(o),
                                                   label_visibility="collapsed")
                            c1v = cols[2].color_picker("Цвет", method_color(mm), key=f"trkc_{mm}",
                                                       label_visibility="collapsed")
                            c2v = cols[3].color_picker("Цвет 2", method_color(m2) if m2 else "#64748B",
                                                       key=f"trkc2_{mm}_{m2}", label_visibility="collapsed",
                                                       disabled=m2 is None)
                            is_log = cols[4].selectbox("Шкала", ["линейная", "логарифм."], key=f"trks_{mm}",
                                                       index=1 if method_log(mm) else 0,
                                                       label_visibility="collapsed") == "логарифм."
                            lo = cols[5].number_input("Мин.", value=None, key=f"trklo_{mm}", placeholder="авто",
                                                      label_visibility="collapsed", format="%.3g")
                            hi = cols[6].number_input("Макс.", value=None, key=f"trkhi_{mm}", placeholder="авто",
                                                      label_visibility="collapsed", format="%.3g")
                            tracks.append({'m1': mm, 'm2': m2, 'c1': c1v, 'c2': c2v, 'log': is_log,
                                           'log2': bool(m2) and method_log(m2), 'range': (lo, hi)})
                        st.caption("Пределы шкалы применяются к первому методу трека. Второй метод рисуется пунктиром "
                                   "со своей шкалой внизу трека (для сопротивлений — логарифмической).")

                    compare = [{'id': i, 'color': gen_by_id[i]['color'], 'curves': gen_by_id[i]['curves']}
                               for i in compare_ids]
                    fig = build_planchette(depth_axis, tracks, active_wells, ref, curr['curves'], curr['id'],
                                           show_ref=show_ref, compare=compare, y_title=y_title, tops=curr['tops'])
                    st.plotly_chart(fig, use_container_width=True,
                                    config={'displaylogo': False, 'scrollZoom': False,
                                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']})
                    st.caption("Выделите область мышью, чтобы увеличить; двойной щелчок — исходный масштаб. "
                               "Тонкие линии — опорные скважины (цвета как на карте), толстые — синтетическая скважина.")
                    if layer_mode:
                        st.caption("Пласты: " + "; ".join(
                            f"{t['name']} {fmt_num(t['top'] + curr['alt'])}–{fmt_num(t['bot'] + curr['alt'])} м по стволу, "
                            f"мощность {fmt_num(t['bot'] - t['top'])} м"
                            for t in curr['tops'] if t['kind'] == 'layer') +
                            ". Кровля и подошва рассчитаны IDW по отбивкам опорных скважин.")

                with st.expander("Связи между методами (коэффициенты корреляции)", expanded=False):
                    pooled = []
                    for k, w in enumerate(active_wells):
                        cols_d = {}
                        for mm, (idx, mat) in ref['curves'].items():
                            if k in idx:
                                cols_d[method_label(mm)] = mat[idx.index(k)]
                        if cols_d:
                            pooled.append(pd.DataFrame(cols_d))
                    ref_corr = pd.concat(pooled, ignore_index=True).corr(min_periods=20) if pooled else pd.DataFrame()
                    syn_corr = pd.DataFrame({method_label(mm): v for mm, v in curr['curves'].items()}).corr(min_periods=20)
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        st.markdown("**Опорные скважины** (все вместе, на интервале)")
                        st.dataframe(ref_corr.round(2), use_container_width=True)
                    with cc2:
                        st.markdown(f"**Синтетическая скважина {curr['id']}**")
                        st.dataframe(syn_corr.round(2), use_container_width=True)
                    st.caption("Если связи между методами сохраняются, таблицы похожи по знакам и примерно по величине.")

            else:
                prof_ids = [i for i in ids if i.startswith("PROF_")]
                default_cols = prof_ids or ids[:6]
                ref_names = {f"{w['name']} (опорная)": w for w in active_wells}
                k1, k2, k3 = st.columns([2.4, 1, 1])
                _sanitize("corr_cols", ids + list(ref_names), multi=True)
                _sanitize("corr_m1", all_available_methods)
                with k1:
                    chosen = st.multiselect("Скважины слева направо:", ids + list(ref_names),
                                            default=default_cols, key="corr_cols",
                                            help="Порядок колонок — порядок выбора. Опорные скважины можно добавить "
                                                 "для сравнения.")
                with k2:
                    cm1 = st.selectbox("Метод", all_available_methods, format_func=method_label, key="corr_m1",
                                       index=all_available_methods.index('GK') if 'GK' in all_available_methods else 0)
                with k3:
                    opts2 = [None] + [o for o in all_available_methods if o != cm1]
                    _sanitize("corr_m2", opts2)
                    cm2 = st.selectbox("Второй метод", opts2, key="corr_m2",
                                       format_func=lambda o: "—" if o is None else method_label(o),
                                       index=opts2.index('PS') if 'PS' in opts2 else 0)
                if chosen:
                    columns = []
                    for c in chosen:
                        if c in gen_by_id:
                            g = gen_by_id[c]
                            columns.append({'name': c, 'curves': g['curves'], 'is_ref': False, 'color': g['color'],
                                            'tops': g['tops']})
                        else:
                            w = ref_names[c]
                            k = active_wells.index(w)
                            curves_w = {mm: mat[idx.index(k)] for mm, (idx, mat) in ref['curves'].items() if k in idx}
                            columns.append({'name': w['name'], 'curves': curves_w, 'is_ref': True, 'color': w['color'],
                                            'tops': ref_tops(k)})
                    fig = build_correlation(depth_axis, columns, cm1, cm2, method_color(cm1),
                                            method_color(cm2) if cm2 else None, method_log(cm1),
                                            bool(cm2) and method_log(cm2), ref['units'], y_title=y_title)
                    st.plotly_chart(fig, use_container_width=True,
                                    config={'displaylogo': False, 'scrollZoom': False,
                                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']})
                    st.caption("Все скважины на одной шкале глубин. Второй метод — пунктиром со шкалой внизу. "
                               "Профиль скважин по линии строится на вкладке карты.")
                else:
                    st.info("Выберите скважины для схемы корреляции.")

    # ------------------------------------------
    # ВКЛАДКА 3: ЭКСПОРТ И ПРОЕКТ
    # ------------------------------------------
    with tab_export:
        if not generated_wells:
            st.info("Нет данных для экспорта. Добавьте точки на карте.")
        else:
            st.subheader("Сводная ведомость синтетических скважин")
            synth_summary = []
            for g in generated_wells:
                if g['tops']:
                    md_lo, md_hi = g['tops'][0]['top'] + g['alt'], g['tops'][-1]['bot'] + g['alt']
                else:
                    md_lo, md_hi = depth_axis[0] + g['alt'], depth_axis[-1] + g['alt']
                row = {"Скважина": g['id'], "X, м": g['x'], "Y, м": g['y'],
                       "Интервал по стволу, м": f"{fmt_num(md_lo)}–{fmt_num(md_hi)}", "Шаг, м": depth_step}
                if use_abs:
                    row["Альтитуда, м"] = round(g['alt'], 1)
                row["Методы"] = ", ".join(method_label(k) for k in g['curves'].keys())
                synth_summary.append(row)
            st.dataframe(pd.DataFrame(synth_summary), use_container_width=True, hide_index=True)
            if layer_mode:
                st.markdown("**Отбивки синтетических скважин** (рассчитаны IDW по отбивкам опорных скважин)")
                st.dataframe(tops_table_for(generated_wells, use_abs), use_container_width=True, hide_index=True)
                st.caption("Эта таблица попадает в архив отдельным файлом и в раздел ~Other каждого LAS.")

            e1, e2 = st.columns([2.2, 1])
            _sanitize("exp_methods", all_available_methods, multi=True)
            with e1:
                exp_methods = st.multiselect("Методы в LAS-файлах", all_available_methods,
                                             default=all_available_methods, format_func=method_label,
                                             key="exp_methods")
            with e2:
                encoding = st.radio("Кодировка", ["UTF-8", "Windows-1251"], horizontal=True,
                                    help="Windows-1251 нужна многим российским программам интерпретации ГИС.")
            enc = 'utf-8' if encoding == "UTF-8" else 'cp1251'

            if exp_methods:
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for gw in generated_wells:
                        zip_file.writestr(las_file_name(gw), make_las_text(gw, depth_axis, ref['units'], exp_methods)
                                          .encode(enc, errors='replace'))
                    if layer_mode:
                        zip_file.writestr("Отбивки_синтетических_скважин.csv",
                                          tops_table_for(generated_wells, use_abs).to_csv(index=False, sep=';')
                                          .encode('utf-8-sig'))
                st.download_button(
                    label=f"Скачать ZIP со всеми LAS-файлами ({len(generated_wells)} скв.)",
                    data=zip_buf.getvalue(), file_name="Synthetic_Wells_Dataset.zip",
                    mime="application/zip", use_container_width=True, type="primary")
            else:
                st.warning("Выберите хотя бы один метод для выгрузки.")

            st.markdown("---")
            st.subheader("Несколько вариантов для студентов")
            st.caption("Те же точки и настройки, но разные номера варианта (seed): у каждого варианта своя изменчивость "
                       "кривых. Каждый вариант лежит в архиве в своей папке.")
            v1, v2 = st.columns([1, 2])
            with v1:
                n_variants = st.number_input("Число вариантов", min_value=2, max_value=50, value=10, step=1)
            with v2:
                st.write("")
                st.write("")
                if st.button("Подготовить архив вариантов", use_container_width=True, disabled=not exp_methods):
                    vbuf = io.BytesIO()
                    rows = []
                    with st.spinner("Генерация вариантов..."):
                        with zipfile.ZipFile(vbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                            for v in range(int(n_variants)):
                                var_wells = run_synth(active_points, seed_offset=v * 1000)
                                folder = f"Вариант_{v + 1:02d}"
                                for gw in var_wells:
                                    zf.writestr(f"{folder}/{las_file_name(gw)}",
                                                make_las_text(gw, depth_axis, ref['units'], exp_methods)
                                                .encode(enc, errors='replace'))
                                if layer_mode:
                                    zf.writestr(f"{folder}/Отбивки.csv", tops_table_for(var_wells, use_abs)
                                                .to_csv(index=False, sep=';').encode('utf-8-sig'))
                                rows.append({"Вариант": v + 1, "Seed": int(base_seed) + v * 1000,
                                             "Скважин": len(var_wells)})
                            zf.writestr("Варианты.csv", pd.DataFrame(rows).to_csv(index=False, sep=';'))
                    st.session_state.variants_zip = (int(n_variants), vbuf.getvalue())
            if st.session_state.variants_zip:
                nv, data = st.session_state.variants_zip
                st.download_button(f"Скачать архив вариантов ({nv} шт.)", data=data,
                                   file_name="Synthetic_Wells_Variants.zip", mime="application/zip",
                                   use_container_width=True)

        st.markdown("---")
        st.subheader("Проект")
        st.caption("Сохраняет загруженные LAS-файлы, координаты и активность скважин, синтетические точки и все настройки. "
                   "Открыть проект можно в панели слева: «Открыть сохраненный проект».")
        if st.button("Подготовить файл проекта", use_container_width=True):
            sources = [(w['file'], w['raw']) for w in wells_data]
            st.session_state.project_zip = build_project_zip(sources, st.session_state.ref_overrides,
                                                             st.session_state.points_list,
                                                             st.session_state.tops_current)
        if st.session_state.project_zip:
            st.download_button("Скачать проект (.zip)", data=st.session_state.project_zip,
                               file_name="SynthGeoGen_project.zip", mime="application/zip",
                               use_container_width=True, type="primary")

    # ------------------------------------------
    # ВКЛАДКА 4: СПРАВКА
    # ------------------------------------------
    with tab_help:
        st.markdown(HELP_MD)


# Запуск модуля отдельно: streamlit run app_geo_cvae.py (основной вход — app.py)
if __name__ == "__main__":
    st.set_page_config(page_title="SynthGeoGen", layout="wide")
    run_geo_app()
