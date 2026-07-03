#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ГИБРИДНЫЙ ГЕНЕРАТОР СИНТЕТИЧЕСКИХ СЕЙСМИЧЕСКИХ ТРАСС
Лабораторные работы №4 и №5

Структура:
1. Базовые классы (из ЛР №1-3) — для справки
2. ЛР №4: Генерация методом Монте-Карло
3. ЛР №5: Учёт вариаций свойств внутри пласта

Автор: Юлия Носырева
Руководитель: доцент Николенко Т.А.
Тюменский индустриальный университет, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. БАЗОВЫЕ КЛАССЫ (из лабораторных работ №1-3)
# ============================================================================

class RickerWavelet:
    """Генератор вейвлета Рикера."""
    
    def __init__(self, dominant_freq=30, dt=0.001, length=0.2):
        """
        dominant_freq : доминантная частота (Гц)
        dt            : интервал дискретизации (с)
        length        : длительность вейвлета (с)
        """
        self.freq = dominant_freq
        self.dt = dt
        self.length = length
        self.time = np.arange(-length/2, length/2, dt)
        tau = np.pi * self.freq * self.time
        self.wavelet = (1 - 2 * tau**2) * np.exp(-tau**2)
    
    def get_wavelet(self):
        """Возвращает временную сетку и вейвлет."""
        return self.time, self.wavelet


class ReflectionCoefficients:
    """Расчёт коэффициентов отражения по импедансам."""
    
    def __init__(self, depths, velocities, densities):
        """
        depths     : глубины границ (м)
        velocities : скорости в слоях (м/с)
        densities  : плотности в слоях (кг/м³)
        """
        self.depths = np.array(depths)
        self.velocities = np.array(velocities)
        self.densities = np.array(densities)
        self.impedances = self.densities * self.velocities
        
        self.coeffs = []
        self.times = []
        cum_time = 0
        prev_depth = 0
        
        for i in range(len(self.depths)):
            Z_up = self.impedances[i]
            Z_down = self.impedances[i+1]
            R = (Z_down - Z_up) / (Z_down + Z_up)
            self.coeffs.append(R)
            
            thickness = self.depths[i] - prev_depth
            v = self.velocities[i]
            cum_time += 2 * thickness / v
            self.times.append(cum_time)
            prev_depth = self.depths[i]
        
        self.coeffs = np.array(self.coeffs)
        self.times = np.array(self.times)
    
    def get_reflectivity_series(self, dt, total_time):
        """Преобразует коэффициенты в дискретную временную серию."""
        ns = int(total_time / dt) + 1
        reflectivity = np.zeros(ns)
        for t, r in zip(self.times, self.coeffs):
            idx = int(t / dt)
            if idx < ns:
                reflectivity[idx] = r
        return reflectivity
    
    def print_model(self):
        """Вывод модели на экран."""
        print("=" * 60)
        print("ГЕОЛОГИЧЕСКАЯ МОДЕЛЬ")
        print("=" * 60)
        for i in range(len(self.depths) + 1):
            v = self.velocities[i] if i < len(self.velocities) else '---'
            rho = self.densities[i] if i < len(self.densities) else '---'
            z = self.impedances[i] if i < len(self.impedances) else '---'
            print(f"Слой {i}: V={v}, ρ={rho}, Z={z}")
        print("\nГраницы:")
        for i, (t, r) in enumerate(zip(self.times, self.coeffs)):
            print(f"  {i}: глубина={self.depths[i]} м, t={t:.3f} с, R={r:.4f}")


class TraceGenerator:
    """Генератор синтетических трасс."""
    
    def __init__(self, wavelet_freq=30, dt=0.001, noise_std=0.0):
        self.dt = dt
        self.noise_std = noise_std
        wav = RickerWavelet(wavelet_freq, dt)
        self.wavelet = wav.wavelet
    
    def generate(self, reflectivity):
        """Генерирует трассу из коэффициентов отражения."""
        trace = np.convolve(reflectivity, self.wavelet, mode='same')
        if self.noise_std > 0:
            trace += np.random.normal(0, self.noise_std, len(trace))
        return trace


# ============================================================================
# 2. ЛАБОРАТОРНАЯ РАБОТА №4: МЕТОД МОНТЕ-КАРЛО
# ============================================================================

def sample_geology(distributions, enforce_sort=True):
    """
    Генерирует один случайный набор геологических параметров.
    
    distributions: словарь с распределениями для скоростей, плотностей и глубин
    enforce_sort : сортировать глубины (обязательно для физической модели)
    
    Возвращает: (depths, velocities, densities)
    """
    depths = []
    velocities = []
    densities = []
    
    # Глубины границ
    for d in distributions['depths']:
        if d['type'] == 'uniform':
            h = np.random.uniform(d['low'], d['high'])
        elif d['type'] == 'normal':
            h = np.random.normal(d['mu'], d['sigma'])
        elif d['type'] == 'lognormal':
            h = np.random.lognormal(d['mu'], d['sigma'])
        else:
            h = d.get('value', 1000)
        depths.append(h)
    
    # Скорости слоёв
    for v in distributions['velocities']:
        if v['type'] == 'normal':
            val = np.random.normal(v['mu'], v['sigma'])
        elif v['type'] == 'uniform':
            val = np.random.uniform(v['low'], v['high'])
        elif v['type'] == 'lognormal':
            val = np.random.lognormal(v['mu'], v['sigma'])
        else:
            val = v.get('value', 2500)
        # Ограничиваем физически возможные значения
        val = max(val, 500)
        velocities.append(val)
    
    # Плотности слоёв
    for r in distributions['densities']:
        if r['type'] == 'normal':
            val = np.random.normal(r['mu'], r['sigma'])
        elif r['type'] == 'uniform':
            val = np.random.uniform(r['low'], r['high'])
        elif r['type'] == 'lognormal':
            val = np.random.lognormal(r['mu'], r['sigma'])
        else:
            val = r.get('value', 2200)
        # Ограничиваем физически возможные значения
        val = max(val, 1500)
        densities.append(val)
    
    # Сортируем глубины (важно — должны возрастать)
    depths = np.array(depths)
    if enforce_sort and len(depths) > 1:
        depths = np.sort(depths)
    
    return np.array(depths), np.array(velocities), np.array(densities)


def generate_ensemble(distributions, M=100, wavelet_freq=30, dt=0.001, 
                      total_time=2.0, noise_std=0.0, verbose=True):
    """
    Генерирует ансамбль из M трасс методом Монте-Карло.
    
    Возвращает:
    - traces: массив [M x num_samples]
    - time_axis: временная ось
    - all_params: список словарей с параметрами каждой модели
    """
    traces = []
    all_params = []
    
    for i in range(M):
        # Шаг 1: случайные параметры
        depths, velocities, densities = sample_geology(distributions)
        
        # Шаг 2: расчёт коэффициентов отражения
        try:
            rc = ReflectionCoefficients(depths, velocities, densities)
            reflectivity = rc.get_reflectivity_series(dt, total_time)
        except Exception as e:
            if verbose:
                print(f"Предупреждение: ошибка при генерации модели {i}: {e}")
            # Пропускаем неудачную реализацию
            continue
        
        # Шаг 3: генерация трассы
        generator = TraceGenerator(wavelet_freq, dt, noise_std)
        trace = generator.generate(reflectivity)
        
        traces.append(trace)
        all_params.append({
            'depths': depths.tolist(),
            'velocities': velocities.tolist(),
            'densities': densities.tolist(),
            'coeffs': rc.coeffs.tolist(),
            'times': rc.times.tolist()
        })
    
    traces = np.array(traces)
    time_axis = np.arange(0, total_time, dt)[:len(traces[0])]
    
    if verbose:
        print(f"Сгенерировано {len(traces)} трасс из {M} попыток")
    
    return traces, time_axis, all_params


def compute_statistics(traces):
    """
    Вычисляет среднюю трассу, стандартное отклонение и 95% доверительные интервалы.
    """
    mean_trace = np.mean(traces, axis=0)
    std_trace = np.std(traces, axis=0)
    n = len(traces)
    
    # 95% доверительный интервал (t-распределение для малых выборок)
    from scipy import stats
    t_val = stats.t.ppf(0.975, n - 1) if n > 1 else 1.96
    ci_lower = mean_trace - t_val * std_trace / np.sqrt(n)
    ci_upper = mean_trace + t_val * std_trace / np.sqrt(n)
    
    return mean_trace, std_trace, ci_lower, ci_upper


def plot_ensemble(traces, time_axis, mean_trace=None, ci_lower=None, ci_upper=None, 
                  show_individual=True, title="Ансамбль трасс (Монте-Карло)"):
    """
    Визуализация ансамбля трасс.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # График 1: Все трассы ансамбля
    if show_individual:
        max_show = min(len(traces), 50)
        for i in range(max_show):
            axes[0].plot(time_axis, traces[i], 'b-', alpha=0.1, linewidth=0.5)
    
    # Средняя трасса и доверительные интервалы
    if mean_trace is None:
        mean_trace = np.mean(traces, axis=0)
    axes[0].plot(time_axis, mean_trace, 'r-', linewidth=2, label='Средняя трасса')
    
    if ci_lower is not None and ci_upper is not None:
        axes[0].fill_between(time_axis, ci_lower, ci_upper, 
                              color='r', alpha=0.2, label='95% ДИ')
    
    axes[0].set_title(title)
    axes[0].set_xlabel('Время (с)')
    axes[0].set_ylabel('Амплитуда')
    axes[0].legend()
    axes[0].grid(True)
    
    # График 2: Стандартное отклонение
    std_trace = np.std(traces, axis=0)
    axes[1].plot(time_axis, std_trace, 'g-', linewidth=2)
    axes[1].fill_between(time_axis, 0, std_trace, color='g', alpha=0.3)
    axes[1].set_title('Стандартное отклонение трасс (мера неопределённости)')
    axes[1].set_xlabel('Время (с)')
    axes[1].set_ylabel('σ')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def analyze_sensitivity(distributions, param_name, param_values, 
                         M=50, wavelet_freq=30, dt=0.001, total_time=2.0):
    """
    Анализ чувствительности: как изменение параметра влияет на разброс трасс.
    """
    results = {}
    
    for val in param_values:
        # Копируем распределения с новым значением
        dist_copy = distributions.copy()
        
        # Находим параметр и заменяем его
        found = False
        for layer_idx, layer in enumerate(dist_copy[param_name]):
            if 'mu' in layer:
                layer['mu'] = val
                found = True
                break
            elif 'low' in layer and 'high' in layer:
                # Для равномерного распределения меняем среднее
                mid = (layer['low'] + layer['high']) / 2
                shift = val - mid
                layer['low'] += shift
                layer['high'] += shift
                found = True
                break
        
        if not found:
            print(f"Параметр {param_name} не найден в распределениях")
            return None
        
        # Генерируем ансамбль
        traces, time_axis, _ = generate_ensemble(
            dist_copy, M=M, wavelet_freq=wavelet_freq, 
            dt=dt, total_time=total_time, verbose=False
        )
        
        # Вычисляем статистику
        mean_trace, std_trace, ci_lower, ci_upper = compute_statistics(traces)
        
        results[val] = {
            'mean': mean_trace,
            'std': std_trace,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'time': time_axis
        }
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Средние трассы
    for val, data in results.items():
        axes[0].plot(data['time'], data['mean'], label=f'{param_name} = {val}')
    axes[0].set_title(f'Средние трассы при разных {param_name}')
    axes[0].set_xlabel('Время (с)')
    axes[0].set_ylabel('Амплитуда')
    axes[0].legend()
    axes[0].grid(True)
    
    # Стандартные отклонения
    for val, data in results.items():
        axes[1].plot(data['time'], data['std'], label=f'{param_name} = {val}')
    axes[1].set_title(f'Стандартное отклонение при разных {param_name}')
    axes[1].set_xlabel('Время (с)')
    axes[1].set_ylabel('σ')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results


# ============================================================================
# 3. ЛАБОРАТОРНАЯ РАБОТА №5: УЧЁТ ВАРИАЦИЙ ВНУТРИ ПЛАСТА
# ============================================================================

def generate_layer_properties(z, V0, rho0, trend_type='linear', params=None):
    """
    Генерирует скорости и плотности внутри слоя по заданному тренду.
    
    Параметры:
    z          : массив глубин внутри слоя (от 0 до H)
    V0, rho0   : значения на верхней границе слоя
    trend_type : 'linear', 'exponential', 'sinusoidal', 'random', 'step'
    params     : дополнительные параметры тренда
    """
    if params is None:
        params = {}
    
    V = np.ones_like(z) * V0
    rho = np.ones_like(z) * rho0
    
    if trend_type == 'linear':
        k = params.get('k', 100)  # градиент скорости (м/с на метр)
        V = V0 + k * z
        rho = rho0 + 0.1 * k * z
        
    elif trend_type == 'exponential':
        k = params.get('k', 0.001)
        V = V0 * np.exp(k * z)
        rho = rho0 * np.exp(0.5 * k * z)
        
    elif trend_type == 'sinusoidal':
        A = params.get('A', 150)      # амплитуда колебаний
        L = params.get('L', 50)       # период (м)
        phi = params.get('phi', 0)    # фаза
        V = V0 + A * np.sin(2 * np.pi * z / L + phi)
        rho = rho0 + 10 * np.sin(2 * np.pi * z / L + phi)
        
    elif trend_type == 'random':
        sigma = params.get('sigma', 50)
        correlation_length = params.get('correlation_length', 10)
        
        # Генерируем коррелированный шум (простое скользящее среднее)
        noise = np.random.normal(0, sigma, len(z))
        window = int(correlation_length / (z[1] - z[0] if len(z) > 1 else 1))
        window = max(1, min(window, len(z) // 2))
        kernel = np.ones(window) / window
        noise_smooth = np.convolve(noise, kernel, mode='same')
        V = V0 + noise_smooth
        rho = rho0 + 0.1 * noise_smooth
        
    elif trend_type == 'step':
        step_depth = params.get('step_depth', 100)  # глубина скачка
        step_amplitude = params.get('step_amplitude', 300)  # амплитуда скачка
        V = V0 + step_amplitude * (z > step_depth)
        rho = rho0 + 10 * (z > step_depth)
    
    # Ограничиваем физически возможные значения
    V = np.clip(V, 500, 8000)
    rho = np.clip(rho, 1500, 3500)
    
    return V, rho


def compute_reflectivity_with_gradient(depths, V0, rho0, trend_type='linear', 
                                       params=None, num_sublayers=100, dt=0.001, 
                                       total_time=2.0):
    """
    Вычисляет коэффициенты отражения для слоя с внутренним трендом.
    
    Возвращает:
    reflectivity : временная серия коэффициентов отражения
    z_micro      : глубины микро-границ
    V_micro      : скорости на микро-границах
    rho_micro    : плотности на микро-границах
    """
    if params is None:
        params = {}
    
    h_top, h_bottom = depths
    H = h_bottom - h_top
    
    if H <= 0:
        raise ValueError("Толщина слоя должна быть положительной")
    
    # Глубины микро-границ
    z_micro = np.linspace(h_top, h_bottom, num_sublayers + 1)
    
    # Свойства на каждой микро-глубине
    V_micro, rho_micro = generate_layer_properties(
        z_micro - h_top,   # относительная глубина
        V0, rho0,
        trend_type,
        params
    )
    
    # Импедансы на микро-границах
    Z_micro = V_micro * rho_micro
    
    # Коэффициенты отражения на микро-границах
    R_micro = []
    for i in range(len(Z_micro) - 1):
        if Z_micro[i] > 0 and Z_micro[i+1] > 0:
            R = (Z_micro[i+1] - Z_micro[i]) / (Z_micro[i+1] + Z_micro[i])
        else:
            R = 0
        R_micro.append(R)
    R_micro = np.array(R_micro)
    
    # Преобразуем во временную серию
    ns = int(total_time / dt) + 1
    reflectivity = np.zeros(ns)
    
    for i, z in enumerate(z_micro[1:]):
        # Время прихода от микро-границы на глубине z
        # Используем среднюю скорость до этой глубины
        avg_v = np.mean(V_micro[:i+2])
        t = 2 * (z - h_top) / avg_v + 2 * h_top / V0
        idx = int(t / dt)
        if idx < ns:
            reflectivity[idx] += R_micro[i]
    
    return reflectivity, z_micro, V_micro, rho_micro, R_micro


def backus_average(V, rho, z):
    """
    Рассчитывает эффективные свойства слоя по формулам Бэкуса.
    
    Возвращает: V_eff, rho_eff
    """
    if len(z) < 2:
        return V[0], rho[0]
    
    dz = np.gradient(z)
    H = z[-1] - z[0]
    
    if H <= 0:
        return np.mean(V), np.mean(rho)
    
    # Средняя плотность
    rho_eff = np.trapz(rho, z) / H
    
    # Средний P-модуль M = ρ·V²
    M = rho * V**2
    M_inv = np.ones_like(M) / M
    M_inv_avg = np.trapz(M_inv, z) / H
    
    # Эффективная скорость
    V_eff = np.sqrt(1 / (rho_eff * M_inv_avg))
    
    return V_eff, rho_eff


def compare_homogeneous_vs_gradient(depths, V0, rho0, trend_type='linear', 
                                     params=None, num_sublayers=100, 
                                     wavelet_freq=30, dt=0.001, total_time=2.0):
    """
    Сравнивает трассы для однородного и неоднородного слоёв.
    """
    if params is None:
        params = {}
    
    # 1. Однородная модель (слой без внутренней структуры)
    # Используем одну границу на подошве слоя с контрастом свойств
    V_bottom = V0  # для простоты считаем, что под слоем такая же скорость
    rho_bottom = rho0
    
    rc_hom = ReflectionCoefficients(
        [depths[1]],  # только граница на подошве
        [V0, V_bottom],
        [rho0, rho_bottom]
    )
    reflectivity_hom = rc_hom.get_reflectivity_series(dt, total_time)
    generator_hom = TraceGenerator(wavelet_freq, dt, noise_std=0.0)
    trace_hom = generator_hom.generate(reflectivity_hom)
    
    # 2. Неоднородная модель (с трендом внутри слоя)
    reflectivity_grad, z_micro, V_micro, rho_micro, R_micro = compute_reflectivity_with_gradient(
        depths, V0, rho0, trend_type, params, num_sublayers, dt, total_time
    )
    generator_grad = TraceGenerator(wavelet_freq, dt, noise_std=0.0)
    trace_grad = generator_grad.generate(reflectivity_grad)
    
    # 3. Расчёт эффективных свойств (Backus)
    # Берём свойства на микро-уровне
    z_full = z_micro
    V_eff, rho_eff = backus_average(V_micro, rho_micro, z_full)
    
    # 4. Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    
    time_axis = np.arange(0, total_time, dt)[:len(trace_hom)]
    
    # Трассы
    axes[0, 0].plot(time_axis, trace_hom, 'b-', linewidth=2, label='Однородный слой')
    axes[0, 0].plot(time_axis, trace_grad, 'r-', linewidth=2, label='Неоднородный слой')
    axes[0, 0].axvline(x=2*depths[0]/V0, color='gray', linestyle='--', alpha=0.5, 
                       label=f'Время входа {2*depths[0]/V0:.3f}с')
    axes[0, 0].set_title(f'Сравнение трасс (тренд: {trend_type})')
    axes[0, 0].set_xlabel('Время (с)')
    axes[0, 0].set_ylabel('Амплитуда')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Коэффициенты отражения на микро-границах
    axes[0, 1].plot(z_micro[1:], R_micro, 'ro-', markersize=3, linewidth=0.5)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].set_title(f'Коэффициенты отражения (n={len(R_micro)} границ)')
    axes[0, 1].set_xlabel('Глубина (м)')
    axes[0, 1].set_ylabel('R')
    axes[0, 1].grid(True)
    
    # Профиль скорости внутри слоя
    axes[1, 0].plot(V_micro, z_micro, 'g-', linewidth=2)
    axes[1, 0].axvline(x=V0, color='k', linestyle='--', alpha=0.5, label=f'V0={V0}')
    axes[1, 0].axvline(x=V_eff, color='r', linestyle='--', alpha=0.7, 
                       label=f'V_eff={V_eff:.0f}')
    axes[1, 0].set_title(f'Профиль скорости внутри слоя (V_eff={V_eff:.0f} м/с)')
    axes[1, 0].set_xlabel('V (м/с)')
    axes[1, 0].set_ylabel('Глубина (м)')
    axes[1, 0].invert_yaxis()
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Разность трасс и статистика
    diff = trace_grad - trace_hom
    axes[1, 1].plot(time_axis, diff, 'k-', linewidth=1.5)
    axes[1, 1].axhline(y=0, color='gray', linestyle='--')
    axes[1, 1].set_title(f'Разность (неоднородная - однородная)\n'
                         f'RMS = {np.sqrt(np.mean(diff**2)):.4f}')
    axes[1, 1].set_xlabel('Время (с)')
    axes[1, 1].set_ylabel('ΔАмплитуда')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Информация о модели
    print("=" * 60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)
    print(f"Тренд: {trend_type}")
    print(f"Верхняя граница: {depths[0]} м, нижняя: {depths[1]} м")
    print(f"Толщина слоя: {depths[1] - depths[0]} м")
    print(f"V0 на верхней границе: {V0} м/с")
    print(f"ρ0 на верхней границе: {rho0} кг/м³")
    print(f"Эффективная скорость (Backus): {V_eff:.0f} м/с")
    print(f"Эффективная плотность: {rho_eff:.0f} кг/м³")
    print(f"Количество микро-границ: {len(R_micro)}")
    print(f"RMS разности: {np.sqrt(np.mean(diff**2)):.4f}")
    
    return trace_hom, trace_grad, {'V_eff': V_eff, 'rho_eff': rho_eff}


def compare_trends(depths, V0, rho0, trend_types=['linear', 'exponential', 
                    'sinusoidal', 'random', 'step'], wavelet_freq=30):
    """
    Сравнивает трассы для разных типов трендов на одном графике.
    """
    dt = 0.001
    total_time = 2.0
    time_axis = np.arange(0, total_time, dt)
    
    traces = {}
    colors = ['b', 'r', 'g', 'c', 'm']
    
    plt.figure(figsize=(12, 6))
    
    for i, trend in enumerate(trend_types):
        params = {}
        if trend == 'linear':
            params = {'k': 150}
        elif trend == 'exponential':
            params = {'k': 0.0008}
        elif trend == 'sinusoidal':
            params = {'A': 200, 'L': 60}
        elif trend == 'random':
            params = {'sigma': 80, 'correlation_length': 15}
        elif trend == 'step':
            params = {'step_depth': 100, 'step_amplitude': 400}
        
        # Генерируем трассу для этого тренда
        reflectivity, _, _, _, _ = compute_reflectivity_with_gradient(
            depths, V0, rho0, trend, params, num_sublayers=100, 
            dt=dt, total_time=total_time
        )
        generator = TraceGenerator(wavelet_freq, dt, 0.0)
        trace = generator.generate(reflectivity)
        
        color = colors[i % len(colors)]
        plt.plot(time_axis[:len(trace)], trace, color=color, 
                 linewidth=1.5, label=trend)
        traces[trend] = trace
    
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.title('Сравнение разных типов трендов внутри слоя')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1.5)
    plt.show()
    
    return traces


# ============================================================================
# 4. ДЕМОНСТРАЦИЯ (примеры использования)
# ============================================================================

def demo_lab4():
    """
    Демонстрация лабораторной работы №4 (Монте-Карло).
    """
    print("\n" + "="*60)
    print("ЛАБОРАТОРНАЯ РАБОТА №4: ГЕНЕРАЦИЯ МЕТОДОМ МОНТЕ-КАРЛО")
    print("="*60)
    
    # Задаём распределения для трёхслойной модели (2 границы)
    distributions = {
        'velocities': [
            {'type': 'normal', 'mu': 2000, 'sigma': 100},
            {'type': 'normal', 'mu': 2500, 'sigma': 150},
            {'type': 'normal', 'mu': 3000, 'sigma': 200}
        ],
        'densities': [
            {'type': 'normal', 'mu': 2100, 'sigma': 50},
            {'type': 'normal', 'mu': 2200, 'sigma': 60},
            {'type': 'normal', 'mu': 2400, 'sigma': 70}
        ],
        'depths': [
            {'type': 'uniform', 'low': 450, 'high': 550},
            {'type': 'uniform', 'low': 1100, 'high': 1300}
        ]
    }
    
    # Генерируем ансамбль
    print("Генерация ансамбля из 100 трасс...")
    traces, time_axis, params = generate_ensemble(
        distributions, M=100, wavelet_freq=30, dt=0.001, 
        total_time=2.0, noise_std=0.01, verbose=True
    )
    
    # Статистика
    mean_trace, std_trace, ci_lower, ci_upper = compute_statistics(traces)
    
    # Визуализация
    plot_ensemble(traces, time_axis, mean_trace, ci_lower, ci_upper)
    
    # Анализ чувствительности
    print("\nАнализ чувствительности: изменение скорости первого слоя")
    analyze_sensitivity(
        distributions, 'velocities', [1800, 2000, 2200, 2400],
        M=30, wavelet_freq=30, dt=0.001, total_time=2.0
    )
    
    return traces, time_axis, params


def demo_lab5():
    """
    Демонстрация лабораторной работы №5 (вариации свойств внутри пласта).
    """
    print("\n" + "="*60)
    print("ЛАБОРАТОРНАЯ РАБОТА №5: УЧЁТ ВАРИАЦИЙ ВНУТРИ ПЛАСТА")
    print("="*60)
    
    # Параметры слоя
    depths = [500, 800]  # слой от 500 до 800 м
    V0 = 2500
    rho0 = 2200
    
    # 1. Сравнение однородного и неоднородного слоёв
    print("\n1. Сравнение однородного и неоднородного слоя (синусоидальный тренд)")
    trace_hom, trace_grad, stats = compare_homogeneous_vs_gradient(
        depths, V0, rho0, 
        trend_type='sinusoidal',
        params={'A': 200, 'L': 60},
        num_sublayers=100,
        wavelet