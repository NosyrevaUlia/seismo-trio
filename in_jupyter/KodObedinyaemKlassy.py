import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# =========================================================
# 1. Предполагается, что у тебя уже есть классы из ЛР2:
#    RickerWavelet, ReflectionCoefficients, TraceGenerator
#    (они используются внутри)
# =========================================================

class HybridGenerator:
    """
    Гибридный генератор, объединяющий:
    1) Монте-Карло (вариация параметров слоёв глобально)
    2) Микрослоистость (вариация свойств ВНУТРИ каждого слоя)
    """
    def __init__(self, wavelet_freq=30, dt=0.001, total_time=2.0):
        self.wavelet_freq = wavelet_freq
        self.dt = dt
        self.total_time = total_time
        self.base_generator = TraceGenerator(wavelet_freq, dt, noise_std=0.0)

    def generate_trace(self, distributions, layer_trends, 
                       use_mc=True, use_trends=True, noise_std=0.0):
        """
        Генерирует одну трассу с учётом обоих эффектов.

        Параметры:
        distributions : словарь с распределениями для Монте-Карло 
                        (как в ЛР4)
        layer_trends  : список словарей с описанием тренда для КАЖДОГО слоя
                        [{'type': 'sinusoidal', 'A': 100, 'L': 50}, 
                         {'type': 'linear', 'k': 200}, ...]
        use_mc        : флаг, включать ли Монте-Карло
        use_trends    : флаг, включать ли микрослоистость
        noise_std     : уровень шума
        """
        # --- ШАГ 1: Глобальная модель (Монте-Карло или фиксированная) ---
        if use_mc:
            # Берём случайные параметры из распределений
            depths, velocities, densities = self._sample_geology(distributions)
        else:
            # Берём средние значения (детерминированная модель)
            depths = self._get_means(distributions['depths'])
            velocities = self._get_means(distributions['velocities'])
            densities = self._get_means(distributions['densities'])

        # --- ШАГ 2: Внутренняя структура слоёв (микрослои) ---
        if use_trends:
            # Генерируем коэффициенты отражения с учётом трендов внутри слоёв
            reflectivity = self._compute_reflectivity_with_trends(
                depths, velocities, densities, layer_trends
            )
        else:
            # Классические коэффициенты (только на границах слоёв)
            rc = ReflectionCoefficients(depths, velocities, densities)
            reflectivity = rc.get_reflectivity_series(self.dt, self.total_time)

        # --- ШАГ 3: Свёртка и шум ---
        generator = TraceGenerator(self.wavelet_freq, self.dt, noise_std)
        trace = generator.generate(reflectivity)
        
        return trace, depths, velocities, densities

    # -------------------------------------------
    # Вспомогательные методы (копия из ЛР4)
    # -------------------------------------------
    def _sample_geology(self, distributions):
        """Генерирует случайную геологию (код из ЛР4)."""
        depths = []
        for d in distributions['depths']:
            if d['type'] == 'uniform':
                h = np.random.uniform(d['low'], d['high'])
            elif d['type'] == 'normal':
                h = np.random.normal(d['mu'], d['sigma'])
            else:
                h = d.get('value', 1000)
            depths.append(h)
        
        velocities = []
        for v in distributions['velocities']:
            if v['type'] == 'normal':
                val = np.random.normal(v['mu'], v['sigma'])
            elif v['type'] == 'uniform':
                val = np.random.uniform(v['low'], v['high'])
            else:
                val = v.get('value', 2500)
            velocities.append(max(val, 500))
        
        densities = []
        for r in distributions['densities']:
            if r['type'] == 'normal':
                val = np.random.normal(r['mu'], r['sigma'])
            elif r['type'] == 'uniform':
                val = np.random.uniform(r['low'], r['high'])
            else:
                val = r.get('value', 2200)
            densities.append(max(val, 1500))
        
        return np.sort(depths), np.array(velocities), np.array(densities)

    def _get_means(self, dist_list):
        """Берёт средние значения (математическое ожидание) для детерминированного режима."""
        means = []
        for d in dist_list:
            if d['type'] == 'uniform':
                means.append((d['low'] + d['high']) / 2)
            elif d['type'] == 'normal':
                means.append(d['mu'])
            else:
                means.append(d.get('value', 1000))
        return np.array(means)

    def _compute_reflectivity_with_trends(self, depths, velocities, densities, layer_trends):
        """
        Расчёт коэффициентов отражения с учётом микрослоистости внутри КАЖДОГО слоя.
        Это аналог кода из ЛР5, но применённый ко всем слоям последовательно.
        """
        ns = int(self.total_time / self.dt) + 1
        reflectivity = np.zeros(ns)
        
        # Проходим по всем слоям
        for i in range(len(depths) + 1):
            # Определяем границы слоя
            if i == 0:
                top = 0
            else:
                top = depths[i-1]
            
            if i == len(depths):
                bottom = depths[-1] + 100  # подошва последнего слоя (условно)
            else:
                bottom = depths[i]
            
            # Если слой нулевой толщины — пропускаем
            if bottom - top <= 0:
                continue
            
            # Базовые свойства слоя
            V0 = velocities[i]
            rho0 = densities[i]
            
            # Получаем описание тренда для этого слоя
            if i < len(layer_trends):
                trend = layer_trends[i]
            else:
                trend = {'type': 'none'}
            
            # Генерируем микро-границы внутри слоя (как в ЛР5)
            num_sublayers = 50  # можно вынести в параметры
            z_micro = np.linspace(top, bottom, num_sublayers + 1)
            
            # Свойства на микро-границах (функция из ЛР5)
            V_micro, rho_micro = self._generate_layer_properties(
                z_micro - top, V0, rho0, trend
            )
            
            # Импедансы
            Z_micro = V_micro * rho_micro
            
            # Коэффициенты отражения на микро-границах
            for j in range(len(Z_micro) - 1):
                if Z_micro[j] > 0 and Z_micro[j+1] > 0:
                    R = (Z_micro[j+1] - Z_micro[j]) / (Z_micro[j+1] + Z_micro[j])
                else:
                    R = 0
                
                # Время прихода от микро-границы
                avg_v = np.mean(V_micro[:j+2])
                t = 2 * (z_micro[j+1] - top) / avg_v + 2 * top / V0 if top > 0 else 0
                idx = int(t / self.dt)
                if idx < ns:
                    reflectivity[idx] += R
        
        return reflectivity

    def _generate_layer_properties(self, z, V0, rho0, trend):
        """Генерация тренда внутри слоя (код из ЛР5)."""
        if trend['type'] == 'none' or trend['type'] == 'linear' and trend.get('k', 0) == 0:
            return np.ones_like(z) * V0, np.ones_like(z) * rho0
        
        V = np.ones_like(z) * V0
        rho = np.ones_like(z) * rho0
        
        if trend['type'] == 'linear':
            k = trend.get('k', 100)
            V = V0 + k * z
            rho = rho0 + 0.1 * k * z
        elif trend['type'] == 'sinusoidal':
            A = trend.get('A', 150)
            L = trend.get('L', 50)
            V = V0 + A * np.sin(2 * np.pi * z / L)
            rho = rho0 + 10 * np.sin(2 * np.pi * z / L)
        elif trend['type'] == 'random':
            sigma = trend.get('sigma', 50)
            V = V0 + np.random.normal(0, sigma, len(z))
            rho = rho0 + np.random.normal(0, 10, len(z))
        
        return np.clip(V, 500, 8000), np.clip(rho, 1500, 3500)


# =========================================================
# 2. ДЕМОНСТРАЦИЯ РАБОТЫ ГИБРИДНОГО ГЕНЕРАТОРА
# =========================================================

def demo_hybrid():
    # 1. Задаём распределения для Монте-Карло (как в ЛР4)
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

    # 2. Задаём тренды для КАЖДОГО слоя (их 3)
    #    Слой 0: от 0 до 500 м, слой 1: от 500 до 1200 м, слой 2: от 1200 до ... 
    layer_trends = [
        {'type': 'linear', 'k': 150},           # слой 0
        {'type': 'sinusoidal', 'A': 200, 'L': 80}, # слой 1
        {'type': 'none'}                        # слой 2 (однородный)
    ]

    # 3. Создаём гибридный генератор
    hybrid = HybridGenerator(wavelet_freq=30, dt=0.001, total_time=2.0)

    # 4. Генерируем 5 трасс с разными комбинациями флагов
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Режим 1: Только Монте-Карло (без трендов)
    traces_mc = []
    for _ in range(10):
        trace, _, _, _ = hybrid.generate_trace(
            distributions, layer_trends, use_mc=True, use_trends=False
        )
        traces_mc.append(trace)
    axes[0].set_title('Режим 1: Только Монте-Карло (глобальная неопределённость)')
    for t in traces_mc:
        axes[0].plot(np.arange(0, 2.0, 0.001)[:len(t)], t, 'b-', alpha=0.3)

    # Режим 2: Только тренды (без Монте-Карло)
    traces_tr = []
    for _ in range(10):
        trace, _, _, _ = hybrid.generate_trace(
            distributions, layer_trends, use_mc=False, use_trends=True
        )
        traces_tr.append(trace)
    axes[1].set_title('Режим 2: Только микрослоистость (вариации внутри слоёв)')
    for t in traces_tr:
        axes[1].plot(np.arange(0, 2.0, 0.001)[:len(t)], t, 'r-', alpha=0.3)

    # Режим 3: ГИБРИД (Монте-Карло + микрослоистость)
    traces_hyb = []
    for _ in range(10):
        trace, _, _, _ = hybrid.generate_trace(
            distributions, layer_trends, use_mc=True, use_trends=True
        )
        traces_hyb.append(trace)
    axes[2].set_title('Режим 3: ГИБРИД (Монте-Карло + микрослоистость)')
    for t in traces_hyb:
        axes[2].plot(np.arange(0, 2.0, 0.001)[:len(t)], t, 'g-', alpha=0.3)

    plt.tight_layout()
    plt.show()


# Запуск демонстрации
if __name__ == "__main__":
    demo_hybrid()