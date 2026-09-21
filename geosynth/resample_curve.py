import numpy as np
from scipy.interpolate import interp1d

def resample_curve(curve_array, target_length):
    """
    Выравнивает количество точек (срезов) каротажной кривой (например, ПС) 
    до желаемой длины с помощью линейной интерполяции.
    
    Аргументы:
        curve_array (np.ndarray или pd.Series): исходный массив значений.
        target_length (int): желаемое количество точек (срезов) на выходе.
        
    Возвращает:
        np.ndarray: новый массив длиной target_length.
    """
    # Преобразуем входные данные в numpy array для универсальности
    curve_array = np.asarray(curve_array)
    original_length = len(curve_array)
    
    # Обработка краевых случаев
    if original_length == 0:
        raise ValueError("Входной массив не должен быть пустым.")
    if target_length <= 0:
        raise ValueError("Целевая длина должна быть больше нуля.")
    if original_length == 1:
        # Если точка всего одна, просто заполняем ей весь целевой массив
        return np.full(target_length, curve_array[0])

    # 1. Создаем исходную ось X (от 0 до 1 с шагами по длине исходного массива)
    x_original = np.linspace(0, 1, original_length)
    
    # 2. Создаем целевую ось X (от 0 до 1 с шагами по целевой длине)
    x_target = np.linspace(0, 1, target_length)
    
    # 3. Обучаем интерполятор на исходной оси X и значениях curve_array
    interpolator = interp1d(x_original, curve_array, kind='linear')
    
    # 4. Получаем новые значения для целевой оси X
    resampled_curve = interpolator(x_target)
    
    return resampled_curve

def morph_wells(well_a, well_b, weight=0.5):
    """
    Создает транзитную (переходную) скважину методом морфинга между двумя исходными.
    
    Аргументы:
        well_a (np.ndarray или pd.Series): данные первой скважины.
        well_b (np.ndarray или pd.Series): данные второй скважины.
        weight (float): вес интерполяции от 0.0 до 1.0 
                        (0.0 = 100% скважина А, 1.0 = 100% скважина Б).
                        
    Возвращает:
        np.ndarray: массив значений транзитной скважины.
    """
    well_a = np.asarray(well_a)
    well_b = np.asarray(well_b)
    
    len_a = len(well_a)
    len_b = len(well_b)
    
    # 1. Находим максимальную длину (эталон)
    max_len = max(len_a, len_b)
    
    # 2. Ресемплим массивы, если они короче максимальной длины
    if len_a < max_len:
        well_a_resampled = resample_curve(well_a, max_len)
    else:
        well_a_resampled = well_a
        
    if len_b < max_len:
        well_b_resampled = resample_curve(well_b, max_len)
    else:
        well_b_resampled = well_b
        
    # 3. Применяем формулу взвешенного среднего
    synthetic_well = well_a_resampled * (1.0 - weight) + well_b_resampled * weight
    
    return synthetic_well

def calculate_cv_stats(series):
    """Считает базовую статистику (мю, сигма, cv) массива/серии."""
    if len(series) == 0 or np.isnan(series).all(): 
        return 0, 0, 0
    mu = np.mean(series)
    sigma = np.std(series)
    cv = sigma / abs(mu) if mu != 0 else 0
    return mu, sigma, cv

def generate_synthetic_from_two_wells(well_a, well_b, weight, cv_multiplier):
    """
    Генерирует финальную транзитную кривую:
    1. Морфинг двух кривых.
    2. Генерация гауссовского шума (переиспользуя логику из app.py).
    3. Применение сглаживания скользящим окном 3.
    """
    import pandas as pd
    
    # 1. Получаем базовую транзитную кривую
    base_transit = morph_wells(well_a, well_b, weight)
    base_series = pd.Series(base_transit)
    
    # 2. Считаем базовую статистику
    mu, sigma, cv = calculate_cv_stats(base_series)
    
    # Считаем тренд (как в app.py)
    window = max(5, len(base_series) // 10)
    trend = base_series.rolling(window=window, center=True, min_periods=1).mean().values
    
    # 3. Применяем алгоритм нормального распределения Гаусса
    new_sigma = cv * cv_multiplier * abs(mu)
    noise = np.random.normal(0, new_sigma, size=len(base_transit))
    
    # Добавляем шум к тренду
    synth = trend + noise
    
    # 4. Применяем сглаживание скользящим окном 3
    synth_smoothed = pd.Series(synth).rolling(window=3, center=True, min_periods=1).mean().values
    
    # Если на краях появились NaN из-за скользящего окна, заполняем их исходными значениями
    nan_mask = np.isnan(synth_smoothed)
    if np.any(nan_mask):
        synth_smoothed[nan_mask] = base_transit[nan_mask]
        
    return synth_smoothed

if __name__ == "__main__":
    # Тест работы функций
    
    # 1. Тест ресемплинга
    print("--- Тест ресемплинга ---")
    original_data = np.random.rand(50)
    desired_length = 100
    resampled_data = resample_curve(original_data, desired_length)
    print(f"Длина исходного массива: {len(original_data)}")
    print(f"Длина массива после ресемплинга: {len(resampled_data)}")
    
    # 2. Тест морфинга двух скважин
    print("\n--- Тест морфинга ---")
    # Создаем две "скважины" разной длины
    # Скважина А (толстый пласт) - 120 точек, значения около 10
    well_a_data = np.random.normal(loc=10.0, scale=1.0, size=120)
    
    # Скважина Б (тонкий пласт) - 60 точек, значения около 20
    well_b_data = np.random.normal(loc=20.0, scale=1.0, size=60)
    
    # Скрещиваем с весом 0.5 (ровно посередине)
    synthetic = morph_wells(well_a_data, well_b_data, weight=0.5)
    
    print(f"Длина скважины А: {len(well_a_data)}")
    print(f"Длина скважины Б: {len(well_b_data)}")
    print(f"Длина транзитной скважины: {len(synthetic)} (совпадает с максимальной)")
    
    print(f"\nСреднее значение скважины А: {np.mean(well_a_data):.2f}")
    print(f"Среднее значение скважины Б: {np.mean(well_b_data):.2f}")
    print(f"Среднее значение транзитной скважины (ожидается ~15.0): {np.mean(synthetic):.2f}")

    # 3. Тест генерации синтетики с шумом
    print("\n--- Тест генерации синтетики с шумом ---")
    cv_multiplier = 1.5
    noisy_synthetic = generate_synthetic_from_two_wells(well_a_data, well_b_data, weight=0.5, cv_multiplier=cv_multiplier)
    
    print(f"Длина зашумленной кривой: {len(noisy_synthetic)}")
    print(f"Среднее значение (ожидается ~15.0): {np.mean(noisy_synthetic):.2f}")
    print(f"Стандартное отклонение исходной транзитной: {np.std(synthetic):.2f}")
    print(f"Стандартное отклонение зашумленной: {np.std(noisy_synthetic):.2f}")

