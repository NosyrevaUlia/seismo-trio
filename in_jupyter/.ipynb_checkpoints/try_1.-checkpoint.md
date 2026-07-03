# класс SeismicGenerator
## def __init__(self, wavelet_type='ricker', wavelet_freq=30, dt=0.002)
Класс задает параметры экземпляра по умолчанию:
wavelet_type='ricker' : тип волны по умолчанию - рикер, других пока нет
wavelet_freq=30 : частота 30
dt=0.002 : шаг времени в 0.002 с

wavelet = None : здесь потом будет сама волна
wavelet_time = None : здесь будет временная ось

## def _create_ricker_wavelet(self, length=0.128)
length=0.128 : длительность импульса во времени по умолчанию

## def _get_wavelet(self)
велвлета нет - создает новый с помощью _create_ricker_wavelet, велвлет есть - пропуск

## def generate_trace(self, reflectivity_series, noise_level=0.0)
Публичный метод, для создания синтетической сейсмической трассы:
reflectivity_series : массив чисел, показывающих как сильно каждый слой породы отражает волну
создает/получает волну с помощью _get_wavelet
свертывает коэфициенты отражения с импульсом
сокращает длину до reflectivity_series
добавляет шум
плюсует к трассе шум и возвращает результат