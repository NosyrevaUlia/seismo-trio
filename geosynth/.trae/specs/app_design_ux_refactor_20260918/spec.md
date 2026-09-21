# GeoSynth app.py — Рефакторинг дизайна, UX, интерактивная карта + починка progress-bar

## Overview
- **Summary**: Кардинально обновить дизайн Streamlit-приложения app.py: светлая академическая тема без эмодзи, двусторонняя синхронизация точек между картой и таблицей координат, исправление AttributeError progress-bar при пакетной генерации. Выдать полный листинг app.py готовым к запуску.
- **Purpose**: Пользователь жалуется на (1) плохую читаемость тёмной темы, (2) нестрогий вид из-за эмодзи, (3) неудобный ввод координат только из таблицы, (4) падение пакетной генерации AttributeError NoneType.progress.
- **Target Users**: Геофизики/петрофизики, генерирующие синтетические каротажные наборы для ML. Работают с Режимом 3 (IDW + CVAE) как с основным инструментом.

## Goals
- G1: Полная миграция CSS на светлую тему с высоким контрастом. Цветовая гамма: фоны белый/светло-серый, текст #212529 без блеклых оттенков. Plotly-карта на светлом фоне с чёрными осями и сеткой.
- G2: Абсолютно чистый строгий интерфейс. Ни одного эмодзи ни в UI-элементах, ни в сообщениях, ни в markdown-заголовках.
- G3: Интерактивная карта как полноценный инструмент ввода. Двойной клик = добавление точки, drag-and-drop = перемещение точек. Двусторонняя синхронизация с session_state.idw_points и st.data_editor.
- G4: Стабильная пакетная генерация режима 3 без падений AttributeError: 'NoneType' object has no attribute 'progress'.
- G5: 100% обратная совместимость логики: все 3 режима работы, 6 методов каротажа, гибридный алгоритм IDW+CVAE, экспорт LAS/ZIP, насыщение флюидами.

## Non-Goals
- NG1: Не менять математику генерации (формула spatial_uncertainty, сигнатура generate_cv_synthetic noise_scale, архитектура GeologicalCVAE).
- NG2: Не менять экспортные мнемоники LAS (GK/PS/BK/IK/NKT/MKZ) и единицы измерения.
- NG3: Не менять загрузку/парсинг LAS-файлов и привязку мнемоник.
- NG4: Не добавлять новые режимы или новые кривые сверх текущих 6.

## Background & Context
- Файл: app.py (≈1460 строк после предыдущего рефактора). Референсные данные: cvae_weights.pt в корне.
- Предыдущие баги, исправленные ранее: (а) NameError generate_cv_synthetic не в глобальной области → перенесены 10 функций в начало; (б) Режим 3 ранний st.stop блокировал data_editor/map → перестановка блоков.
- Текущее окружение: Python .venv, streamlit 1.x, torch 2.14 CPU, plotly 7.1, lasio 0.3x, scipy, pandas, matplotlib.
- Известная проблема progress-bar (пункт 4 ТЗ): строка 1309 использует `with st.progress(...) as prog_bar:` — в некоторых версиях Streamlit контекст-менеджер progress не возвращает объект через `as` → prog_bar = None → строка 1374 `prog_bar.progress(...)` AttributeError.

## Functional Requirements
### Тема Дизайн (пункт 1 ТЗ)
- **FR-1**: Стилизовать Streamlit UI (заголовки, сайдбар, main, формы, экспандеры, file_uploader, слайдеры, кнопки, input, selectbox, expander, st.data_editor, st.info/warning/error/success boxes) в СВЕТЛОЙ палитре с контрастным чёрным текстом #212529. Никаких блеклых цветов текста.
- **FR-2**: Plotly go.Figure для режима 3 должен использовать `plot_bgcolor="#FFFFFF"`, `paper_bgcolor="#F8F9FA"`, оси linecolor #212529, tickfont #212529, gridcolor #DEE2E6. Опорные точки = синие #0D6EFD outline #052C65. Синтетические = красные #DC3545 outline #842029.
- **FR-3**: matplotlib планшеты 6-трекового сравнения (все 3 режима) также рендерятся на светлом фоне (удалить `plt.style.use('dark_background')`, использовать `seaborn-v0_8-whitegrid` или дефолтный светлый). Текст осей/легенд #212529.
- **FR-4**: Удалить вшитые inline-цвета hex из markdown (например `<h3 style='color: #10B981'>`) — заменить на нейтральный #212529 или класс.

### Строгий стиль (пункт 2 ТЗ)
- **FR-5**: Полный регреп всего файла: удалить ВСЕ эмодзи Unicode из строковых литералов (заголовки, st.markdown, st.info/warning/error/success, кнопки, лейблы слайдеров, expander-заголовки, sidebar headings).
- **FR-6**: Если после удаления эмодзи заголовок становится неясным, добавить 1-2 слово на русском языке вместо эмодзи.

### Интерактивная карта (пункт 3 ТЗ)
- **FR-7**: Карта Режим 3 поддерживает scrollZoom=реальное колёсико мыши (Plotly `config=dict(scrollZoom=True, displayModeBar=True)`).
- **FR-8**: Двойной клик ЛКМ по пустому месту карты → добавить новую строку в таблицу координат с авто-ID "SYNTH_N+1" и координатами клика (в системе координат карты X/Y метры). Таблица и session_state.idw_points синхронно обновляются.
- **FR-9**: Drag-and-drop зажатой ЛКМ по существующей точке (красный ромб синтетики) → переместить её. Новые координаты записываются в session_state.idw_points и в таблицу data_editor.
- **FR-10**: Двусторонняя связь: пользователь редактирует X/Y в st.data_editor → карта перерисовывает красные ромбы на новые координаты без перезагрузки страницы. Обратная связь FR-8/FR-9 обновляет таблицу из карты.
- **FR-11**: Если библиотеки streamlit-plotly-events отсутствуют → **установить pip install** в .venv. Если нет возможности получить события Plotly → использовать streamlit-folium (установить pip install folium streamlit-folium в .venv) и folium.ClickForMarker/Draw с обработкой точек в st.session_state. Какой-либо из вариантов ДОЛЖЕН реально доставлять координаты клика в код.

### AttributeError прогресс-бара (пункт 4 ТЗ)
- **FR-12**: Заменить контекст `with st.progress(0, text=...) as prog_bar:` на явное создание через `prog_bar = st.progress(0, text="...")` или аналоги с гарантией non-None объекта. Обновлять `.progress(fraction, text=...)` в цикле.
- **FR-13**: Перед первой записью в .progress() проверять `if prog_bar is not None:` — defensive программирование.
- **FR-14**: После завершения цикла прогресс-бара сбросить текст на "Готово", доля 1.0.

### Обратная совместимость
- **FR-15**: Режимы 1 и 2 продолжают работать без изменений логики генерации (CSS только визуально меняется).
- **FR-16**: Режим 3 гибридный алгоритм spatial_uncertainty * CVAE-influence для ГК/ПС и noise_scale для БК/ИК/НКТ/МКЗ остаётся без изменений математики.
- **FR-17**: ZIP-экспорт и X/Y LAS header injection работают идентично.

## Non-Functional Requirements
- **NFR-1**: Синтаксис Python. py_compile exit=0.
- **NFR-2**: Streamlit boot без ModuleNotFoundError/ImportError. Все новые пакеты (folium, streamlit-folium, streamlit-plotly-events) должны быть pip-installed в .venv.
- **NFR-3**: Страница загружается < 10 сек на холостом запуске без LAS.
- **NFR-4**: В консоли браузера 0 JS errors уровня Error.
- **NFR-5**: Каждый из 3 радио-режимов переключается и показывает ожидаемую структуру сайдбара и центральной панели.
- **NFR-6**: Отсутствие эмодзи: автоматический grep всего app.py по regex-Unicode эмодзи возвращает 0 совпадений.

## Constraints
- **Технические**: Python 3.11 .venv, Streamlit 1.x. Нельзя переходить с Streamlit на Dash/FastAPI. Нельзя менять cvae_weights.pt. Нельзя менять 6 кривых ГК/ПС/БК/ИК/НКТ/МКЗ.
- **Бизнес**: Интерфейс строгий академический — РЕАЛЬНО НИ ОДНОГО эмодзи в UI-строках. Текст #212529 на светлом фоне — минимум WCAG AA контраст 4.5:1.
- **Зависимости**: Если нужны streamlit-plotly-events/folium/streamlit-folium — pip install в .venv до запуска валидации.

## Assumptions
- A1: streamlit-folium или streamlit-plotly-events поставляют реальные callback-координаты кликов/драг-событий и это работает со Streamlit 1.3x+.
- A2: streamlit 1.x st.progress() возвращает DeltaGenerator non-None если вызывать напрямую без контекст-менеджера.
- A3: Пользователь примет любую из двух библиотек (plotly-events или folium+streamlit-folium) для интерактивной карты, по критерию «реально работает двусторонняя синхронизация».

## Acceptance Criteria

### AC-1: Светлая тема CSS и читаемость
- **Type**: `rule`
- **Given**: Streamlit запущен на :8501, пользователь открыл режим 1 или 3
- **When**: Делаем browser_snapshot страницы и инспектируем body/stAppViewContainer computed-style
- **Then**: `background-color` находится в наборе {#FFFFFF, #F8F9FA, #F1F3F5, #E9ECEF}, `color` body текста = #212529 или rgba(33,37,41,1). Ни один текстовый элемент не имеет color с альфой < 0.85 и не светлее #495057.
- **Pass Condition**: оба условия истинны + Plotly-карта plot_bgcolor="#FFFFFF" paper_bgcolor="#F8F9FA" в go.Layout dict
- **Evidence**: browser_snapshot refs с цветами + grep app.py на `plot_bgcolor=`, `paper_bgcolor=`, `plt.style.use('dark_background')` (последний должен отсутствовать)

### AC-2: Ноль эмодзи в коде приложения
- **Type**: `rule`
- **Given**: файл app.py существует
- **When**: запускаем Python regex-grep по всем строкам app.py: range `[\U0001F000-\U0001FFFF]` + диапазоны общих эмодзи + символы стрелок 👉👈🎨🧬🌍 и т.д.
- **Then**: Найдено 0 совпадений (исключая комментарии разрешать нельзя — и строки тоже чистые)
- **Pass Condition**: `count == 0`
- **Evidence**: вывод команды grep с count=0

### AC-3: Карта интерактивная — добавление точек двойным кликом, drag-and-drop, двусторонняя синхронизация
- **Type**: `rule`
- **Given**: Пользователь в Режиме 3, 0 LAS, карта отрендерилась с точкой SYNTH_1 = (250, 250)
- **When 1**: Пользователь делает double-клик по координатам карты (400, 500)
- **Then 1**: В session_state.idw_points добавляется новая строка ID=SYNTH_2 X=400 Y=500; в st.data_editor при следующем snapshot она видна как вторая строка; на карте второй красный ромб в (400, 500)
- **When 2**: Пользователь drag-and-drop SYNTH_1 из (250,250) в (100, 150)
- **Then 2**: session_state.idw_points[0]['X'] = 100, ['Y'] = 150; таблица data_editor обновилась; карта перерисовала SYNTH_1 в (100,150)
- **When 3**: Пользователь редактирует ячейку data_editor X для SYNTH_2 с 400 → 700
- **Then 3**: На карте ромб SYNTH_2 переехал в X=700; session_state.idw_points синхрон
- **Pass Condition**: All 3 sub-Then истинны
- **Evidence**: Скрипт Python eval session_state + браузерный snapshot data_editor grid и plotly legend markers coords

### AC-4: Scroll Zoom включён на карте
- **Type**: `rule`
- **Given**: Карта отрендерена в режиме 3
- **When**: смотрим в Plotly config / folium tiles параметры
- **Then**: scrollZoom = True или folium zoom_control=колесо=on
- **Pass Condition**: zoom параметр включён
- **Evidence**: grep app.py на `scrollZoom` / `zoomControl` / `wheel` с true

### AC-5: AttributeError progress-bar исправлен
- **Type**: `rule`
- **Given**: Режим 3, загружены 2 LAS-файла, valid_points содержит 10 точек, cvae_net не None
- **When**: запускается пакетная генерация
- **Then**: Прогресс-бар монотонно растёт от 0 до 1.0, текст меняется "Подготовка..." → "Генерация SYNTH_1..." → "Готово"; Исключений AttributeError NoneType.progress не возникает. st.success отображается с N=10. ZIP кнопка рендерится.
- **Pass Condition**: генерация доходит до конца без исключений и без None.progress
- **Evidence**: stdout Streamlit лог без traceback + browser_snapshot showing st.success banner с текстом числа

### AC-6: Py compile и Streamlit boot clean
- **Type**: `rule`
- **Given**: свежая запись app.py на диск
- **When**: `python -m py_compile app.py` → `exit 0`; затем `python -m streamlit run app.py --server.headless true --port 8501`
- **Then**: Uvicorn started on :::8501, 0 traceback в stdout за первые 30 секунд
- **Pass Condition**: оба шага успешны
- **Evidence**: py_compile exit=0 + Uvicorn banner в stdout

### AC-7: Математика генерации неизменна
- **Type**: `rule`
- **Given**: app.py после рефакторинга
- **When**: grep по 3 строкам: spatial_uncertainty формула; generate_cv_synthetic noise_scale вызов для RES/COND/NPHI/GMZ; CVAE alpha = cvae_influence * spatial_uncertainty
- **Then**: Точное совпадение формул с оригиналом: `1.2 * (1 - max_w)` clip(0.15,1.0); noise_scale передаётся в generate_cv_synthetic; alpha умножает gk_noise/ps_noise и прибавляет к det_trend.
- **Pass Condition**: все 3 формулы сохранены дословно
- **Evidence**: grep content -n 3 формулы

### AC-8: Читаемость контраст (rubric)
- **Type**: `rubric`
- **Dimension**: Контрастность и читаемость интерфейса
- **Scale**: 1-5
- **Anchors**: 1 = тёмно-серая плашка с бледно-серым текстом контраст < 2:1; 3 = стандартная Streamlit дефолтная; 5 = каждый текст контраст ≥ 4.5:1 относительно своего фона, заголовки жирные, лейблы слайдеров явные
- **Pass Threshold**: >= 4
- **Evidence**: browser_snapshot compute-style нескольких элементов (label slider, heading h2, st.info текст, sidebar heading) — проверка контраста manual

### AC-9: Строгость интерфейса (rubric)
- **Type**: `rubric`
- **Dimension**: Академичность строгость отсутствие декоративных элементов
- **Scale**: 1-5
- **Anchors**: 1 = заголовки усыпаны эмодзи каждый; 3 = половина заголовков без эмодзи; 5 = ноль эмодзи в UI, иконок нет, текст-ориентированный интерфейс как у научного приложения
- **Pass Threshold**: 5
- **Evidence**: regex grep count=0 + browser_snapshot all headings/buttons

## Open Questions
- [ ] Q1: Какая библиотека предпочтительнее для интерактивной карты: **streamlit-plotly-events** (Plotly) или **folium + streamlit-folium** (Leaflet/Folium)? — если пользователь не ответил до Approve, реализуем оба варианта с приоритетом Plotly (как в оригинальном ТЗ п.1 было Plotly).
