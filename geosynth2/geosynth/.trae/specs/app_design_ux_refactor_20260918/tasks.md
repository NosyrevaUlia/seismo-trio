# GeoSynth app.py — Рефакторинг дизайна, UX, интерактивная карта + починка progress-bar: Implementation Plan

## Task 1: Миграция CSS на светлую тему + Plotly/matplotlib светлые фоны
- **Status**: `pending`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - Заменить весь CSS-блок Modern Dark Dashboard на новый: body/main фон #FFFFFF или #F8F9FA, текст строго #212529, лейблы #212529, input/selectbox фон белый #FFFFFF, рамка #DEE2E6, кнопка submit синий #0D6EFD hover #0B5ED7, сайдбар фон #F1F3F5 рамка справа #DEE2E6, экспандеры #F8F9FA.
  - Заменить inline style hex: `color: #10B981` → #212529.
  - Plotly go.Figure layout (режим 3 карта): `plot_bgcolor="#FFFFFF"`, `paper_bgcolor="#F8F9FA"`, `xaxis=dict(linecolor="#212529", gridcolor="#DEE2E6", tickfont=dict(color="#212529"), titlefont=dict(color="#212529"))`. Маркеры синие #0D6EFD line #052C65, красные #DC3545 line #842029.
  - Удалить `plt.style.use('dark_background')`; заменить на `plt.style.use('seaborn-v0_8-whitegrid')` или дефолт + fig.patch.set_facecolor('white'), ax.set_facecolor('white').
- **Acceptance Criteria Addressed**: AC-1, AC-8
- **Test Requirements**:
  - `rule` TR-1.1: grep app.py для `plt.style.use('dark_background')` count=0; grep `plot_bgcolor="#FFFFFF"` count>=1; grep `paper_bgcolor="#F8F9FA"` count>=1; grep `color: #212529` или `color:#212529` в CSS-блок count>=5. Evidence: stdout grep.
  - `rule` TR-1.2: py_compile app.py exit=0 после правок. Evidence: stdout.
  - `rubric` TR-1.3: контрастность CSS; scale 1-5; anchors 1=яркие блёклые, 3=средне, 5=каждый #212529 текст на #F8F9FA контраст >= 4.5; threshold >=4; Evidence: визуальный review CSS blob.
- **Notes**: Оставить все CSS классы (block-container / status-box / badge-done / plot-container) только в светлых тонах.

## Task 2: Удаление ВСЕХ эмодзи из всех строковых UI-литералов
- **Status**: `pending`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - Рекурсивный grep app.py по диапазонам Unicode эмодзи и символов-стрелок/значков: `[\U0001F000-\U0001FFFF]`, `[\U00002600-\U000027BF]`, `[\U0001F300-\U0001F9FF]`, одиночные emoji символы 🎨🧬🌍📍📋🗺️📦🚀🧠♻️✅⚠️ℹ️❌👉👈👇👆🎯🧪📊📈📉🔲🗑️ и мн. др.
  - Удалить эмодзи из: `st.sidebar.markdown`, `st.markdown`, заголовки st.info/warning/error/success, button лейблы, expander заголовки, sidebar radio option labels, selectbox/h3 текст.
  - Если теряется смысл добавить 1-2 русских слова, например "Внешний вид" вместо "🎨 Внешний вид".
- **Acceptance Criteria Addressed**: AC-2, AC-9
- **Test Requirements**:
  - `rule` TR-2.1: Python regex-scan app.py по эмодзи диапазонам (включая U+1Fxxx через UTF-8) returns 0 matches. Evidence: stdout scan script result.
  - `rubric` TR-2.2: строгость интерфейса; scale 1-5; anchors 1=эмодзи много 3=немного 5=ноль; threshold 5; Evidence: TR-2.1 pass + ручной browser snapshot review

## Task 3: Установить библиотеку интерактивной карты (Plotly events или Folium) в .venv
- **Status**: `pending`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - Шаг 1: pip install `streamlit-plotly-events==0.0.7` или последнюю версию в .venv.
  - Если TR-3.1 не работает (события не прилетают): шаг 2: pip install `folium`, `streamlit-folium`.
  - Выбрать ту библиотеку, у которой получаются реальные координаты клика/драг.
- **Acceptance Criteria Addressed**: AC-3, AC-4, NFR-2
- **Test Requirements**:
  - `rule` TR-3.1: Import test `from streamlit_plotly_events import plotly_events` или `from streamlit_folium import st_folium` returns ImportError=0 в .venv python. Evidence: stdout.
  - `rule` TR-3.2: scrollZoom=True в Plotly config или folium zoom включен по умолчанию. Evidence: grep app.py.

## Task 4: Реализовать двустороннюю синхронизацию Карта ↔ session_state.idw_points ↔ st.data_editor (double-click + drag-and-drop)
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 3
- **Description**:
  - Инициализация session_state.idw_points если не существует: df с колонками ["ID", "X", "Y"], строка SYNTH_1=(250,250).
  - st.data_editor редактирует session_state.idw_points через key="idw_points_editor" или через callback on_change.
  - Plotly Events или streamlit-folium st_folium:
    - Double-click или click (в зависимости от API) по пустому месту карты: определяем координаты клика; ищем в session_state.idw_points свободный авто-ID SYNTH_N; добавляем строку; set session state rerun.
    - Drag-and-drop: если библиотека поддерживает событие dragend — ловим новый X/Y и обновляем строку по ID точки. Если нет drag события — альтернатива: select point + смещение delta в session_state через click с selection; либо используем st_folium DrawExport polygon/marker с feature editor.
  - Каждую итерацию рисуем карту заново из session_state.idw_points (синие опорные, красные синтетические ромбы).
- **Acceptance Criteria Addressed**: AC-3, AC-4
- **Test Requirements**:
  - `rule` TR-4.1: При старте режима 3 без LAS session_state.idw_points содержит ровно 1 строку SYNTH_1 (250, 250). Evidence: st.write(session_state.idw_points) / script eval.
  - `rule` TR-4.2: Поведение FR-8/9/10: ручная проверка или мок session_state путем set_state + rerun; проверка что датафрейм синхронен с маркерами. Evidence: 2 snapshot до/после.
  - `rubric` TR-4.3: UX интерактивности карты; scale 1-5; 1=только таблица 3=кликабельно добавить точку но нет драга 5=и двойной клик и драг и таблица синхрон; threshold >=4.

## Task 5: Устранить AttributeError 'NoneType'.progress (замена ontext-manager st.progress на явный объект)
- **Status**: `pending`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - Строка 1309 оригинала: `with st.progress(0, text="Подготовка...") as prog_bar:` → заменить на:
    - `prog_placeholder = st.empty()`
    - `prog_bar = prog_placeholder.progress(0, text="Подготовка генерации...")`
    - Внутри цикла: защитить `if prog_bar is not None: prog_bar.progress(..., text=...)`
    - После цикла: установить progress(1.0, text="Генерация завершена") → st.success.
  - Так же проверить: есть ли стоп-условие AttributeError в режиме 1 или 2 аналогичного типа? Если есть — исправить там же.
- **Acceptance Criteria Addressed**: AC-5
- **Test Requirements**:
  - `rule` TR-5.1: grep app.py на `with st.progress(...) as ...` → count=0. Evidence: stdout.
  - `rule` TR-5.2: grep `prog_bar is not None` перед `.progress(` → count>=1 OR prog_bar гарантированно создаётся до цикла (traceback py_compile + code review creation).
  - `rule` TR-5.3: После цикла `.progress(1.0, text="Генерация завершена")` вызван. Evidence: grep.

## Task 6: Сквозная валидация: py_compile, Streamlit boot clean, browser smoke на 3 режимах + grep ноль эмодзи
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Tasks 1,2,3,4,5
- **Description**:
  - 6.1 py_compile exit=0
  - 6.2 Kill старые python/streamlit, старт server :8501
  - 6.3 Browser navigate http://localhost:8501, дождаться GeoSynth title
  - 6.4 Проверить UI: ноль эмодзи в заголовках (смотрим snapshot)
  - 6.5 Переключиться на Режим 3 «Пространственная интерполяция»: data_editor + карта Plotly/Folium отрендерены со светлой темой (белый фон, чёрная сетка)
  - 6.6 Убедиться, что карта имеет scrollZoom=true или folium zoom колесом работает
  - 6.7 Console messages error count=0
- **Acceptance Criteria Addressed**: AC-1, AC-2, AC-4, AC-6, AC-8, AC-9
- **Test Requirements**:
  - `rule` TR-6.1: py_compile exit=0.
  - `rule` TR-6.2: Uvicorn banner "started on :::8501" stdout.
  - `rule` TR-6.3: browser_snapshot title="GeoSynth Professional", 3 radio options, ни одного ref с emoji-текстом, фон белый/светло-серый.
  - `rule` TR-6.4: browser_console_messages error level 0.
  - `rubric` TR-6.5: Общее качество дизайна; scale 1-5; threshold >=4.

## Task 7: Независимый review (Review gate). Выполняется на фазе Review после tasks 1-6 completed
- **Status**: `pending`
- **Priority**: medium
- **Depends On**: Tasks 1-6
- **Description**: Read-only review с нуля: проверить каждую AC → создать review.md и записать CP-R1..CP-R6, CP-U1..CP-U3.
- **Acceptance Criteria Addressed**: Все AC
- **Test Requirements**: Отсутствуют. Review отдельно.
