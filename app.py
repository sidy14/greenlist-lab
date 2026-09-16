"""ai-text-lab Streamlit UI - full version with RTL sidebar."""
import sys, pathlib, statistics
import streamlit as st

_ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))

from sentence_analyzer import SentenceAnalyzer, aggregate
from gauge import render_gauge, highlight_sentences
import ai_text_lab

st.set_page_config(
    page_title="AI-Text-Lab",
    page_icon="[lab]",
    layout="wide",
)
# --- Google Analytics ---
import streamlit.components.v1 as components
components.html(
    '''
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-L8WQVWHX6B"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-L8WQVWHX6B');
    </script>
    ''',
    height=0,
)
# --- end Google Analytics ---


# ---------------- Language FIRST (so CSS can use it) ----------------
if "lang" not in st.session_state:
    st.session_state.lang = "ar"

with st.sidebar:
    st.markdown("### 🌐 Language / اللغة")
    choice = st.radio("lang", ["العربية", "English"], horizontal=True,
                      label_visibility="collapsed", key="_lang")
    st.session_state.lang = "ar" if choice == "العربية" else "en"

AR = st.session_state.lang == "ar"
def T(ar, en): return ar if AR else en

# ---------------- Beautiful RTL CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --primary: #e74c3c;
        --primary-dark: #c0392b;
        --success: #27ae60;
        --warning: #f39c12;
        --bg-card: #ffffff;
        --bg-page: #f8fafc;
        --text-main: #1a202c;
        --text-muted: #64748b;
        --border: #e2e8f0;
        --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-lg: 0 10px 25px rgba(0,0,0,0.08);
        --radius: 14px;
        --radius-sm: 8px;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'IBM Plex Sans Arabic', 'Inter', -apple-system, sans-serif !important;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px;
    }

    /* ===== SIDEBAR on the RIGHT (RTL) ===== */
    [data-testid="stSidebar"] {
        right: 0 !important;
        left: auto !important;
        border-left: 1px solid var(--border);
        border-right: none;
        background: linear-gradient(180deg, #ffffff 0%, #fafbfc 100%);
        direction: rtl;
    }
    [data-testid="stSidebar"] > div:first-child {
        direction: rtl !important;
        text-align: right !important;
        padding-right: 1.25rem;
        padding-left: 1rem;
    }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        right: 0 !important;
        left: auto !important;
    }
    [data-testid="stSidebar"] * {
        direction: rtl;
        text-align: right;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] [data-baseweb="select"] {
        direction: rtl !important;
        text-align: right !important;
    }
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stRadio label {
        flex-direction: row-reverse !important;
        justify-content: flex-end !important;
    }

    /* ===== MAIN content RTL ===== */
    [data-testid="stAppViewContainer"] .main {
        direction: rtl;
        text-align: right;
    }
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label {
        text-align: right;
    }
    code, pre, .stCode, [data-testid="stCodeBlock"] {
        direction: ltr !important;
        text-align: left !important;
    }

    /* ===== Textarea ===== */
    .stTextArea textarea {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'IBM Plex Sans Arabic', sans-serif !important;
        font-size: 16px !important;
        line-height: 2 !important;
        border-radius: var(--radius) !important;
        border: 2px solid var(--border) !important;
        background: var(--bg-card) !important;
        padding: 16px !important;
        transition: border-color 0.2s;
    }
    .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 4px rgba(231, 76, 60, 0.1) !important;
    }

    /* ===== Titles ===== */
    h1 {
        font-weight: 700 !important;
        color: var(--text-main) !important;
        background: linear-gradient(135deg, #2c3e50 0%, #e74c3c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding-bottom: 0.5rem;
    }
    h2, h3 { color: var(--text-main) !important; font-weight: 600 !important; }

    /* ===== Metric cards ===== */
    .metric-card {
        background: var(--bg-card);
        border-radius: var(--radius);
        padding: 18px 22px;
        text-align: center;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    .metric-card .value {
        font-size: 30px; font-weight: 700;
        color: var(--text-main); line-height: 1.2;
    }
    .metric-card .label {
        font-size: 13px; color: var(--text-muted);
        margin-top: 6px; font-weight: 500;
    }

    /* ===== Buttons ===== */
    .stButton > button {
        border-radius: var(--radius) !important;
        padding: 12px 22px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: all 0.2s !important;
        border: none !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(231, 76, 60, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(231, 76, 60, 0.4) !important;
    }
    .stButton > button[kind="secondary"] {
        background: var(--bg-card) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: var(--primary) !important;
        color: var(--primary) !important;
    }

    .section-title {
        font-size: 18px; font-weight: 700;
        color: var(--text-main);
        margin-top: 28px; margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--primary);
        display: inline-block;
    }
    .verdict-badge {
        display: inline-block;
        padding: 10px 24px;
        border-radius: 24px;
        font-weight: 700;
        font-size: 15px;
        margin: 12px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------- helpers (defined BEFORE use) ----------------
def _verdict_label(score: float, ar: bool) -> str:
    if score > 0.7:
        return "نص مكتوب بالذكاء الاصطناعي" if ar else "Text written by AI"
    if score > 0.4:
        return "مشكوك فيه — قد يكون مزيجاً" if ar else "Uncertain — likely mixed"
    if score > 0.15:
        return "قد يكون بشرياً مع لمسات AI" if ar else "Probably human, some AI hints"
    return "نص بشري" if ar else "Human-written"


def _is_arabic(text: str) -> bool:
    arabic = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    return arabic / max(len(text), 1) > 0.15


# ---------------- header ----------------
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.markdown(f"<h1>{T('مختبر نص الذكاء الاصطناعي', 'AI-Text-Lab')}</h1>",
                unsafe_allow_html=True)
    st.caption(T("كشف وتنظيف النصوص المولَّدة — كشف على مستوى الجملة",
                 "Detect and clean AI text — sentence-level detection"))
with col_h2:
    st.markdown("<div style='text-align:right; padding-top:20px; color:#6c757d; font-size:13px;'>v0.3.0</div>",
                unsafe_allow_html=True)


# ---------------- sidebar settings ----------------
with st.sidebar:
    st.divider()
    st.markdown(f"### {T('خيارات العرض', 'Display options')}")
    show_gauge = st.checkbox(T("عرض العدّاد الدائري", "Show circular gauge"), value=True)
    show_highlight = st.checkbox(T("تلوين الجمل", "Highlight sentences"), value=True)
    show_table = st.checkbox(T("جدول مفصّل", "Detailed table"), value=False)
    min_chars = st.slider(T("الحد الأدنى لطول الجملة", "Min sentence length"),
                          10, 200, 40, 10)

    st.divider()
    st.markdown(f"### {T('التنظيف العربي', 'Arabic cleanup')}")
    ar_alef = st.checkbox(T("تطبيع الألف (أ إ آ ← ا)", "Normalize alef variants"),
                          value=False, key="ar_alef")
    ar_ya = st.checkbox(T("تطبيع الياء (ى ئ ← ي)", "Normalize ya variants"),
                        value=False, key="ar_ya")
    ar_tashkeel = st.checkbox(T("تجريد كل التشكيل", "Strip all tashkeel"),
                              value=False, key="ar_tashkeel")

    st.divider()
    st.markdown(f"### {T('الكشف الإحصائي', 'Statistical detection')}")
    use_stat = st.checkbox(
        T("تشغيل Green-List / SynthID", "Enable Green-List / SynthID"),
        value=False, key="use_stat",
        help=T("يحتاج تحميل نموذج. بطيء لكنه دقيق.",
               "Loads a model. Slower but accurate."),
    )
    stat_model = st.selectbox(
        T("النموذج", "Model"),
        ["openai-community/gpt2", "Qwen/Qwen2.5-0.5B-Instruct"],
        index=0, disabled=not use_stat, key="stat_model",
    )

    st.divider()
    st.caption(T("تعمل الأداة على العربية والإنجليزية.",
                 "Works on Arabic and English."))


# ---------------- input area ----------------
st.markdown(f"<div class='section-title'>{T('النص المُدخل', 'Input text')}</div>",
            unsafe_allow_html=True)

default_text = (
    "يُعدّ التعليم أساس نهضة المجتمعات وتقدّمها، فهو الوسيلة التي يكتسب بها "
    "الإنسان المعرفة والمهارات. ولا يقتصر التعليم على حفظ المعلومات، بل يهدف "
    "إلى تنمية التفكير النقدي وتعزيز القدرة على حل المشكلات.\n\n"
    "ذهبتُ أمس إلى السوق لأشتري بعض الفواكه. كانت الأسعار مرتفعة قليلاً هذه "
    "المرة، لكن البائع كان لطيفاً. تحدثنا قليلاً عن الطقس وعن الموسم الجديد."
)
text = st.text_area(T("الصق النص هنا", "Paste text here"),
                    value=default_text, height=260,
                    label_visibility="collapsed", key="input_text")

# stats
word_count = len(text.split())
char_count = len(text)
sent_count = len([s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()])
c1, c2, c3 = st.columns(3)
for col, val, lab in [
    (c1, word_count, T("كلمة", "words")),
    (c2, char_count, T("حرف", "chars")),
    (c3, sent_count, T("جملة (تقريبي)", "sentences (approx)")),
]:
    col.markdown(f"<div class='metric-card'><div class='value'>{val}</div>"
                 f"<div class='label'>{lab}</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- action buttons ----------------
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn1:
    detect_btn = st.button(T("🔍 تحقق من الذكاء الاصطناعي", "🔍 Detect AI"),
                           type="primary", use_container_width=True)
with col_btn2:
    clean_btn = st.button(T("🧹 تنظيف النص", "🧹 Clean text"),
                          use_container_width=True)
with col_btn3:
    clear_btn = st.button(T("🗑️ مسح", "🗑️ Clear"), use_container_width=True)


# ---------------- detection ----------------
@st.cache_resource
def get_analyzer():
    return SentenceAnalyzer()


if detect_btn and text.strip():
    with st.spinner(T("جاري التحليل على مستوى الجملة...", "Analyzing sentence by sentence...")):
        analyzer = get_analyzer()
        scores = analyzer.analyze(text, min_chars=min_chars)
        agg = aggregate(scores)

    # Balanced final score
    final_score = (agg["mean"] + agg["ai_ratio"]) / 2

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{T('النتيجة', 'Result')}</div>",
                unsafe_allow_html=True)

    col_g, col_s = st.columns([1, 2])

    with col_g:
        if show_gauge:
            st.markdown(render_gauge(final_score,
                                     label=T("ذكاء اصطناعي", "AI")),
                        unsafe_allow_html=True)
        verdict = _verdict_label(final_score, AR)
        color = "#e74c3c" if final_score > 0.7 else "#f39c12" if final_score > 0.4 \
                else "#f1c40f" if final_score > 0.15 else "#27ae60"
        st.markdown(f"<div style='text-align:center;'><span class='verdict-badge' "
                    f"style='background:{color}; color:white;'>{verdict}</span></div>",
                    unsafe_allow_html=True)

    with col_s:
        st.markdown(f"<div class='section-title'>{T('إحصائيات', 'Statistics')}</div>",
                    unsafe_allow_html=True)
        rows = [
            (T("عدد الجمل", "Sentences"), agg["n"]),
            (T("متوسط درجة AI", "Mean AI score"), f"{agg['mean']:.1%}"),
            (T("نسبة الجمل المُصنّفة AI", "Sentences tagged AI"),
             f"{agg['ai_ratio']:.1%}"),
            (T("نسبة الجمل المُصنّفة بشرية", "Sentences tagged Human"),
             f"{agg['human_ratio']:.1%}"),
        ]
        for lab, val in rows:
            st.markdown(f"<div style='display:flex; justify-content:space-between;"
                        f" padding:6px 0; border-bottom:1px solid #f0f0f0;'>"
                        f"<span style='color:#6c757d;'>{lab}</span>"
                        f"<span style='font-weight:600; color:#2c3e50;'>{val}</span>"
                        f"</div>", unsafe_allow_html=True)

    # Highlighted sentences
    if show_highlight:
        st.markdown(f"<div class='section-title'>"
                    f"{T('تلوين الجمل', 'Sentence highlighting')}</div>",
                    unsafe_allow_html=True)
        rtl = _is_arabic(text)
        st.markdown(highlight_sentences(scores, rtl=rtl), unsafe_allow_html=True)
        st.markdown(
            f"<div style='margin-top:16px; font-size:13px; color:#6c757d;'>"
            f"<span style='background:rgba(231,76,60,0.20); padding:3px 8px; "
            f"border-radius:4px; margin-left:8px;'>"
            f"{T('ذكاء اصطناعي', 'AI')}</span>"
            f"<span style='background:rgba(243,156,18,0.20); padding:3px 8px; "
            f"border-radius:4px; margin-left:8px;'>"
            f"{T('مشكوك', 'Uncertain')}</span>"
            f"<span style='background:rgba(39,174,96,0.15); padding:3px 8px; "
            f"border-radius:4px;'>{T('بشري', 'Human')}</span>"
            f"</div>", unsafe_allow_html=True)

    if show_table:
        st.markdown(f"<div class='section-title'>"
                    f"{T('تفصيل الجمل', 'Per-sentence detail')}</div>",
                    unsafe_allow_html=True)
        import pandas as pd
        df = pd.DataFrame([
            {
                T("الجملة", "Sentence"): s.text[:80] + ("..." if len(s.text) > 80 else ""),
                T("الدرجة", "Score"): f"{s.score:.1%}",
                T("الحكم", "Verdict"): s.verdict,
                T("الطول", "Length"): s.length,
            }
            for s in scores
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

elif clean_btn and text.strip():
    from analyzer import Analyzer
    if ar_alef or ar_ya or ar_tashkeel:
        analyzer = Analyzer(
            arabic_alef=ar_alef,
            arabic_ya=ar_ya,
            arabic_strip_tashkeel=ar_tashkeel,
        )
        result = analyzer.analyze(text).cleaned_text
    else:
        result = ai_text_lab.clean(text)
    st.markdown(f"<div class='section-title'>"
                f"{T('النص المنظّف', 'Cleaned text')}</div>",
                unsafe_allow_html=True)
    if ar_alef or ar_ya or ar_tashkeel:
        st.caption(T("تم تطبيق خيارات التنظيف العربي المحدّدة.",
                     "Applied selected Arabic cleanup options."))
    st.code(result, language=None)
    st.download_button(T("📥 تنزيل", "📥 Download"), result,
                       file_name="cleaned.txt", mime="text/plain")

elif clear_btn:
    st.rerun()


st.markdown("---")
st.caption("ai-text-lab v0.3.0  |  github.com/sidy14/greenlist-lab")