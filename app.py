"""
ai-text-lab Streamlit UI - bilingual (English / Arabic).

Run: streamlit run app.py
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import ai_text_lab


# ------------------------------------------------------------------ i18n
T = {
    "en": {
        "lang_label": "Language",
        "title": "AI-Text-Lab",
        "subtitle": "Detect and clean AI-text surface artifacts and statistical watermarks (Latin + Arabic)",
        "settings": "Settings",
        "stat_detection": "Statistical detection (loads torch + model)",
        "model_label": "Model for statistical detection",
        "normalize_alef": "Normalize alef variants (alef-hamza to bare alef)",
        "normalize_ya": "Normalize ya variants",
        "strip_tashkeel": "Strip ALL tashkeel",
        "tip": "Paste text on the left, see the report and cleaned output on the right.",
        "input_h": "Input",
        "analyze": "Analyze",
        "analyzing": "Analyzing...",
        "report_h": "Report",
        "surface_findings": "Surface findings",
        "arabic_findings": "Arabic findings",
        "chars_changed": "Chars net change",
        "surface_artifacts": "Surface artifacts (Latin)",
        "arabic_artifacts": "Arabic artifacts",
        "stat_watermark": "Statistical watermark",
        "cleaned_text": "Cleaned text",
        "download": "Download cleaned text",
        "diff_h": "Diff (character-level)",
        "not_arabic": "Text is not Arabic (or <15% Arabic chars)",
        "none_detected": "None detected",
        "not_attempted": "Not attempted (enable in sidebar)",
        "watermarked": "WATERMARKED",
        "not_watermarked": "not watermarked",
        "tashkeel_density": "Tashkeel density",
        "about": "About",
        "about_body": "ai-text-lab v0.2.0 - detect and clean surface artifacts and statistical watermarks in Latin and Arabic text.",
        "footer": "ai-text-lab v0.2.0  |  github.com/sidy14/greenlist-lab",
    },
    "ar": {
        "lang_label": "\u0627\u0644\u0644\u063a\u0629",
        "title": "\u0645\u062e\u062a\u0628\u0631 \u0646\u0635 \u0627\u0644\u0630\u0643\u0627\u0621 \u0627\u0644\u0627\u0635\u0637\u0646\u0627\u0639\u064a",
        "subtitle": "\u0643\u0634\u0641 \u0648\u062a\u0646\u0638\u064a\u0641 \u0622\u062b\u0627\u0631 \u0627\u0644\u0630\u0643\u0627\u0621 \u0627\u0644\u0627\u0635\u0637\u0646\u0627\u0639\u064a \u0627\u0644\u0633\u0637\u062d\u064a\u0629 \u0648\u0627\u0644\u0639\u0644\u0627\u0645\u0627\u062a \u0627\u0644\u0645\u0627\u0626\u064a\u0629 \u0627\u0644\u0625\u062d\u0635\u0627\u0626\u064a\u0629 (\u0644\u0627\u062a\u064a\u0646\u064a + \u0639\u0631\u0628\u064a)",
        "settings": "\u0627\u0644\u0625\u0639\u062f\u0627\u062f\u0627\u062a",
        "stat_detection": "\u0627\u0644\u0643\u0634\u0641 \u0627\u0644\u0625\u062d\u0635\u0627\u0626\u064a (\u064a\u062d\u0645\u0651\u0644 \u0627\u0644\u0646\u0645\u0648\u0630\u062c)",
        "model_label": "\u0646\u0645\u0648\u0630\u062c \u0627\u0644\u0643\u0634\u0641",
        "normalize_alef": "\u062a\u0637\u0628\u064a\u0639 \u0627\u0644\u0623\u0644\u0641",
        "normalize_ya": "\u062a\u0637\u0628\u064a\u0639 \u0627\u0644\u064a\u0627\u0621",
        "strip_tashkeel": "\u062a\u062c\u0631\u064a\u062f \u0643\u0644 \u0627\u0644\u062a\u0634\u0643\u064a\u0644",
        "tip": "\u0627\u0644\u0635\u0642 \u0627\u0644\u0646\u0635 \u0639\u0644\u0649 \u0627\u0644\u064a\u0645\u064a\u0646\u060c \u0648\u0634\u0627\u0647\u062f \u0627\u0644\u062a\u0642\u0631\u064a\u0631 \u0648\u0627\u0644\u0646\u062a\u064a\u062c\u0629 \u0639\u0644\u0649 \u0627\u0644\u064a\u0633\u0627\u0631.",
        "input_h": "\u0627\u0644\u0645\u062f\u062e\u0644\u0627\u062a",
        "analyze": "\u062a\u062d\u0644\u064a\u0644",
        "analyzing": "\u062c\u0627\u0631\u064a \u0627\u0644\u062a\u062d\u0644\u064a\u0644...",
        "report_h": "\u0627\u0644\u062a\u0642\u0631\u064a\u0631",
        "surface_findings": "\u0627\u0644\u0622\u062b\u0627\u0631 \u0627\u0644\u0633\u0637\u062d\u064a\u0629",
        "arabic_findings": "\u0627\u0644\u0622\u062b\u0627\u0631 \u0627\u0644\u0639\u0631\u0628\u064a\u0629",
        "chars_changed": "\u0635\u0627\u0641\u064a \u0627\u0644\u062a\u063a\u064a\u064a\u0631",
        "surface_artifacts": "\u0627\u0644\u0622\u062b\u0627\u0631 \u0627\u0644\u0633\u0637\u062d\u064a\u0629 (\u0644\u0627\u062a\u064a\u0646\u064a\u0629)",
        "arabic_artifacts": "\u0627\u0644\u0622\u062b\u0627\u0631 \u0627\u0644\u0639\u0631\u0628\u064a\u0629",
        "stat_watermark": "\u0627\u0644\u0639\u0644\u0627\u0645\u0629 \u0627\u0644\u0645\u0627\u0626\u064a\u0629 \u0627\u0644\u0625\u062d\u0635\u0627\u0626\u064a\u0629",
        "cleaned_text": "\u0627\u0644\u0646\u0635 \u0627\u0644\u0645\u0646\u0638\u0651\u0641",
        "download": "\u062a\u0646\u0632\u064a\u0644 \u0627\u0644\u0646\u0635 \u0627\u0644\u0645\u0646\u0638\u0651\u0641",
        "diff_h": "\u0627\u0644\u0641\u0631\u0642 (\u0639\u0644\u0649 \u0645\u0633\u062a\u0648\u0649 \u0627\u0644\u0645\u062d\u0631\u0641)",
        "not_arabic": "\u0627\u0644\u0646\u0635 \u0644\u064a\u0633 \u0639\u0631\u0628\u064a\u0627\u064b (\u0623\u0648 \u0623\u0642\u0644 \u0645\u0646 15\u066a \u0645\u062d\u0627\u0631\u0641 \u0639\u0631\u0628\u064a\u0629)",
        "none_detected": "\u0644\u0645 \u064a\u064f\u0643\u062a\u0634\u0641 \u0634\u064a\u0621",
        "not_attempted": "\u0644\u0645 \u064a\u064f\u062c\u0631\u064e (\u0641\u0639\u0651\u0644\u0647 \u0641\u064a \u0627\u0644\u0634\u0631\u064a\u0637 \u0627\u0644\u062c\u0627\u0646\u0628\u064a)",
        "watermarked": "\u0645\u064f\u0639\u0644\u0651\u064e\u0645 \u0628\u0645\u0627\u0626\u064a\u0629",
        "not_watermarked": "\u063a\u064a\u0631 \u0645\u064f\u0639\u0644\u0651\u064e\u0645",
        "tashkeel_density": "\u0643\u062b\u0627\u0641\u0629 \u0627\u0644\u062a\u0634\u0643\u064a\u0644",
        "about": "\u062d\u0648\u0644",
        "about_body": "\u0645\u062e\u062a\u0628\u0631 \u0646\u0635 \u0627\u0644\u0630\u0643\u0627\u0621 \u0627\u0644\u0627\u0635\u0637\u0646\u0627\u0639\u064a v0.2.0 - \u0643\u0634\u0641 \u0648\u062a\u0646\u0638\u064a\u0641 \u0627\u0644\u0622\u062b\u0627\u0631 \u0627\u0644\u0633\u0637\u062d\u064a\u0629 \u0648\u0627\u0644\u0639\u0644\u0627\u0645\u0627\u062a \u0627\u0644\u0645\u0627\u0626\u064a\u0629 \u0641\u064a \u0627\u0644\u0646\u0635\u0648\u0635 \u0627\u0644\u0644\u0627\u062a\u064a\u0646\u064a\u0629 \u0648\u0627\u0644\u0639\u0631\u0628\u064a\u0629.",
        "footer": "ai-text-lab v0.2.0  |  github.com/sidy14/greenlist-lab",
    },
}


# ------------------------------------------------------------- state
if "lang" not in st.session_state:
    st.session_state.lang = "en"

st.set_page_config(
    page_title="AI-Text-Lab" if st.session_state.lang == "en" else "\u0645\u062e\u062a\u0628\u0631 \u0646\u0635 \u0627\u0644\u0630\u0643\u0627\u0621 \u0627\u0644\u0627\u0635\u0637\u0646\u0627\u0639\u064a",
    page_icon="[lab]",
    layout="wide",
)


# ------------------------------------------------------- language picker
with st.sidebar:
    st.markdown("### \U0001F310 " + T["en"]["lang_label"] + " / " + T["ar"]["lang_label"])
    choice = st.radio(
        "Language",
        ["English", "\u0627\u0644\u0639\u0631\u0628\u064a\u0629"],
        index=0 if st.session_state.lang == "en" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="_lang_radio",
    )
    st.session_state.lang = "en" if choice == "English" else "ar"

L = st.session_state.lang
S = T[L]


# ------------------------------------------------------ RTL style (ar)
if L == "ar":
    st.markdown(
        """
        <style>
        html, body, .stApp {
            direction: rtl;
        }
        .stMarkdown, .stCaption, h1, h2, h3, h4, h5, h6 {
            text-align: right;
        }
        .stTextArea textarea, .stTextInput input {
            direction: rtl;
            text-align: right;
        }
        .stButton button {
            direction: rtl;
        }
        .stCode, code {
            direction: ltr;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------ header
st.title(S["title"])
st.caption(S["subtitle"])


# --------------------------------------------------------- settings
with st.sidebar:
    st.divider()
    st.header(S["settings"])
    use_model = st.checkbox(S["stat_detection"], value=False)
    model_id = st.selectbox(
        S["model_label"],
        [
            "openai-community/gpt2",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ],
        index=0,
        disabled=not use_model,
    )
    arabic_alef = st.checkbox(S["normalize_alef"], value=False)
    arabic_ya = st.checkbox(S["normalize_ya"], value=False)
    arabic_tashkeel = st.checkbox(S["strip_tashkeel"], value=False)
    st.divider()
    st.caption(S["tip"])
    st.divider()
    with st.expander(S["about"]):
        st.write(S["about_body"])


# ------------------------------------------------------------- layout
col_left, col_right = st.columns([1, 1])
if L == "ar":
    col_input, col_report = col_right, col_left
else:
    col_input, col_report = col_left, col_right

with col_input:
    st.subheader(S["input_h"])
    default_text = (
        "This is p\u0430rt of a test with \u200b a zero-width space.\n\n"
        "\u0645\u0631\u062d\u0628\u064b\u0627 \u0628\u0643\u0645\u060c "
        "\u0627\u0644\u0623\u0631\u0642\u0627\u0645: "
        "\u0661\u0662\u0663\u0664\u0665\u060c "
        "\u0648\u0627\u0644\u0641\u0627\u0635\u0644\u0629 "
        "\u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u060c"
    )
    text = st.text_area(
        S["input_h"],
        value=default_text,
        height=300,
        label_visibility="collapsed",
    )
    analyze_btn = st.button(S["analyze"], type="primary", use_container_width=True)


if analyze_btn and text.strip():
    with st.spinner(S["analyzing"]):
        report = ai_text_lab.analyze(
            text,
            model_id=(model_id if use_model else None),
            arabic_alef=arabic_alef,
            arabic_ya=arabic_ya,
            arabic_strip_tashkeel=arabic_tashkeel,
        )

    with col_report:
        st.subheader(S["report_h"])
        s = report.surface
        a = report.arabic
        n = report.normalization
        st_d = report.statistical

        m1, m2, m3 = st.columns(3)
        m1.metric(S["surface_findings"], s["total_findings"])
        m2.metric(
            S["arabic_findings"],
            sum(a["counts"].values()) if a["is_arabic"] else 0,
        )
        m3.metric(S["chars_changed"], n["net_delta"])

        st.markdown("### " + S["surface_artifacts"])
        if s["total_findings"] == 0:
            st.success(S["none_detected"])
        else:
            st.json(s["counts"])

        st.markdown("### " + S["arabic_artifacts"])
        if not a["is_arabic"]:
            st.info(S["not_arabic"])
        elif not a["counts"]:
            st.success(S["none_detected"])
        else:
            st.json(a["counts"])
            pct = round(a["tashkeel_density"] * 100, 2)
            st.caption(S["tashkeel_density"] + ": " + str(pct) + "%")

        st.markdown("### " + S["stat_watermark"])
        if st_d.get("status") == "not_attempted":
            st.info(S["not_attempted"])
        elif st_d.get("status") == "error":
            st.error(st_d.get("message"))
        else:
            for key in ("greenlist", "synthid"):
                info = st_d.get(key)
                if info:
                    verdict = S["watermarked"] if info["prediction"] else S["not_watermarked"]
                    zval = round(info["z"], 3)
                    st.metric(key + " z", str(zval), help=verdict)

    st.divider()
    st.subheader(S["cleaned_text"])
    st.code(report.cleaned_text, language=None)
    st.download_button(
        S["download"],
        data=report.cleaned_text,
        file_name="cleaned.txt",
        mime="text/plain",
    )

    st.subheader(S["diff_h"])
    st.code(ai_text_lab.diff(text, mode="chars"), language=None)


st.divider()
st.caption(S["footer"])