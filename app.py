"""Professor Gemini - Hybrid AI Learning Platform.

A Streamlit application that uses Gemini for content generation
and Claude for structural planning, adversarial critique, and synthesis.

Usage:
    streamlit run app.py --server.port 8502
"""

import atexit
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from config.settings import get_settings
from core.pipeline import Pipeline
from core.single_prompt_pipeline import SinglePromptPipeline
from utils.logging_utils import RequestLogger, configure_logging
from utils.file_utils import FileManager


# Configure logging on import
configure_logging()


# ============== CUSTOM CSS ==============

def inject_custom_css(theme: str = "dark") -> None:
    """Inject custom CSS for professional styling."""

    # Color schemes - Google-inspired clean design
    if theme == "light":
        colors = {
            "bg_primary": "#ffffff",
            "bg_secondary": "#f8f9fa",
            "bg_tertiary": "#f1f3f4",
            "bg_input": "#ffffff",
            "text_primary": "#202124",
            "text_secondary": "#5f6368",
            "text_muted": "#9aa0a6",
            "accent": "#b8860b",
            "accent_hover": "#996f0a",
            "accent_light": "#fef7e0",
            "border": "#dadce0",
            "border_light": "#e8eaed",
            "success": "#1e8e3e",
            "success_bg": "#e6f4ea",
            "warning": "#f9ab00",
            "error": "#d93025",
            "console_bg": "#202124",
            "console_text": "#e8eaed",
        }
    else:  # dark
        colors = {
            "bg_primary": "#0d0d0d",
            "bg_secondary": "#1a1a1a",
            "bg_tertiary": "#262626",
            "bg_input": "#1a1a1a",
            "text_primary": "#e8eaed",
            "text_secondary": "#9aa0a6",
            "text_muted": "#5f6368",
            "accent": "#DAA520",
            "accent_hover": "#e6b422",
            "accent_light": "rgba(218, 165, 32, 0.15)",
            "border": "#3c4043",
            "border_light": "#5f6368",
            "success": "#81c995",
            "success_bg": "rgba(129, 201, 149, 0.15)",
            "warning": "#fdd663",
            "error": "#f28b82",
            "console_bg": "#0d0d0d",
            "console_text": "#e8eaed",
        }

    css = f"""
    <style>
        /* ===== IMPORTS ===== */
        @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@400;500&family=Roboto+Mono:wght@400&display=swap');

        /* ===== ROOT VARIABLES ===== */
        :root {{
            --bg-primary: {colors['bg_primary']};
            --bg-secondary: {colors['bg_secondary']};
            --bg-tertiary: {colors['bg_tertiary']};
            --bg-input: {colors['bg_input']};
            --text-primary: {colors['text_primary']};
            --text-secondary: {colors['text_secondary']};
            --text-muted: {colors['text_muted']};
            --accent: {colors['accent']};
            --accent-hover: {colors['accent_hover']};
            --accent-light: {colors['accent_light']};
            --border: {colors['border']};
            --border-light: {colors['border_light']};
            --success: {colors['success']};
            --success-bg: {colors['success_bg']};
            --warning: {colors['warning']};
            --error: {colors['error']};
            --console-bg: {colors['console_bg']};
            --console-text: {colors['console_text']};
            --font-display: 'Google Sans', 'Segoe UI', Roboto, sans-serif;
            --font-body: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'Roboto Mono', 'SF Mono', Consolas, monospace;
        }}

        /* ===== BASE STYLES ===== */
        html, body {{
            font-size: 14px !important;
        }}

        .stApp {{
            background: var(--bg-primary) !important;
            font-family: var(--font-body) !important;
        }}

        [data-testid="stAppViewContainer"] {{
            background: var(--bg-primary) !important;
        }}

        [data-testid="stHeader"] {{
            background: transparent !important;
        }}

        /* ===== SIDEBAR ===== */
        [data-testid="stSidebar"] {{
            background: var(--bg-secondary) !important;
            border-right: 1px solid var(--border) !important;
        }}

        [data-testid="stSidebar"] [data-testid="stMarkdown"] {{
            color: var(--text-secondary) !important;
        }}

        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stTextInput label {{
            color: var(--text-muted) !important;
            font-size: 11px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.8px !important;
            font-weight: 500 !important;
        }}

        /* ===== TYPOGRAPHY ===== */
        h1, h2, h3, h4, h5, h6 {{
            font-family: var(--font-display) !important;
            color: var(--text-primary) !important;
        }}

        p, span, div {{
            color: var(--text-primary);
        }}

        /* ===== PAGE HEADER ===== */
        .page-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 24px 0 20px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }}

        .page-title-section {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .page-icon {{
            font-size: 28px;
        }}

        .page-title {{
            font-family: var(--font-display) !important;
            font-size: 22px !important;
            font-weight: 400 !important;
            color: var(--text-primary) !important;
            margin: 0 !important;
            letter-spacing: 0;
        }}

        .page-subtitle {{
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 2px;
        }}

        .header-actions {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        /* ===== SECTION LABEL ===== */
        .section-label {{
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-muted);
            margin-bottom: 12px;
        }}

        /* ===== INPUT CONTAINER ===== */
        .input-container {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
        }}

        /* ===== ACTION ROW ===== */
        .action-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
        }}

        .action-row-left {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}

        .action-row-right {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        /* ===== STATUS PILL ===== */
        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 16px;
            font-size: 13px;
            font-weight: 500;
        }}

        .status-pill.ready {{
            background: var(--bg-tertiary);
            color: var(--text-secondary);
        }}

        .status-pill.success {{
            background: var(--success-bg);
            color: var(--success);
        }}

        .status-pill-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
        }}

        /* ===== BUTTONS ===== */
        .stButton > button {{
            font-family: var(--font-display) !important;
            font-weight: 500 !important;
            font-size: 14px !important;
            border-radius: 4px !important;
            padding: 8px 24px !important;
            transition: all 0.1s ease !important;
            height: 36px !important;
            min-height: 36px !important;
            min-width: 80px !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }}

        .stButton > button[kind="primary"] {{
            background: var(--accent) !important;
            color: #000 !important;
            border: none !important;
        }}

        .stButton > button[kind="primary"]:hover {{
            background: var(--accent-hover) !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
        }}

        .stButton > button[kind="secondary"] {{
            background: transparent !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
        }}

        .stButton > button[kind="secondary"]:hover {{
            background: var(--bg-tertiary) !important;
        }}

        .stButton > button:disabled {{
            opacity: 0.38 !important;
            cursor: not-allowed !important;
        }}

        /* ===== TEXT AREA ===== */
        .stTextArea {{
            margin-bottom: 0 !important;
        }}

        .stTextArea label {{
            display: none !important;
        }}

        .stTextArea textarea {{
            font-family: var(--font-body) !important;
            font-size: 14px !important;
            background: var(--bg-input) !important;
            border: 1px solid var(--border) !important;
            border-radius: 4px !important;
            color: var(--text-primary) !important;
            padding: 12px 16px !important;
            min-height: 120px !important;
            resize: vertical !important;
            line-height: 1.5 !important;
        }}

        .stTextArea textarea:focus {{
            border-color: var(--accent) !important;
            box-shadow: none !important;
            outline: none !important;
        }}

        .stTextArea textarea::placeholder {{
            color: var(--text-muted) !important;
        }}

        /* ===== TEXT INPUT ===== */
        .stTextInput input {{
            font-family: var(--font-body) !important;
            font-size: 14px !important;
            background: var(--bg-input) !important;
            border: 1px solid var(--border) !important;
            border-radius: 4px !important;
            color: var(--text-primary) !important;
            padding: 8px 12px !important;
        }}

        .stTextInput input:focus {{
            border-color: var(--accent) !important;
            box-shadow: none !important;
            outline: none !important;
        }}

        /* ===== OUTPUT SECTION ===== */
        .output-label {{
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-muted);
            margin-bottom: 12px;
        }}

        /* ===== EXPANDER ===== */
        [data-testid="stExpander"] {{
            background: var(--bg-secondary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            overflow: hidden;
        }}

        [data-testid="stExpander"] > details {{
            background: var(--bg-secondary) !important;
        }}

        [data-testid="stExpander"] > details > summary {{
            background: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            padding: 16px 20px !important;
            font-family: var(--font-display) !important;
            font-weight: 500 !important;
            font-size: 14px !important;
        }}

        [data-testid="stExpander"] > details > summary:hover {{
            background: var(--bg-tertiary) !important;
        }}

        [data-testid="stExpander"] > details > summary span {{
            color: var(--text-primary) !important;
        }}

        [data-testid="stExpander"] > details > div {{
            background: var(--bg-secondary) !important;
            padding: 0 20px 20px 20px !important;
        }}

        /* Text inside expanders */
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] span,
        [data-testid="stExpander"] li,
        [data-testid="stExpander"] td,
        [data-testid="stExpander"] th {{
            color: var(--text-primary) !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
        }}

        /* Headings inside expanders */
        [data-testid="stExpander"] h1 {{
            font-size: 24px !important;
            font-weight: 400 !important;
            margin-top: 24px !important;
            margin-bottom: 12px !important;
        }}

        [data-testid="stExpander"] h2 {{
            font-size: 18px !important;
            font-weight: 500 !important;
            margin-top: 24px !important;
            margin-bottom: 8px !important;
            padding-bottom: 8px !important;
            border-bottom: 1px solid var(--border) !important;
        }}

        [data-testid="stExpander"] h3 {{
            font-size: 16px !important;
            font-weight: 500 !important;
            margin-top: 20px !important;
            margin-bottom: 8px !important;
        }}

        [data-testid="stExpander"] h1,
        [data-testid="stExpander"] h2,
        [data-testid="stExpander"] h3,
        [data-testid="stExpander"] h4,
        [data-testid="stExpander"] h5,
        [data-testid="stExpander"] h6 {{
            color: var(--text-primary) !important;
        }}

        /* Code inside expanders */
        [data-testid="stExpander"] code {{
            background: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            font-family: var(--font-mono) !important;
            font-size: 13px !important;
        }}

        [data-testid="stExpander"] pre {{
            background: var(--console-bg) !important;
            color: var(--console-text) !important;
            padding: 16px !important;
            border-radius: 8px !important;
            overflow-x: auto !important;
            margin: 16px 0 !important;
        }}

        [data-testid="stExpander"] pre code {{
            background: transparent !important;
            color: var(--console-text) !important;
            padding: 0 !important;
        }}

        /* Lists inside expanders */
        [data-testid="stExpander"] ul,
        [data-testid="stExpander"] ol {{
            color: var(--text-primary) !important;
            padding-left: 24px !important;
            margin: 12px 0 !important;
        }}

        [data-testid="stExpander"] li {{
            margin-bottom: 6px !important;
        }}

        /* Tables inside expanders */
        [data-testid="stExpander"] table {{
            border-collapse: collapse !important;
            width: 100% !important;
            margin: 16px 0 !important;
        }}

        [data-testid="stExpander"] table th {{
            background: var(--bg-tertiary) !important;
            border: 1px solid var(--border) !important;
            padding: 10px 14px !important;
            text-align: left !important;
            font-weight: 500 !important;
        }}

        [data-testid="stExpander"] table td {{
            border: 1px solid var(--border) !important;
            padding: 10px 14px !important;
        }}

        /* Blockquotes */
        [data-testid="stExpander"] blockquote {{
            border-left: 4px solid var(--accent) !important;
            padding-left: 16px !important;
            margin: 16px 0 !important;
            color: var(--text-secondary) !important;
        }}

        /* Links */
        [data-testid="stExpander"] a {{
            color: var(--accent) !important;
            text-decoration: none !important;
        }}

        [data-testid="stExpander"] a:hover {{
            text-decoration: underline !important;
        }}

        /* Strong text */
        [data-testid="stExpander"] strong,
        [data-testid="stExpander"] b {{
            color: var(--text-primary) !important;
            font-weight: 500 !important;
        }}

        /* HR */
        [data-testid="stExpander"] hr {{
            border: none !important;
            border-top: 1px solid var(--border) !important;
            margin: 24px 0 !important;
        }}

        /* ===== DIVIDER ===== */
        hr {{
            border: none !important;
            border-top: 1px solid var(--border) !important;
            margin: 16px 0 !important;
        }}

        /* ===== ALERTS ===== */
        [data-testid="stAlert"] {{
            background: var(--bg-secondary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 4px !important;
        }}

        [data-testid="stAlert"] p {{
            color: var(--text-primary) !important;
        }}

        /* ===== SCROLLBAR ===== */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}

        ::-webkit-scrollbar-track {{
            background: transparent;
        }}

        ::-webkit-scrollbar-thumb {{
            background: var(--border);
            border-radius: 4px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: var(--text-muted);
        }}

        /* ===== HIDE STREAMLIT BRANDING ===== */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        [data-testid="stDecoration"] {{ display: none; }}

        /* ===== SIDEBAR COLLAPSE BUTTON ===== */
        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="stSidebarNavCollapseIcon"],
        button[kind="header"] {{
            color: var(--text-secondary) !important;
        }}

        [data-testid="stSidebarCollapsedControl"] svg,
        [data-testid="collapsedControl"] svg {{
            color: var(--text-secondary) !important;
            stroke: var(--text-secondary) !important;
        }}

        /* Sidebar expand/collapse arrow */
        [data-testid="stSidebar"] button[kind="headerNoPadding"],
        [data-testid="stSidebar"] [data-testid="stSidebarNavCollapseIcon"],
        [data-testid="baseButton-headerNoPadding"] {{
            color: var(--text-secondary) !important;
        }}

        [data-testid="baseButton-headerNoPadding"] svg {{
            color: var(--text-secondary) !important;
        }}

        /* ===== THEME TOGGLE ===== */
        [data-testid="stButton"][data-key="theme_toggle"] button {{
            background: var(--bg-secondary) !important;
            border: 1px solid var(--border) !important;
            padding: 0 !important;
            font-size: 16px !important;
            width: 36px !important;
            min-width: 36px !important;
            height: 36px !important;
            border-radius: 4px !important;
        }}

        [data-testid="stButton"][data-key="theme_toggle"] button:hover {{
            background: var(--bg-tertiary) !important;
        }}

        /* ===== SAVE BUTTON (TOP) ===== */
        [data-testid="stButton"][data-key="save_top"] button {{
            height: 36px !important;
            min-height: 36px !important;
        }}

        /* ===== STATUS INDICATORS ===== */
        .status-indicator {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: var(--text-secondary);
        }}

        .status-indicator .dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}

        .status-indicator .dot.connected {{ background: var(--success); }}
        .status-indicator .dot.disconnected {{ background: var(--error); }}

        /* ===== TOAST ===== */
        [data-testid="stToast"] {{
            background: var(--bg-secondary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 4px !important;
        }}

        /* ===== SPINNER ===== */
        .stSpinner > div {{
            border-top-color: var(--accent) !important;
        }}

        /* ===== MARKDOWN ===== */
        .stMarkdown, .stMarkdown p, .stMarkdown span,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p {{
            color: var(--text-primary) !important;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] code {{
            color: var(--text-secondary) !important;
        }}

        [data-testid="stSidebar"] code {{
            background: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            font-size: 12px !important;
        }}

        [data-testid="stSidebar"] .stTextInput input {{
            background: var(--bg-input) !important;
            border-color: var(--border) !important;
            color: var(--text-primary) !important;
        }}

        /* Labels */
        .stTextInput label, .stTextArea label, .stSelectbox label {{
            color: var(--text-muted) !important;
        }}

        /* ===== EMPTY STATE ===== */
        .empty-state {{
            text-align: center;
            padding: 48px 24px;
        }}

        .empty-state-icon {{
            font-size: 48px;
            margin-bottom: 16px;
            opacity: 0.4;
        }}

        .empty-state-text {{
            font-size: 14px;
            color: var(--text-muted);
        }}

        /* ===== FOOTER ===== */
        .app-footer {{
            text-align: center;
            padding: 32px 16px;
            color: var(--text-muted);
            font-size: 12px;
        }}

        /* ===== COLUMN SPACING ===== */
        [data-testid="stHorizontalBlock"] {{
            gap: 8px !important;
            flex-wrap: nowrap !important;
        }}

        [data-testid="column"] {{
            padding: 0 !important;
            min-width: 0 !important;
        }}

        /* Ensure action row buttons don't shrink too small */
        [data-testid="stHorizontalBlock"] [data-testid="column"]:has([data-testid="stButton"]) {{
            min-width: 80px !important;
            flex-shrink: 0 !important;
        }}

        /* ===== SELECT BOX ===== */
        .stSelectbox [data-baseweb="select"] {{
            background: var(--bg-input) !important;
        }}

        .stSelectbox [data-baseweb="select"] > div {{
            background: var(--bg-input) !important;
            border-color: var(--border) !important;
            border-radius: 4px !important;
        }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


# ============== SESSION STATE ==============

def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    defaults = {
        "pipeline_result": None,
        "single_prompt_result": None,
        "generation_mode": "deep_dive",  # or "single_prompt"
        "cyrus_root": get_settings().cyrus_root_path,
        "timer_elapsed": None,
        "topic_input": "",
        "theme": "dark",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============== COMPONENTS ==============

def render_sidebar() -> None:
    """Render the sidebar with configuration options."""
    settings = get_settings()

    st.sidebar.markdown("### Status")

    gemini_ok = settings.is_gemini_configured()
    claude_ok = settings.is_claude_configured()

    st.sidebar.markdown(
        f"""
        <div class="status-indicator">
            <span class="dot {'connected' if gemini_ok else 'disconnected'}"></span>
            <span>Gemini {'Connected' if gemini_ok else 'Not Configured'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        f"""
        <div class="status-indicator">
            <span class="dot {'connected' if claude_ok else 'disconnected'}"></span>
            <span>Claude {'Connected' if claude_ok else 'Not Configured'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown("### Configuration")

    cyrus_root = st.sidebar.text_input(
        "Cyrus Root Path",
        value=st.session_state.cyrus_root,
        help="Path to Cyrus project for Nebula integration",
    )
    st.session_state.cyrus_root = cyrus_root

    file_manager = FileManager(cyrus_root)
    cyrus_available, _ = file_manager.is_cyrus_available()

    if cyrus_available:
        st.sidebar.markdown(
            '<div class="status-indicator"><span class="dot connected"></span><span>Cyrus accessible</span></div>',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")

    st.sidebar.markdown("### Settings")
    st.sidebar.markdown(f"Critique: `{'ON' if settings.enable_critique else 'OFF'}`")
    st.sidebar.markdown(f"Local Synthesis: `{'ON' if settings.local_synthesis else 'OFF'}`")

    st.sidebar.markdown("---")

    st.sidebar.markdown("### Model")
    st.sidebar.code(settings.gemini_model, language=None)


def render_output_section() -> None:
    """Render the output section with Master Guide or Single Prompt result."""
    mode = st.session_state.generation_mode

    if mode == "deep_dive":
        result = st.session_state.pipeline_result
        if not result:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-state-icon">📚</div>
                    <div class="empty-state-text">Enter a topic above to generate a comprehensive guide</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        if not result.success:
            st.error(f"Pipeline failed: {result.error}")
            return

        if result.low_confidence_sections > 0:
            st.warning(
                f"**Review Recommended:** {result.low_confidence_sections} section(s) "
                "did not pass the Bar Raiser review after maximum attempts."
            )

        st.markdown('<div class="output-label">Generated Guide</div>', unsafe_allow_html=True)

        with st.expander("View Full Guide", expanded=True):
            st.markdown(result.master_guide)

    else:  # single_prompt mode
        result = st.session_state.single_prompt_result
        if not result:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-state-icon">⚡</div>
                    <div class="empty-state-text">Enter a prompt above to generate content with Knowledge Base context</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        if not result.success:
            st.error(f"Generation failed: {result.error}")
            return

        st.markdown(
            f'<div class="output-label">Generated Content ({result.context_file_count} KB docs used)</div>',
            unsafe_allow_html=True,
        )

        with st.expander("View Output", expanded=True):
            st.markdown(result.output)


def run_pipeline(topic: str) -> None:
    """Execute the pipeline for a given topic."""
    start_time = datetime.now()

    pipeline = Pipeline()
    result = pipeline.execute(topic)

    elapsed = (datetime.now() - start_time).total_seconds()

    st.session_state.timer_elapsed = elapsed
    st.session_state.pipeline_result = result


def run_single_prompt(prompt: str) -> None:
    """Execute single prompt with Knowledge Base context."""
    start_time = datetime.now()

    pipeline = SinglePromptPipeline()
    result = pipeline.execute(prompt)

    elapsed = (datetime.now() - start_time).total_seconds()

    st.session_state.timer_elapsed = elapsed
    st.session_state.single_prompt_result = result


# ============== MAIN ==============

def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="Professor Gemini",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    init_session_state()

    theme = st.session_state.theme
    if theme == "system":
        theme = "dark"
    inject_custom_css(theme)

    render_sidebar()

    # ===== PAGE HEADER =====
    header_left, header_right = st.columns([5, 1])

    with header_left:
        st.markdown(
            """
            <div class="page-title-section">
                <span class="page-icon">🎓</span>
                <div>
                    <div class="page-title">Professor Gemini</div>
                    <div class="page-subtitle">Deep learning with hybrid AI</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with header_right:
        # Save button and theme toggle in top right
        col_save, col_theme = st.columns([2, 1])

        with col_save:
            # Determine which result to use based on mode
            if st.session_state.generation_mode == "deep_dive":
                result = st.session_state.pipeline_result
                save_disabled = not (result and result.success)
                content_to_save = result.master_guide if result else None
                low_conf = result.low_confidence_sections if result else 0
                mode_str = "deep_dive"
            else:
                result = st.session_state.single_prompt_result
                save_disabled = not (result and result.success)
                content_to_save = result.output if result else None
                low_conf = 0  # Single prompt has no confidence tracking
                mode_str = "single_prompt"

            if st.button("Save", disabled=save_disabled, use_container_width=True, key="save_top"):
                if result and result.success and content_to_save:
                    file_manager = FileManager(st.session_state.cyrus_root)
                    success, filepath, message = file_manager.save_guide(
                        content=content_to_save,
                        low_confidence_count=low_conf,
                        mode=mode_str,
                    )
                    if success:
                        st.toast(f"Saved: {filepath.split('/')[-1]}", icon="✅")
                        if mode_str == "deep_dive":
                            RequestLogger().log_session(result)
                    else:
                        st.toast(f"Failed: {message}", icon="🚫")

        with col_theme:
            current_theme = st.session_state.theme
            theme_icon = "🌙" if current_theme == "dark" else "☀️" if current_theme == "light" else "💻"
            next_theme = {"dark": "light", "light": "system", "system": "dark"}

            if st.button(theme_icon, key="theme_toggle", help=f"Theme: {current_theme.title()}"):
                st.session_state.theme = next_theme[current_theme]
                st.rerun()

    st.markdown("---")

    # ===== API CHECK =====
    settings = get_settings()
    if not settings.is_fully_configured():
        st.error(
            "**API Keys Required**\n\n"
            "Configure your API keys in `.env`:\n"
            "- `GEMINI_API_KEY`\n"
            "- `ANTHROPIC_API_KEY` (if USE_CLAUDE=true)"
        )
        st.stop()

    # ===== MODE SELECTION =====
    st.markdown('<div class="section-label">Mode</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Generation Mode",
        options=["deep_dive", "single_prompt"],
        format_func=lambda x: "Deep Dive (4-step)" if x == "deep_dive" else "Single Prompt (with KB context)",
        index=0 if st.session_state.generation_mode == "deep_dive" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="mode_radio",
    )
    if mode != st.session_state.generation_mode:
        st.session_state.generation_mode = mode
        st.rerun()

    # ===== INPUT SECTION =====
    if st.session_state.generation_mode == "deep_dive":
        st.markdown('<div class="section-label">Topic</div>', unsafe_allow_html=True)
        placeholder = "Enter a topic to explore...\n\nExamples:\n• Distributed consensus algorithms\n• Kubernetes architecture\n• Real-time data streaming"
    else:
        st.markdown('<div class="section-label">Prompt</div>', unsafe_allow_html=True)
        placeholder = "Enter your prompt...\n\nExamples:\n• Generate Principal TPM lexicon terms for system design\n• Create a comparison table of consensus protocols\n• Summarize key SRE metrics across all documents"

    topic = st.text_area(
        "Topic/Prompt",
        height=120,
        placeholder=placeholder,
        key="topic_input",
        label_visibility="collapsed",
    )

    # ===== ACTION ROW =====
    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([4, 2, 1, 1])

    with col1:
        # Status indicator
        if st.session_state.timer_elapsed is not None:
            elapsed = st.session_state.timer_elapsed
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            timer_text = f"{minutes}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.1f}s"
            st.markdown(
                f'<div class="status-pill success"><span class="status-pill-dot"></span>Completed in {timer_text}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-pill ready"><span class="status-pill-dot"></span>Ready</div>',
                unsafe_allow_html=True,
            )

    with col3:
        start_clicked = st.button(
            "Generate",
            type="primary",
            disabled=not topic.strip(),
            use_container_width=True,
        )

    with col4:
        if st.button("Clear", use_container_width=True):
            st.session_state.pipeline_result = None
            st.session_state.single_prompt_result = None
            st.session_state.timer_elapsed = None
            st.session_state.topic_input = ""
            st.rerun()

    # ===== EXECUTION =====
    if start_clicked and topic.strip():
        # Clear previous results before generating new one
        st.session_state.pipeline_result = None
        st.session_state.single_prompt_result = None
        st.session_state.timer_elapsed = None

        if st.session_state.generation_mode == "deep_dive":
            with st.spinner("Generating guide..."):
                run_pipeline(topic.strip())
        else:
            with st.spinner("Generating with KB context..."):
                run_single_prompt(topic.strip())
        st.rerun()

    st.markdown("---")

    # ===== OUTPUT =====
    render_output_section()

    # ===== FOOTER =====
    st.markdown(
        '<div class="app-footer">Professor Gemini</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
