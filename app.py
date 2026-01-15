"""Professor Gemini - Hybrid AI Learning Platform.

A Streamlit application that uses Gemini for content generation
and Claude for structural planning, adversarial critique, and synthesis.

Usage:
    streamlit run app.py --server.port 8502
"""

import streamlit as st
from datetime import datetime

from config.settings import get_settings, CriticStrictness
from core.pipeline import Pipeline
from utils.logging_utils import RequestLogger, configure_logging
from utils.file_utils import FileManager


# Configure logging on import
configure_logging()


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    defaults = {
        "pipeline_result": None,
        "status_messages": [],
        "is_running": False,
        "cyrus_root": get_settings().cyrus_root_path,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_status_message(message: str) -> None:
    """Add a status message to the session state.

    Args:
        message: Status message to add.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.status_messages.append(f"[{timestamp}] {message}")


def render_sidebar() -> None:
    """Render the sidebar with configuration options."""
    settings = get_settings()

    st.sidebar.title("Professor Gemini")
    st.sidebar.markdown("*Hybrid AI Learning Platform*")

    st.sidebar.divider()

    # API Status
    st.sidebar.subheader("API Status")

    gemini_status = "Connected" if settings.is_gemini_configured() else "Not Configured"
    gemini_color = "green" if settings.is_gemini_configured() else "red"
    st.sidebar.markdown(f":{gemini_color}[●] **Gemini:** {gemini_status}")

    claude_status = "Connected" if settings.is_claude_configured() else "Not Configured"
    claude_color = "green" if settings.is_claude_configured() else "red"
    st.sidebar.markdown(f":{claude_color}[●] **Claude:** {claude_status}")

    st.sidebar.divider()

    # Configuration
    st.sidebar.subheader("Configuration")

    cyrus_root = st.sidebar.text_input(
        "Cyrus Root Path",
        value=st.session_state.cyrus_root,
        help="Path to Cyrus project for Nebula integration",
    )
    st.session_state.cyrus_root = cyrus_root

    # File manager check
    file_manager = FileManager(cyrus_root)
    cyrus_available, cyrus_msg = file_manager.is_cyrus_available()
    if cyrus_available:
        st.sidebar.markdown(":green[✓] Cyrus project accessible")
    else:
        st.sidebar.markdown(f":orange[⚠] {cyrus_msg}")

    st.sidebar.divider()

    # Critic Strictness Info
    st.sidebar.subheader("Critic Strictness")
    st.sidebar.markdown("""
    - **Attempts 1-2:** HIGH strictness
    - **Attempt 3:** MEDIUM strictness (relaxed)
    - After 3 attempts: Accept with low confidence flag
    """)

    st.sidebar.divider()

    # Model Info
    st.sidebar.subheader("Models")
    st.sidebar.markdown(f"**Gemini:** `{settings.gemini_model}`")
    st.sidebar.markdown(f"**Claude:** `{settings.claude_model}`")

    st.sidebar.divider()

    # Settings Info
    st.sidebar.subheader("Settings")
    st.sidebar.markdown(f"Max Workers: `{settings.max_workers}`")
    st.sidebar.markdown(f"Max Retries: `{settings.max_retries}`")
    st.sidebar.markdown(f"API Timeout: `{settings.api_timeout}s`")


def render_status_console() -> None:
    """Render the status console with real-time updates."""
    st.subheader("Status Console")

    # Create scrollable container for status messages
    status_container = st.container(height=300)

    with status_container:
        if st.session_state.status_messages:
            for msg in st.session_state.status_messages:
                # Color code based on content
                if "APPROVED" in msg or "complete" in msg.lower():
                    st.markdown(f":green[{msg}]")
                elif "REJECTED" in msg or "FAIL" in msg:
                    st.markdown(f":orange[{msg}]")
                elif "failed" in msg.lower() or "error" in msg.lower():
                    st.markdown(f":red[{msg}]")
                else:
                    st.markdown(f"`{msg}`")
        else:
            st.markdown("*Waiting for pipeline execution...*")


def render_output_section() -> None:
    """Render the output section with Master Guide."""
    result = st.session_state.pipeline_result

    if not result:
        st.info("Run the pipeline to generate a Master Guide")
        return

    if not result.success:
        st.error(f"Pipeline failed: {result.error}")
        return

    # Success header with metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sections", result.total_sections)
    with col2:
        st.metric("Low Confidence", result.low_confidence_sections)
    with col3:
        st.metric("Duration", f"{result.total_duration_ms / 1000:.1f}s")
    with col4:
        st.metric("Session", result.session_id[:8])

    st.divider()

    # Low confidence warning
    if result.low_confidence_sections > 0:
        st.warning(
            f"⚠️ **Review Recommended:** {result.low_confidence_sections} section(s) "
            "did not pass the Bar Raiser review after 3 attempts. These sections are "
            "marked with an orange indicator in the guide below."
        )

    # Master Guide content
    st.subheader("Master Guide")

    # Add low confidence markers to content
    display_content = result.master_guide

    # Display in expandable markdown
    with st.expander("View Full Guide", expanded=True):
        st.markdown(display_content)

    st.divider()

    # Add to Nebula button
    st.subheader("Save to Cyrus")

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("📁 Add to Nebula", type="primary", use_container_width=True):
            file_manager = FileManager(st.session_state.cyrus_root)
            success, filepath, message = file_manager.save_guide(
                content=result.master_guide,
                low_confidence_count=result.low_confidence_sections,
            )

            if success:
                st.success(f"✓ {message}")
                st.code(filepath, language=None)

                # Log the session
                logger = RequestLogger()
                logger.log_session(result)
            else:
                st.error(f"Failed: {message}")

    with col2:
        st.markdown(
            f"Saves to: `{st.session_state.cyrus_root}/gemini-responses/`"
        )


def render_deep_dive_details() -> None:
    """Render detailed deep dive results in an expander."""
    result = st.session_state.pipeline_result

    if not result or not result.deep_dive_results:
        return

    with st.expander("Deep Dive Details", expanded=False):
        for i, dive in enumerate(result.deep_dive_results, 1):
            status_icon = "🟢" if not dive.low_confidence else "🟠"

            st.markdown(f"### {status_icon} Topic {i}: {dive.topic[:60]}...")

            if dive.attempts:
                cols = st.columns(len(dive.attempts))
                for j, (col, attempt) in enumerate(zip(cols, dive.attempts)):
                    with col:
                        result_icon = "✓" if attempt.critique_passed else "✗"
                        st.markdown(f"**Attempt {attempt.attempt_number}** {result_icon}")
                        st.markdown(f"Strictness: `{attempt.strictness.value}`")
                        st.markdown(f"Draft: `{attempt.draft_duration_ms}ms`")
                        st.markdown(f"Critique: `{attempt.critique_duration_ms}ms`")

                        if not attempt.critique_passed:
                            with st.popover("View Feedback"):
                                st.markdown(attempt.critique_feedback)

            st.divider()


def run_pipeline(topic: str) -> None:
    """Execute the pipeline for a given topic.

    Args:
        topic: Topic to process.
    """
    st.session_state.is_running = True
    st.session_state.status_messages = []
    st.session_state.pipeline_result = None

    pipeline = Pipeline(status_callback=add_status_message)
    result = pipeline.execute(topic)

    st.session_state.pipeline_result = result
    st.session_state.is_running = False


def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="Professor Gemini",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    render_sidebar()

    # Main content area
    st.title("🎓 Professor Gemini")
    st.markdown("*Deep learning with hybrid AI: Gemini generates, Claude critiques*")

    st.divider()

    # Check API configuration
    settings = get_settings()
    if not settings.is_fully_configured():
        st.error(
            "⚠️ **API Keys Required**\n\n"
            "Please configure your API keys in the `.env` file:\n"
            "- `GEMINI_API_KEY`: Your Google Gemini API key\n"
            "- `ANTHROPIC_API_KEY`: Your Anthropic Claude API key"
        )
        st.stop()

    # Two-column layout
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Input")

        topic = st.text_area(
            "Enter topic for deep dive",
            height=200,
            placeholder=(
                "Enter a topic you want to learn about...\n\n"
                "Examples:\n"
                "- Distributed consensus algorithms\n"
                "- Kubernetes architecture and orchestration\n"
                "- Real-time data streaming pipelines\n"
                "- API gateway design patterns"
            ),
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            start_button = st.button(
                "🚀 Start Deep Dive",
                type="primary",
                disabled=st.session_state.is_running or not topic.strip(),
                use_container_width=True,
            )

        with col2:
            if st.button(
                "🗑️ Clear",
                disabled=st.session_state.is_running,
                use_container_width=True,
            ):
                st.session_state.status_messages = []
                st.session_state.pipeline_result = None
                st.rerun()

        if start_button and topic.strip():
            with st.spinner("Pipeline running..."):
                run_pipeline(topic.strip())
            st.rerun()

    with right_col:
        render_status_console()

    st.divider()

    # Output section
    render_output_section()

    # Deep dive details
    render_deep_dive_details()

    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Professor Gemini | Hybrid AI Learning Platform | "
        f"Port {settings.app_port}"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
