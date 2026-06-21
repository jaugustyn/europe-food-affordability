"""Streamlit entrypoint for the Europe Food Affordability dashboard."""
from __future__ import annotations

import streamlit as st

from src.dashboard.layout import render_page_intro
from src.dashboard.sections.exploration import render_exploration
from src.dashboard.sections.methodology import render_methodology
from src.dashboard.sections.output import render_output
from src.dashboard.sections.overview import render_overview
from src.dashboard.state import build_dashboard_context


def main() -> None:
    st.set_page_config(
        page_title="Dostępność Żywności w Europie",
        page_icon="🍞",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    context = build_dashboard_context()
    render_page_intro()
    if context.df_filtered.empty or context.latest.empty:
        st.warning("Brak obserwacji dla wybranych filtrów.")
        st.stop()
        return
    render_overview(context)
    render_exploration(context)
    render_methodology(context)
    render_output(context)


if __name__ == "__main__":
    main()
