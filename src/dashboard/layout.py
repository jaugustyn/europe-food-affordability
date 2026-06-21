"""Shared page-level Streamlit layout."""
from __future__ import annotations

import streamlit as st

from src.dashboard.support import APP_TITLE


def render_page_intro() -> None:
    st.title(APP_TITLE)
    st.caption(
        "Dashboard do porównywania presji cen żywności, inflacji ogólnej, dochodów gospodarstw "
        "domowych i dostępności żywności w krajach Europy na podstawie danych Eurostatu."
    )
    st.info(
        "Podstawowa interpretacja: **dodatnia luka dostępności żywności** oznacza, że ceny żywności "
        "rosną szybciej niż dochody, co wskazuje na pogorszenie dostępności. W przypadku dochodu "
        "i wzrostu dochodu wyższe wartości oznaczają większy bufor finansowy gospodarstw domowych."
    )

    st.markdown(
        """
    <style>
        html {scroll-behavior: smooth;}
        .block-container {padding-top: 1.8rem; padding-bottom: 3rem;}
        h2 {margin-top: 2rem;}
        .section-help {
            position: relative;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.15rem;
            height: 1.15rem;
            color: #64748b;
            cursor: help;
            font-size: 0.9rem;
            vertical-align: middle;
            border-radius: 999px;
        }
        .section-help:hover,
        .section-help:focus {
            color: #0f172a;
            outline: none;
            background: #e2e8f0;
        }
        .section-tooltip {
            visibility: hidden;
            opacity: 0;
            position: absolute;
            left: 1.45rem;
            top: 50%;
            transform: translateY(-50%);
            z-index: 20;
            width: min(440px, 70vw);
            padding: 0.65rem 0.75rem;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.16);
            color: #0f172a;
            font-size: 0.86rem;
            font-weight: 400;
            line-height: 1.35;
            text-align: left;
            white-space: normal;
            pointer-events: none;
            transition: opacity 120ms ease;
        }
        .section-help:hover .section-tooltip,
        .section-help:focus .section-tooltip {
            visibility: visible;
            opacity: 1;
        }
        div[data-testid="stMetricValue"] {font-size: 1.75rem;}
    </style>
    """,
        unsafe_allow_html=True,
    )
