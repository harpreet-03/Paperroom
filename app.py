"""Paperroom: a focused Streamlit workspace for the arXiv Digest & QA agent."""
from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.nodes import qa as qa_node
from src.pipeline import load_session, run_pipeline, save_session

st.set_page_config(page_title="Paperroom", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "state" not in st.session_state:
    st.session_state.state = None
if "qa_pairs" not in st.session_state:
    st.session_state.qa_pairs = []
if "workspace_view" not in st.session_state:
    st.session_state.workspace_view = "Overview"


def palette() -> dict[str, str]:
    if st.session_state.dark_mode:
        return {"bg": "#101713", "surface": "#18221b", "side": "#142018", "ink": "#edf5ef", "muted": "#a7baad", "line": "#34483a", "field": "#0f1712", "note": "#183e2c", "note_ink": "#d7f5e1"}
    return {"bg": "#f8fbf8", "surface": "#ffffff", "side": "#eef5ef", "ink": "#16231b", "muted": "#64746a", "line": "#d9e5dc", "field": "#ffffff", "note": "#e4f5e9", "note_ink": "#174b34"}


def inject_style(colors: dict[str, str]) -> None:
    css = """
    <style>
      :root { --bg:__BG__; --surface:__SURFACE__; --side:__SIDE__; --ink:__INK__; --muted:__MUTED__; --line:__LINE__; --field:__FIELD__; --note:__NOTE__; --note-ink:__NOTEINK__; --green:#19724f; }
      .stApp, [data-testid="stAppViewContainer"] { background:var(--bg)!important; color:var(--ink)!important; }
      [data-testid="stHeader"] { background:var(--bg)!important; border-bottom:1px solid var(--line); }
      [data-testid="stSidebar"] { background:var(--side)!important; border-right:1px solid var(--line); }
      [data-testid="stSidebar"] > div:first-child { padding-top:4.2rem; }
      .block-container { max-width:1200px; padding-top:5.2rem; padding-bottom:4rem; }
      h1,h2,h3,p,label,span,li { color:var(--ink)!important; }
      h1 { font-size:2.45rem!important; letter-spacing:-.045em; margin-bottom:.25rem!important; }
      h2 { letter-spacing:-.03em; } h3 { font-size:1.05rem!important; }
      .eyebrow { color:var(--green)!important; font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
      .muted, .stCaption, [data-testid="stCaptionContainer"] p { color:var(--muted)!important; }
      .hero { border-bottom:1px solid var(--line); padding-bottom:1.45rem; margin-bottom:1.45rem; }
      .paper-meta { color:var(--muted)!important; margin-top:-.5rem; margin-bottom:1.25rem; }
      .metric { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:1rem; min-height:84px; }
      .metric .label { color:var(--muted)!important; font-size:.7rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; }
      .metric .value { color:var(--ink)!important; font-weight:700; margin-top:.35rem; overflow-wrap:anywhere; }
      .note { background:var(--note); color:var(--note-ink)!important; border-left:3px solid var(--green); border-radius:0 10px 10px 0; padding:.8rem 1rem; }
      .note * { color:var(--note-ink)!important; }
      [data-testid="stTextInput"] input, [data-testid="stChatInput"] textarea { background:var(--field)!important; color:var(--ink)!important; border:1px solid var(--line)!important; border-radius:9px!important; }
      [data-testid="stTextInput"] input::placeholder, [data-testid="stChatInput"] textarea::placeholder { color:var(--muted)!important; opacity:1; }
      .stButton > button, .stLinkButton > a { background:var(--surface)!important; color:var(--ink)!important; border:1px solid var(--line)!important; border-radius:9px!important; font-weight:650!important; min-height:2.45rem; }
      .stButton > button *, .stLinkButton > a * { color:inherit!important; }
      .stButton > button[kind="primary"] { background:var(--green)!important; border-color:var(--green)!important; color:#fff!important; }
      .stButton > button[kind="primary"]:hover { background:#0d593a!important; color:#fff!important; }
      .stButton > button:disabled { background:#dfe9e1!important; color:#5e6e63!important; opacity:1!important; }
      [data-testid="stChatMessage"] { background:var(--surface)!important; border:1px solid var(--line); border-radius:12px; }
      [data-testid="stRadio"] { margin-bottom:.65rem; }
      [data-testid="stRadio"] label { background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:.35rem .75rem; }
      [data-testid="stRadio"] label:has(input:checked) { background:var(--note); border-color:var(--green); }
      [data-testid="stDivider"] { border-color:var(--line); }

      /* --- bullet cards (Overview tab) --- */
      .bullet { display:flex; gap:.65rem; align-items:flex-start; background:var(--surface);
        border:1px solid var(--line); border-radius:10px; padding:.7rem .9rem; margin-bottom:.5rem;
        font-size:.93rem; line-height:1.5; transition:border-color .15s ease, transform .1s ease; }
      .bullet:hover { transform:translateX(2px); }
      .bullet-icon { flex-shrink:0; font-size:1rem; line-height:1.4; }
      .bullet-method { border-left:3px solid #3b82f6; }
      .bullet-results { border-left:3px solid var(--green); }
      .bullet-limitations { border-left:3px solid #d97706; }

      /* --- category chips under the paper title --- */
      .chip-row { display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0 1rem 0; }
      .chip { display:inline-block; font-size:.72rem; font-weight:700; letter-spacing:.03em;
        padding:.22rem .6rem; border-radius:999px; background:var(--note); color:var(--note-ink)!important;
        border:1px solid var(--line); }

      /* --- chat empty state --- */
      .chat-empty { text-align:center; padding:2.4rem 1rem; border:1px dashed var(--line);
        border-radius:14px; background:var(--surface); color:var(--muted)!important; margin:1rem 0 1.4rem 0; }
      .chat-empty .big { font-size:2rem; margin-bottom:.35rem; }

      /* --- suggestion buttons: make them read as pills, not full-width blocks --- */
      div[data-testid="column"] .stButton > button { font-size:.86rem!important; padding:.4rem .8rem!important;
        border-radius:999px!important; min-height:auto!important; }
    </style>
    """
    for key, value in {"BG": colors["bg"], "SURFACE": colors["surface"], "SIDE": colors["side"], "INK": colors["ink"], "MUTED": colors["muted"], "LINE": colors["line"], "FIELD": colors["field"], "NOTE": colors["note"], "NOTEINK": colors["note_ink"]}.items():
        css = css.replace(f"__{key}__", value)
    st.markdown(css, unsafe_allow_html=True)


def reset_workspace() -> None:
    st.session_state.state = None
    st.session_state.qa_pairs = []
    st.session_state.workspace_view = "Overview"
    st.session_state.pop("pending_question", None)


def recent_sessions() -> list[dict[str, str]]:
    sessions_dir = Path(__file__).parent / "sessions"
    records: list[dict[str, str]] = []
    for path in sessions_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            briefing = data.get("briefing", {})
            records.append({"id": str(data.get("collection_name", path.stem)), "title": str(briefing.get("title", path.stem)), "date": str(briefing.get("published", ""))[:10], "modified": str(path.stat().st_mtime)})
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(records, key=lambda item: item["modified"], reverse=True)[:6]


def show_list(items: list[str], empty: str = "Not available.") -> None:
    if not items:
        st.caption(empty)
        return
    for item in items:
        st.markdown(f"- {item}")


_BULLET_ICONS = {"method": "⚙️", "results": "📈", "limitations": "⚠️"}


def render_bullets(items: list[str], tone: str, empty: str = "Not available.") -> None:
    """Render a list as accent-colored cards instead of plain markdown bullets."""
    if not items:
        st.caption(empty)
        return
    icon = _BULLET_ICONS.get(tone, "•")
    html = "".join(
        f"<div class='bullet bullet-{tone}'>"
        f"<span class='bullet-icon'>{icon}</span><span>{escape(item)}</span></div>"
        for item in items
    )
    st.markdown(html, unsafe_allow_html=True)


def useful_suggestions(items: list[str]) -> list[str]:
    """Hide generic suggestions from older saved sessions or weak LLM output."""
    generic = ("what is this paper all about", "what is the paper about", "summarise the paper", "summarize the paper", "give me a summary")
    kept: list[str] = []
    for item in items:
        normalized = item.strip().lower().rstrip("?.!")
        if len(normalized.split()) >= 5 and normalized not in generic and normalized not in kept:
            kept.append(item.strip())
    return kept[:5]


def ask(state, question: str) -> None:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching the paper for evidence..."):
            response = qa_node.answer(state, question)
        st.write(response)
    st.session_state.qa_pairs.append((question, response))
    save_session(state)


inject_style(palette())

with st.sidebar:
    st.markdown("<div class='eyebrow'>Paperroom</div>", unsafe_allow_html=True)
    st.markdown("### Research workspace")
    st.caption("Brief, inspect, and chat with one paper at a time.")
    st.toggle("Dark mode", key="dark_mode")
    st.divider()
    st.markdown("<div class='eyebrow'>Model status</div>", unsafe_allow_html=True)
    provider = os.environ.get("LLM_PROVIDER", "not configured")
    key_name = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "ollama": "OLLAMA_HOST"}.get(provider)
    st.write(f"**{provider.title()}**")
    st.caption("Ready" if key_name and os.environ.get(key_name) else "Configure .env to run the agent")
    st.divider()
    st.markdown("<div class='eyebrow'>Recent papers</div>", unsafe_allow_html=True)
    history = recent_sessions()
    if history:
        chosen = st.selectbox("Open recent paper", history, format_func=lambda item: f"{item['title'][:46]} ({item['date']})", label_visibility="collapsed")
        if st.button("Open selected paper", use_container_width=True):
            loaded = load_session(chosen["id"])
            if loaded:
                st.session_state.state = loaded
                st.session_state.qa_pairs = list(loaded.qa_history)
                st.session_state.workspace_view = "Overview"
                st.rerun()
    else:
        st.caption("No saved papers yet.")
    st.divider()
    if st.button("＋ New paper", use_container_width=True):
        reset_workspace()
        st.rerun()

st.markdown("<div class='hero'><div class='eyebrow'>arXiv paper intelligence</div><h1>Understand the paper.<br>Then ask better questions.</h1><p class='muted'>Search arXiv, get a grounded executive briefing, then explore the paper through retrieval-augmented chat.</p></div>", unsafe_allow_html=True)
search, go = st.columns([5, 1])
with search:
    query = st.text_input("Paper query", placeholder="Topic, arXiv ID, or arXiv URL", label_visibility="collapsed")
with go:
    run = st.button("Research", type="primary", use_container_width=True)

if run:
    if not query.strip():
        st.warning("Enter a research topic, arXiv ID, or arXiv URL.")
    else:
        reset_workspace()
        with st.status("Preparing the paper...", expanded=True) as status:
            for step in ("Searching arXiv", "Reading the PDF", "Indexing paper sections", "Writing the briefing"):
                st.write(f"• {step}")
            state = run_pipeline(query.strip())
            status.update(label="Briefing ready" if state.status == "done" else "Could not prepare this paper", state="complete" if state.status == "done" else "error")
        st.session_state.state = state
        st.session_state.qa_pairs = list(state.qa_history)
        if state.status == "done" and state.briefing:
            save_session(state)
        elif state.status == "no_candidates":
            st.warning("No papers matched. Try a broader topic or check the arXiv ID.")
        else:
            st.error("The agent could not create a briefing.")
            for error in state.errors:
                st.caption(error)

state = st.session_state.state
if state is None:
    cols = st.columns(3)
    for col, title, text in zip(cols, ("1. Find", "2. Brief", "3. Chat"), ("Search by topic or exact arXiv ID.", "Review the problem, method, claims, and limitations.", "Ask questions grounded in retrieved paper text.")):
        with col:
            with st.container(border=True):
                st.markdown(f"<div class='eyebrow'>{title}</div>", unsafe_allow_html=True)
                st.write(text)
elif state.briefing:
    b = state.briefing
    st.markdown("<div class='eyebrow'>Current paper</div>", unsafe_allow_html=True)
    st.header(b.title)
    st.markdown(f"<div class='paper-meta'>{escape(', '.join(b.authors))} · arXiv:{escape(b.arxiv_id)} · {escape(b.published[:10])}</div>", unsafe_allow_html=True)
    metric_cols = st.columns([1, 1, 1, 1.25])
    metrics = [("Paper ID", b.arxiv_id), ("Published", b.published[:10]), ("Indexed sections", str(len(state.parsed.sections)) if state.parsed else "Saved index")]
    for col, (label, value) in zip(metric_cols[:3], metrics):
        with col:
            st.markdown(f"<div class='metric'><div class='label'>{escape(label)}</div><div class='value'>{escape(value)}</div></div>", unsafe_allow_html=True)
    with metric_cols[3]:
        st.link_button("Open original on arXiv ↗", b.link, use_container_width=True)
    if state.parsed and state.parsed.parse_warning:
        st.markdown(f"<div class='note'>Source note: {escape(state.parsed.parse_warning)}</div>", unsafe_allow_html=True)

    action_col, _ = st.columns([1.35, 3])
    with action_col:
        if st.button("💬 Chat with this paper", type="primary", use_container_width=True):
            st.session_state.workspace_view = "Chat"
    view = st.radio("Workspace", ["Overview", "Chat", "Source"], horizontal=True, key="workspace_view", label_visibility="collapsed")

    if view == "Overview":
        if state.selected_paper and state.selected_paper.categories:
            chips = "".join(f"<span class='chip'>{escape(c)}</span>" for c in state.selected_paper.categories)
            st.markdown(f"<div class='chip-row'>{chips}</div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<div class='eyebrow'>🧭 Why this matters</div>", unsafe_allow_html=True)
            st.write(b.summary)

        with st.container(border=True):
            st.markdown("<div class='eyebrow'>❓ Problem</div>", unsafe_allow_html=True)
            st.write(b.problem_statement)

        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.markdown("<div class='eyebrow'>⚙️ Approach</div>", unsafe_allow_html=True)
                render_bullets(b.method, "method")
            with st.container(border=True):
                st.markdown("<div class='eyebrow'>⚠️ Limitations</div>", unsafe_allow_html=True)
                render_bullets(b.limitations, "limitations")
        with right:
            with st.container(border=True):
                st.markdown("<div class='eyebrow'>📈 Key claims &amp; results</div>", unsafe_allow_html=True)
                render_bullets(b.key_results, "results")
            with st.container(border=True):
                st.markdown("<div class='eyebrow'>💡 Try asking</div>", unsafe_allow_html=True)
                for ov_index, sq in enumerate(useful_suggestions(b.suggested_questions)[:3]):
                    if st.button(sq, key=f"ov-sq-{ov_index}", use_container_width=True):
                        st.session_state.pending_question = sq
                        st.session_state.workspace_view = "Chat"
                        st.rerun()

    elif view == "Chat":
        st.markdown(
            "<div class='note'><strong>Grounded RAG chat.</strong> Broad "
            "questions (\"summarize this\", \"what's it about\") answer from "
            "the briefing; specific questions retrieve the most relevant "
            "indexed paper sections. When the source does not support an "
            "answer, the assistant says so instead of guessing.</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='eyebrow'>Suggested questions</div>", unsafe_allow_html=True)
        suggested = ["🧾 Summarize this paper"] + useful_suggestions(b.suggested_questions)[:5]
        qcols = st.columns(min(3, len(suggested)))
        for index, suggestion in enumerate(suggested):
            label = suggestion.replace("🧾 ", "")
            if qcols[index % len(qcols)].button(suggestion, key=f"suggestion-{index}", use_container_width=True):
                st.session_state.pending_question = label

        st.write("")  # small spacer between chips and the conversation

        if not st.session_state.qa_pairs and "pending_question" not in st.session_state:
            st.markdown(
                "<div class='chat-empty'><div class='big'>💬</div>"
                "Ask anything about this paper — a suggested question above "
                "is a good place to start.</div>",
                unsafe_allow_html=True,
            )
        else:
            for prior_q, prior_a in st.session_state.qa_pairs:
                with st.chat_message("user"):
                    st.write(prior_q)
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(prior_a)

        pending = st.session_state.pop("pending_question", None)
        typed = st.chat_input("Ask about the method, evidence, results, limitations, or a specific claim")
        if pending or typed:
            ask(state, pending or typed)
    else:
        st.caption("These are the extracted sections used to build the local RAG index.")
        if state.parsed and state.parsed.sections:
            for name, text in state.parsed.sections.items():
                with st.expander(name.title(), expanded=name.lower() in {"abstract", "introduction"}):
                    st.write(text)
        else:
            st.info("This saved session retains its searchable index. Re-run the paper to browse extracted sections.")
else:
    st.info("No briefing is available for this session.")