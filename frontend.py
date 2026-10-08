import os
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from config import (
    AVIATION_STACK_API_KEY,
    OPENWEATHER_API_KEY,
    TAVILY_API_KEY,
)
from graph import app


EXAMPLE_PROMPTS = {
    "A week in Japan": (
        "Plan a 7-day Japan trip under Rs. 2 lakh. I prefer budget hotels "
        "and no overnight flights."
    ),
    "A weekend in Italy": (
        "Plan a relaxed 3-day Rome trip for two, with great local food, "
        "walkable neighborhoods, and a moderate budget."
    ),
}

AGENT_LABELS = {
    "flight_agent": "Flights",
    "hotel_agent": "Stays",
    "weather_agent": "Weather",
    "budget_agent": "Budget",
    "itinerary_agent": "Itinerary",
}

RESULT_SECTIONS = (
    ("flight_results", "Flights", "✈️"),
    ("hotel_results", "Places to stay", "🛏️"),
    ("weather_results", "Weather", "☀️"),
    ("budget_results", "Budget check", "💰"),
)


def _new_thread(user_id: str) -> None:
    st.session_state.thread_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
    st.session_state.pop("latest_result", None)
    st.session_state.pop("waiting_for_approval", None)
    st.session_state.travel_request = ""


def _use_example(prompt: str) -> None:
    st.session_state.travel_request = prompt


st.set_page_config(
    page_title="Wayfarer · AI Travel Planner",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    :root {
        --ink: #162b32;
        --muted: #65777b;
        --paper: #f5f7f3;
        --teal: #176c63;
        --mint: #e3f1e9;
        --line: #dce5df;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 82% 0%, rgba(213, 235, 221, .55), transparent 31rem),
            var(--paper);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #edf2ec;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1160px; padding-top: 2.2rem; padding-bottom: 4rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -.035em; }
    h1, h2 { font-family: 'Playfair Display', Georgia, serif; }
    h1 { font-size: clamp(2.7rem, 6vw, 4.35rem) !important; line-height: 1.05 !important; }
    h2 { font-size: 1.8rem !important; }
    p, label, .stCaption { color: var(--muted); }
    .eyebrow {
        color: var(--teal); font-size: .73rem; font-weight: 700;
        letter-spacing: .16em; text-transform: uppercase; margin-bottom: .7rem;
    }
    .hero-copy { color: var(--muted); font-size: 1.08rem; max-width: 650px; line-height: 1.7; }
    .brand {
        font-family: 'Playfair Display', Georgia, serif; font-size: 1.55rem;
        color: var(--ink); font-weight: 700; margin: 0 0 .25rem;
    }
    .brand-mark { color: var(--teal); }
    .section-kicker {
        color: var(--teal); font-size: .72rem; text-transform: uppercase;
        letter-spacing: .13em; font-weight: 700; margin-bottom: .25rem;
    }
    .agent-pill {
        display: inline-block; background: var(--mint); color: #24564f;
        border: 1px solid #cce3d5; border-radius: 999px;
        font-size: .83rem; font-weight: 600; padding: .4rem .75rem; margin: .2rem .3rem .1rem 0;
    }
    .stButton > button {
        border-radius: 12px; border: 1px solid var(--line); min-height: 2.8rem;
        color: var(--ink); background: rgba(255,255,255,.76); font-weight: 600;
        transition: transform .15s ease, border-color .15s ease, background .15s ease;
    }
    .stButton > button:hover { border-color: var(--teal); color: var(--teal); transform: translateY(-1px); }
    .stButton > button[kind="primary"] {
        background: var(--teal); border-color: var(--teal); color: #fff;
    }
    .stButton > button[kind="primary"]:hover { background: #12594f; color: #fff; }
    div[data-testid="stTextArea"] textarea {
        background: rgba(255,255,255,.82); border: 1px solid var(--line);
        border-radius: 14px; line-height: 1.6;
    }
    div[data-testid="stTextInput"] input {
        background: rgba(255,255,255,.82); border-radius: 10px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,.72); border-color: var(--line); border-radius: 16px;
    }
    [data-testid="stExpander"] { background: rgba(255,255,255,.72); border-radius: 14px; }
    .sidebar-note { font-size: .82rem; line-height: 1.55; color: var(--muted); }
    .thread-label {
        overflow-wrap: anywhere; color: var(--muted); font-size: .78rem;
        background: rgba(255,255,255,.7); border-radius: 8px; padding: .55rem .65rem;
    }
    hr { border-color: var(--line); }
    @media (max-width: 640px) {
        .block-container { padding-top: 1.2rem; }
        h1 { font-size: 2.8rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "user_id" not in st.session_state:
    st.session_state.user_id = "traveler"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"traveler_{uuid.uuid4().hex[:8]}"
if "travel_request" not in st.session_state:
    st.session_state.travel_request = ""

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">✦</span> Wayfarer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-note">Thoughtful trips, planned together.</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-kicker">Your workspace</div>', unsafe_allow_html=True)
    user_id = st.text_input(
        "Traveler ID",
        key="user_id",
        help="Used to keep your planning threads organized.",
    )
    if st.button("＋  Start a new trip", use_container_width=True):
        _new_thread(user_id.strip() or "traveler")
        st.rerun()

    st.markdown(
        f'<div class="thread-label">Current thread<br><strong>{st.session_state.thread_id}</strong></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown('<div class="section-kicker">Connected services</div>', unsafe_allow_html=True)
    service_status = (
        ("Groq AI", bool(os.getenv("GROQ_API_KEY"))),
        ("Travel search", bool(TAVILY_API_KEY)),
        ("Flight data", bool(AVIATION_STACK_API_KEY)),
        ("Weather", bool(OPENWEATHER_API_KEY)),
    )
    for service, connected in service_status:
        marker = "🟢" if connected else "⚪"
        st.markdown(f"{marker} **{service}** {'ready' if connected else 'needs a key'}")
    st.divider()
    st.markdown(
        '<div class="sidebar-note">Plans are suggestions, not bookings. '
        'Confirm prices, schedules, entry rules, and local conditions before you travel.</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="eyebrow">Your next chapter starts here</div>', unsafe_allow_html=True)
st.title("Make room for\nthe journey.")
st.markdown(
    '<p class="hero-copy">Tell us what you have in mind. Your planning team will bring together '
    'flight ideas, stays, local weather, and a practical itinerary—then you decide what feels right.</p>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-kicker">A little inspiration</div>', unsafe_allow_html=True)
example_columns = st.columns(len(EXAMPLE_PROMPTS))
for column, (label, prompt) in zip(example_columns, EXAMPLE_PROMPTS.items()):
    with column:
        st.button(
            label,
            key=f"example_{label}",
            use_container_width=True,
            on_click=_use_example,
            args=(prompt,),
        )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-kicker">Start planning</div>', unsafe_allow_html=True)
query = st.text_area(
    "Describe your trip",
    key="travel_request",
    placeholder=(
        "Where would you like to go, when, and what matters most? "
        "Share your budget, pace, travel companions, or any must-haves."
    ),
    height=145,
    label_visibility="collapsed",
)

plan_column, hint_column = st.columns([1, 2], vertical_alignment="center")
with plan_column:
    create_plan = st.button(
        "✨  Create my draft plan",
        type="primary",
        use_container_width=True,
    )
with hint_column:
    st.caption("Your draft is reviewed by you before a polished final plan is created.")

if create_plan:
    if not query.strip():
        st.warning("Add a destination or a few trip details to get started.")
    else:
        with st.spinner("Your travel team is putting the pieces together…"):
            result = app.invoke(
                {
                    "messages": [HumanMessage(content=query.strip())],
                    "user_id": user_id.strip() or "traveler",
                    "user_query": query.strip(),
                    "flight_results": "",
                    "hotel_results": "",
                    "weather_results": "",
                    "budget_results": "",
                    "itinerary": "",
                    "final_response": "",
                    "llm_calls": 0,
                },
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )
        st.session_state.latest_result = result
        st.session_state.waiting_for_approval = "__interrupt__" in result

result = st.session_state.get("latest_result")
if result:
    st.divider()
    st.markdown('<div class="eyebrow">Your travel brief</div>', unsafe_allow_html=True)
    st.header("The plan is taking shape")

    reasoning = result.get("supervisor_reasoning", "")
    if reasoning:
        st.markdown(reasoning)

    selected_agents = result.get("selected_agents", [])
    if selected_agents:
        st.markdown(
            " ".join(
                f'<span class="agent-pill">{AGENT_LABELS.get(agent, agent.replace("_", " ").title())}</span>'
                for agent in selected_agents
            ),
            unsafe_allow_html=True,
        )

    for start in range(0, len(RESULT_SECTIONS), 2):
        columns = st.columns(2, gap="medium")
        for column, (key, title, icon) in zip(columns, RESULT_SECTIONS[start : start + 2]):
            content = result.get(key)
            if content:
                with column:
                    with st.container(border=True):
                        st.markdown(f"### {icon} {title}")
                        st.markdown(str(content))

    interrupts = result.get("__interrupt__", [])
    draft = (
        getattr(interrupts[0], "value", {}).get("draft_itinerary", "")
        if interrupts
        else result.get("itinerary", "")
    )
    if draft:
        st.markdown('<div class="section-kicker">First draft</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(draft)

if st.session_state.get("waiting_for_approval") and result:
    st.divider()
    st.markdown('<div class="eyebrow">Over to you</div>', unsafe_allow_html=True)
    st.header("Does this feel like your trip?")
    st.caption("Approve the draft to polish it, or request changes with a little feedback.")

    with st.container(border=True):
        decision = st.radio(
            "Your review",
            ["Looks good — polish this plan", "I’d like some changes"],
            horizontal=True,
            label_visibility="collapsed",
        )
        feedback = st.text_area(
            "What should change?",
            placeholder="Tell us what to adjust, add, or rethink.",
            disabled=decision.startswith("Looks good"),
            height=100,
        )
        if st.button("Submit review", type="primary"):
            if decision.startswith("I’d like") and not feedback.strip():
                st.warning("Add a note about what you’d like changed.")
            else:
                with st.spinner("Preparing your final travel plan…"):
                    final_result = app.invoke(
                        Command(
                            resume={
                                "approved": decision.startswith("Looks good"),
                                "feedback": feedback.strip(),
                            }
                        ),
                        config={"configurable": {"thread_id": st.session_state.thread_id}},
                    )
                st.session_state.latest_result = final_result
                st.session_state.waiting_for_approval = False
                st.rerun()

final_result = st.session_state.get("latest_result")
if final_result and final_result.get("final_response"):
    st.divider()
    st.markdown('<div class="eyebrow">Made for you</div>', unsafe_allow_html=True)
    st.header("Your travel plan")
    with st.container(border=True):
        st.markdown(final_result["final_response"])
    st.caption("Before you book, verify all prices, availability, schedules, and travel requirements.")
