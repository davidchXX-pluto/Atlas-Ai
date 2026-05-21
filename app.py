import streamlit.components.v1 as components
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import json
import os
from duckduckgo_search import DDGS
import ast
import operator as op
import requests

# ================= SAFE MATH =================
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
}

def safe_eval(expr):
    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](
                _eval(node.left),
                _eval(node.right),
            )
        else:
            raise ValueError("Unsupported expression")
    return _eval(ast.parse(expr, mode="eval").body)

# ================= CONSTANTS =================
MODEL_NAME = "gemini-2.5-flash"

INTENTS = {
    "weather": ["weather", "temperature", "forecast", "rain", "hot", "cold"],
    "fx": ["exchange rate", "forex", "currency", "usd", "inr", "eur", "gbp"],
    "news": ["news", "latest", "headlines", "breaking"],
    "version": ["version", "latest version", "released"]
}

feedback_triggers = {
    "too_long": ["too long", "shorter", "keep it short"],
    "too_simple": ["too simple", "more detail", "explain more"]
}

# ================= SETUP =================
load_dotenv(dotenv_path=".env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY is not set. Please add it in your .env file.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

def internet_search(query, max_results=3):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(r["body"])
    return results

def get_news_from_newsdata(query="", country="in", max_articles=5):
    api_key = os.getenv("NEWSDATA_API_KEY")
    if not api_key:
        return None

    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": api_key,
        "q": query,
        "country": country,
        "language": "en",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None

    if data.get("status") != "success":
        return None

    articles = data.get("results", [])[:max_articles]
    if not articles:
        return None

    summaries = []
    for a in articles:
        title = a.get("title", "No title")
        source = a.get("source_id", "unknown source")
        summaries.append(f"• {title} ({source})")

    return "\n".join(summaries)

def load_memory():
    try:
        with open("memory.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_memory(memory):
    with open("memory.json", "w") as f:
        json.dump(memory, f, indent=2)

memory = load_memory()

creator_name = memory.get("creator", {}).get("name", "the developer")
creator_purpose = memory.get("creator", {}).get("purpose", "to help people")
explain_style = memory.get("preferences", {}).get("explain_style", "step_by_step")

learning = memory.get("learning", {})
feedback = learning.get("feedback", {})
verbosity = "short" if feedback.get("too_long", 0) > feedback.get("too_simple", 0) else "normal"

# ================= SYSTEM PROMPT =================
system_prompt = (
    f"You are Atlas-AI, an AI assistant created by {creator_name}. "
    f"You were created {creator_purpose}.\n\n"
    "IMPORTANT OVERRIDES:\n"
    "- If asked who created you, answer directly in one short sentence.\n"
    "- If asked about API keys or secrets, say you cannot share them.\n"
    "- If asked about hacking or security, do not invent protections.\n\n"
    f"Explain things in a {explain_style} manner using very simple language.\n"
    f"Adjust answer length to be {verbosity}.\n\n"
    "Never guess facts or calculations. If unsure, say so.\n\n"
)

# ================= UI =================
st.set_page_config(page_title="Atlas-AI", page_icon="🧠", layout="centered")
st.title("🧠 Atlas-AI")
st.caption("A calm, honest assistant for non-technical users.")

st.warning(
    "⚠️ Atlas-AI is in early testing. "
    "Do not rely on it for financial, medical, or legal decisions."
)

if "chat" not in st.session_state:
    st.session_state.chat = []

# ================= CHAT HISTORY =================
for role, content in st.session_state.chat:
    st.chat_message(role).write(content)

# ================= INPUT =================
user_input = st.chat_input("Type your question and press Enter", key="chat_input_main")

if user_input:
    st.session_state.chat.append(("user", user_input))
    st.chat_message("user").write(user_input)

    # ---------- RATE LIMIT ----------
    st.session_state.requests = st.session_state.get("requests", 0) + 1
    if st.session_state.requests > 20:
        st.error("Rate limit reached. Please refresh later.")
        st.stop()

    # ---------- FEEDBACK ----------
    learning = memory.setdefault("learning", {})
    feedback = learning.setdefault("feedback", {"too_long": 0, "too_simple": 0})

    for key, phrases in feedback_triggers.items():
        if any(p in user_input.lower() for p in phrases):
            feedback[key] += 1
            save_memory(memory)
            answer = "Got it. I’ll adjust how I answer from now on."
            st.session_state.chat.append(("assistant", answer))
            st.chat_message("assistant").write(answer)
            st.stop()

    # ---------- SECRET NOTES ----------
    secret_notes = memory.get("secret_notes", {})

    secret_triggered = False

    for person, data in secret_notes.items():
        expected_phrase = f"{person.lower()} {data['trigger'].lower()}"

        if user_input.lower().strip() == expected_phrase:
            answer = data["message"]
            st.session_state.chat.append(("assistant", answer))
            st.chat_message("assistant").write(answer)
            secret_triggered = True
            break

    if secret_triggered:
        st.stop()

    # ---------- SAFE MATH ----------
    if user_input and any(op in user_input for op in ["+", "-", "*", "/"]) and len(user_input.split()) <= 5:
        try:
            answer = str(safe_eval(user_input))
            st.session_state.chat.append(("assistant", answer))
            st.chat_message("assistant").write(answer)
            st.stop()
        except:
            pass

    # ---------- intent detection ----------
    detected_intent = None
    for intent, keywords in INTENTS.items():
        if any(k in user_input.lower() for k in keywords):
            detected_intent = intent
            break

    # ---------- internet routing & API handling ----------
    if detected_intent:

        if detected_intent == "news":
            with st.spinner("Fetching latest news…"):
                news = get_news_from_newsdata(query=user_input)

            if not news:
                answer = "I couldn’t find recent news right now."
            else:
                answer = (
                    "Here are some recent headlines:\n\n"
                    f"{news}\n\n"
                    "Based on recent available data."
                )

            st.session_state.chat.append(("assistant", answer))
            st.chat_message("assistant").write(answer)
            st.stop()

        elif detected_intent == "weather":
            with st.spinner("Fetching weather…"):
                answer = "Weather API not yet integrated."
            st.session_state.chat.append(("assistant", answer))
            st.chat_message("assistant").write(answer)
            st.stop()

        elif detected_intent == "fx":
            with st.spinner("Fetching exchange rate…"):
                answer = "Forex API not yet integrated."
            st.session_state.chat.append(("assistant", answer))
            st.chat_message("assistant").write(answer)
            st.stop()

        elif detected_intent == "version":
            with st.spinner("Fetching version info…"):
                answer = "Version info not yet available."
            st.session_state.chat.append(("assistant", answer))
            st.chat_message("assistant").write(answer)
            st.stop()

    # --- NORMAL response (fallback) ---
    with st.spinner("Thinking…"):
        try:
            response = model.generate_content(
                f"{system_prompt}\n\nUser: {user_input}"
            )
            answer = response.text
        except Exception as e:
            answer = f"Gemini API error: {str(e)}"

    st.session_state.chat.append(("assistant", answer))
    st.chat_message("assistant").write(answer)
