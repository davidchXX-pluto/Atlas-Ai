import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import json
import os
from duckduckgo_search import DDGS

# ---------- constants ----------
MODEL_NAME = "llama-3.3-70b-versatile"

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

# ---------- setup ----------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("❌ GROQ_API_KEY is not set. Please add it in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=api_key)

def internet_search(query, max_results=3):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(r["body"])
    return results

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

# ---------- simulated learning ----------
learning = memory.get("learning", {})
feedback = learning.get("feedback", {})

verbosity = "short" if feedback.get("too_long", 0) > feedback.get("too_simple", 0) else "normal"

# ---------- system prompt ----------
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

# ---------- UI ----------
st.set_page_config(page_title="Atlas-AI", page_icon="🧠", layout="centered")
st.title("🧠 Atlas-AI")
st.caption("A calm, honest assistant for non-technical users.")

# ---------- beta notice ----------
st.info(
    "🧪 **Beta Test Version**\n\n"
    "Atlas-AI is currently in early testing.\n"
    "If something feels confusing, wrong, or broken — that’s expected.\n\n"
    "👉 Please try unusual questions and share feedback."
)

st.warning(
    "⚠️ Atlas-AI may give incorrect or incomplete answers.\n\n"
    "Do **not** rely on it for:\n"
    "- Financial decisions\n"
    "- Medical advice\n"
    "- Legal guidance"
)

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.sidebar:
    st.header("Controls")
    if st.button("🧹 Clear chat"):
        st.session_state.chat = []
        st.experimental_rerun()

# ---------- chat history ----------
for role, content in st.session_state.chat:
    st.chat_message(role).write(content)

# ---------- input ----------
user_input = st.chat_input("Type your question and press Enter")

if user_input:
    st.session_state.chat.append(("user", user_input))
    st.chat_message("user").write(user_input)

    user_input = st.chat_input("Type your question and press Enter")

if user_input:
    st.session_state.chat.append(("user", user_input))
    st.chat_message("user").write(user_input)

    # ---------- rate limiting ----------
    if "requests" not in st.session_state:
        st.session_state.requests = 0

    st.session_state.requests += 1

    if st.session_state.requests > 20:
        st.error("Rate limit reached. Please refresh later.")
        st.stop()

    # ---------- feedback learning ----------
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

    # ---------- simple math ----------
    if any(op in user_input for op in ["+", "-", "*", "/"]) and len(user_input.split()) <= 5:
        try:
            answer = str(eval(user_input))
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

    # ---------- internet routing ----------
    if detected_intent:
        if detected_intent == "weather":
            search_query = "current weather in Delhi today temperature"
            instruction = "Summarize temperature and conditions only."

        elif detected_intent == "fx":
            search_query = f"{user_input} exchange rate today"
            instruction = "Give the exchange rate clearly."

        elif detected_intent == "news":
            search_query = f"latest news {user_input}"
            instruction = "Summarize key headlines only."

        elif detected_intent == "version":
            search_query = f"{user_input} latest version release date"
            instruction = "Give version number and release date."

        with st.spinner("Searching the internet…"):
            results = internet_search(search_query)

        if not results:
            answer = "I couldn’t find recent information right now."
        else:
            context = "\n".join(results)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Here is recent information:\n{context}\n\n"
                            f"{instruction} Say it is based on recent available data."
                        )
                    }
                ],
                temperature=0.3
            )
            answer = response.choices[0].message.content

        st.session_state.chat.append(("assistant", answer))
        st.chat_message("assistant").write(answer)
        st.stop()

    # ---------- normal response ----------
    with st.spinner("Thinking…"):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3
        )
        answer = response.choices[0].message.content

    st.session_state.chat.append(("assistant", answer))
    st.chat_message("assistant").write(answer)
