import json
import os
from groq import Groq
from dotenv import load_dotenv
from duckduckgo_search import DDGS

# ---------- helpers ----------
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

# ---------- setup ----------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

print("Atlas-AI: Ask me something (type 'exit' to quit)")

# ---------- main loop ----------
while True:
    user_input = input("You: ").strip()

    # EXIT
    if user_input.lower() == "exit":
        print("Atlas-AI: Later.")
        break

    # FEEDBACK LEARNING
    learning = memory.setdefault("learning", {})
    feedback = learning.setdefault("feedback", {"too_long": 0, "too_simple": 0})

    feedback_hit = False
    for key, phrases in feedback_triggers.items():
        if any(p in user_input.lower() for p in phrases):
            feedback[key] += 1
            save_memory(memory)
            print("Atlas-AI: Got it. I’ll adjust how I answer from now on.")
            feedback_hit = True
            break

    if feedback_hit:
        continue

    # SIMPLE MATH
    if any(op in user_input for op in ["+", "-", "*", "/"]) and len(user_input.split()) <= 5:
        try:
            print("Atlas-AI:", eval(user_input))
            continue
        except:
            pass

    # INTENT DETECTION
    detected_intent = None
    for intent, keywords in INTENTS.items():
        if any(k in user_input.lower() for k in keywords):
            detected_intent = intent
            break

    # INTERNET ROUTING
    if detected_intent:
        if detected_intent == "weather":
            search_query = "current weather in Delhi today temperature"
            instruction = "Summarize temperature and conditions only."

        elif detected_intent == "fx":
            search_query = f"{user_input} exchange rate today"
            instruction = "Give the exchange rate clearly. Do not explain forex."

        elif detected_intent == "news":
            search_query = f"latest news {user_input}"
            instruction = "Summarize key headlines only."

        elif detected_intent == "version":
            search_query = f"{user_input} latest version release date"
            instruction = "Give version number and release date."

        results = internet_search(search_query)

        if not results:
            print("Atlas-AI: I couldn’t find recent information right now.")
            continue

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

        print("Atlas-AI:", response.choices[0].message.content)
        continue

    # NORMAL RESPONSE
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.3
    )

    print("Atlas-AI:", response.choices[0].message.content)
