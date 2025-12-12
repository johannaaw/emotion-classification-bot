# app.py — NeMo (Natural Emotion) — enhanced followups, greeting, no-repeat replies

import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import json
from typing import List, Dict
from torch.utils.data import DataLoader
import re

# ---------------------------
# Config (adjust paths here)
# ---------------------------
# MODEL_PATH = "models/bert_nateraw_emotion"
MODEL_PATH = "https://drive.google.com/file/d/1xaN3GU9Sl0KCoK2B_IHGvhyXJV39HiY4/view?usp=sharing"
REPLIES_PATH = "data/replies.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

# ---------------------------
# Page settings
# ---------------------------
st.set_page_config(
    page_title="NeMo – Natural Emotion",
    layout="wide"
)

# ---------------------------
# Utilities
# ---------------------------
def normalize_id2label(raw_id2label):
    norm = {}
    for k, v in raw_id2label.items():
        try:
            k_int = int(k)
        except:
            k_int = k
        norm[int(k_int)] = str(v)
    return norm


@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.to(DEVICE)
    model.eval()

    raw_id2label = getattr(model.config, "id2label", None)
    if raw_id2label is None:
        raw_id2label = {i: str(i) for i in range(model.config.num_labels)}

    id2label_map = normalize_id2label(raw_id2label)
    return tokenizer, model, id2label_map


def batch_predict(texts: List[str], tokenizer, model, id2label_map: Dict[int, str], batch_size=16):
    ds = Dataset.from_dict({"text": texts})

    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length")

    tokenized = ds.map(preprocess, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # mapping from dataset label string -> emotion name
    emotion_names = {
        "0": "sadness",
        "1": "joy",
        "2": "love",
        "3": "anger",
        "4": "fear",
        "5": "surprise"
    }
    
    preds, probs = [], []

    with torch.no_grad():
        for batch in DataLoader(tokenized, batch_size=batch_size):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            soft = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(soft, dim=-1).cpu().numpy()

            for i, pid in enumerate(pred_ids):
                dataset_label = id2label_map.get(int(pid), str(pid))
                emotion_name = emotion_names.get(dataset_label, dataset_label)
                preds.append(dataset_label)
                probs.append(float(soft[i, pid]))

    return preds, probs


# ---------------------------
# Load model + replies
# ---------------------------
tokenizer, model, id2label_map = load_model_and_tokenizer(MODEL_PATH)

try:
    with open(REPLIES_PATH, "r", encoding="utf-8") as f:
        replies_db = json.load(f)
except FileNotFoundError:
    # default replies; you can also provide *_followup pools like "joy_followup"
    replies_db = {
        "sadness": [
            "I'm sorry you're feeling this way. I'm here to listen.",
            "That sounds heavy — do you want to tell me what happened?"
        ],
        "joy": [
            "That's wonderful! Tell me more.",
            "Amazing — I'm happy for you!"
        ],
        "love": [
            "That sounds heartfelt ❤️",
            "So sweet — would you like to share more?"
        ],
        "anger": [
            "I hear your frustration. Want to talk about it?",
            "That sounds upsetting. Would talking it through help?"
        ],
        "fear": [
            "That must be scary. You're not alone.",
            "I hear your concern — want to tell me more about it?"
        ],
        "surprise": [
            "Wow — that was unexpected! How do you feel?",
            "That must have been quite a shock!"
        ],
        # optional followup pools (examples)
        "joy_followup": [
            "Still excited? Tell me what's keeping you excited.",
            "What part of that made you happiest?"
        ],
        "sadness_followup": [
            "Are you feeling a bit better or is it still heavy?",
            "Do you want some ideas to cope with this feeling right now?"
        ]
    }

# Adjust based on your dataset labels
dataset_label_to_name = {
    "0": "sadness",
    "1": "joy",
    "2": "love",
    "3": "anger",
    "4": "fear",
    "5": "surprise"
}

# ---------------------------
# Sidebar: NeMo Profile
# ---------------------------
st.sidebar.title("NeMo Profile")
user_name = st.sidebar.text_input("Your name", value="")
preferred_tone = st.sidebar.selectbox("Tone", ["friendly", "formal", "casual"])
show_probs = st.sidebar.checkbox("Show prediction confidence", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** `" + MODEL_PATH + "`")


# ---------------------------
# Chat State
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # (role, text, emotion, prob)

# rotation index for replies per emotion to reduce repetition
if "reply_idx" not in st.session_state:
    st.session_state.reply_idx = {}  # map emotion -> next index

# Initialize 'input' BEFORE the widget is created to avoid modifying widget-backed keys later
if "input" not in st.session_state:
    st.session_state["input"] = ""


# ---------------------------
# Conversation helpers (greetings, followups, context)
# ---------------------------
GREETINGS = {
    "hello", "hi", "hey", "heythere", "good morning", "good afternoon", "good evening",
    "hiya", "sup", "yo", "howdy", "greetings"
}
# more robust greetings pattern to catch hi!, hello., hiya etc.
greeting_re = re.compile(r'^\s*(hi|hello|hey|hiya|howdy|yo|sup|good morning|good afternoon|good evening)\b[^\n]*$', re.I)

# follow-up keywords: short replies or explicit follow-up signals
FOLLOWUP_KEYWORDS = {"yes", "no", "still", "again", "also", "same", "Not really".lower(), "same", "ok", "okay", "k", "sure", "yeah", "yep"}

def is_greeting_only(text: str, first_user_message: bool) -> bool:
    """
    Detect if text is a greeting. Only treat as first-message greeting when first_user_message=True.
    We'll consider short messages that match greeting_re or contain only greeting tokens.
    """
    if not text or not first_user_message:
        return False
    s = text.strip()
    if greeting_re.match(s):
        return True
    # fallback: token-level check (e.g., "hi!" or "hello :)")
    tokens = re.findall(r"[a-zA-Z]+", s.lower())
    if len(tokens) <= 2 and all(t in GREETINGS for t in tokens):
        return True
    return False

def is_followup(text: str) -> bool:
    """
    Simple heuristic to detect follow-up replies:
    - short messages (<=5 tokens)
    - or contain common follow-up words/phrases
    """
    if not text:
        return False
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z]+", text)]
    if len(tokens) <= 5:
        # consider short answers like "yes", "no", "still sad", "not really"
        if any(t in FOLLOWUP_KEYWORDS for t in tokens):
            return True
        # pure short tokens (e.g., "yes" or "no") are followups
        if len(tokens) <= 2:
            return True
    # also if text begins with "and" or "also" it's likely a followup
    if text.strip().lower().startswith(("and ", "also ")):
        return True
    return False

def choose_reply_for_emotion(emotion: str, followup: bool=False, avoid_text: str=None) -> str:
    """
    Select a reply template for the given emotion.
    - If followup=True, tries an emotion_followup pool (e.g. "joy_followup")
    - Avoid returning the same reply text as avoid_text (last bot message)
    - Use session_state.reply_idx to rotate replies per emotion
    """
    # select pool
    pool_name = emotion + "_followup" if followup and (emotion + "_followup") in replies_db else emotion
    pool = replies_db.get(pool_name, replies_db.get(emotion, ["I see. Tell me more."]))

    # ensure we have rotation index
    idx_map = st.session_state.reply_idx
    if pool_name not in idx_map:
        idx_map[pool_name] = 0

    # try up to len(pool) times to pick a different text than avoid_text
    chosen = None
    for _ in range(len(pool)):
        i = idx_map[pool_name] % len(pool)
        candidate = pool[i]
        idx_map[pool_name] = (idx_map[pool_name] + 1) % len(pool)  # advance for next time
        if avoid_text is None or candidate.strip() != avoid_text.strip():
            chosen = candidate
            break
    if chosen is None:
        # fallback: just pick a random one
        chosen = pool[0]
    return chosen

# ---------------------------
# Main UI
# ---------------------------
st.title("🤖 NeMo – Natural Emotion")
st.write("A lightweight, emotion-aware conversational assistant.")

chat_box = st.container()

def render_chat():
    chat_box.empty()
    with chat_box:
        for (role, text, emo, prob) in st.session_state.history:
            if role == "user":
                st.markdown(f"**You:** {text}")
            else:
                label = f"**[{emo}]**" if emo else ""
                if show_probs and prob is not None:
                    st.markdown(f"**NeMo:** {text} {label}  \n_confidence: {prob:.2f}_")
                else:
                    st.markdown(f"**NeMo:** {text} {label}")

render_chat()

# Callback moved here so it can safely modify st.session_state["input"]
def send_message():
    msg = st.session_state.get("input", "").strip()
    if not msg:
        return

    # Count how many prior user messages to determine "first user message"
    prior_user_msgs = sum(1 for r in st.session_state.history if r[0] == "user")
    first_user_message = prior_user_msgs == 0

    # Greeting check: only on the very first user message, treat short greeting specially
    if is_greeting_only(msg, first_user_message):
        # reply with greeting and ask how user is feeling (don't run model)
        greet_text = f"Hello{(' ' + (user_name or '') ) if user_name else ''}! Nice to meet you. How are you feeling today?"
        st.session_state.history.append(("bot", greet_text, "greeting", None))
        # clear input
        st.session_state["input"] = ""
        return

    # Append user message
    st.session_state.history.append(("user", msg, None, None))

    # Detect follow-up (heuristic)
    followup_flag = is_followup(msg)

    # Build context: if you want to include previous N turns, you can build here (omitted for brevity)
    # Predict emotion for current message (you can change to feed concatenated context)
    preds, probs = batch_predict([msg], tokenizer, model, id2label_map)
    raw_label = preds[0]
    prob = probs[0]
    emotion = dataset_label_to_name.get(raw_label, raw_label)

    # Determine last bot's emotion & last bot text (for avoid/repeat logic)
    last_bot_emo = None
    last_bot_text = None
    # walk history backwards to find last bot message
    for role, txt, emo, p in reversed(st.session_state.history[:-1]):  # exclude the user message appended above
        if role == "bot":
            last_bot_emo = emo
            last_bot_text = txt
            break

    # If the predicted emotion is the same as last bot emotion, try to pick a DIFFERENT reply
    avoid_text = None
    if last_bot_emo == emotion and last_bot_text:
        avoid_text = last_bot_text

    # Choose reply, prefer followup pool if heuristic suggests it's a followup
    reply_template = choose_reply_for_emotion(emotion, followup=followup_flag, avoid_text=avoid_text)

    # Personalize
    reply = (
        reply_template
        .replace("{{name}}", user_name or "")
        .replace("{{tone}}", preferred_tone)
    )

    # Append bot reply with emotion and prob
    st.session_state.history.append(("bot", reply, emotion, prob))

    # CLEAR the text area value — allowed here because we're inside a callback
    st.session_state["input"] = ""

    # After modification Streamlit will re-run and render_chat() will show updated history

# Use a widget with the same key; on_click calls the callback which will clear the session_state key
st.text_area("Type your message here:", key="input", height=80)
st.button("Send", on_click=send_message)

# ---------------------------
# Debug (optional)
# ---------------------------
# with st.expander("Label Mapping (Debug)"):
#     st.write(id2label_map)
#     st.write(dataset_label_to_name)
#     st.write("reply_idx:", st.session_state.reply_idx)
