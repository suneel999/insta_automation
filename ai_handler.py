"""
Instagram AI Handler for Optimum Nutrition (ON)
Retrieval-grounded, memory-aware, and strict about using only KB facts.
"""

import os
import re
import json
import logging
import difflib
import sqlite3
import threading
import time
from typing import Dict, Optional
from types import SimpleNamespace
try:
    import redis as redis_lib
except Exception:
    redis_lib = None
from google import genai
import requests
from dotenv import load_dotenv
from on_knowledge import ON_KNOWLEDGE_BASE

# Load local env
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logger = logging.getLogger(__name__)

# --- MEMORY STORE ---
USER_STATES = {}  # { user_id: ConversationState }
CONVERSATION_HISTORY = {}  # { user_id: [str] }
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "25"))
DB_LOCK = threading.Lock()
STATE_DB_PATH = os.getenv("STATE_DB_PATH", os.path.join(os.path.dirname(__file__), "conversation_state.db"))
STATE_TTL_SECONDS = int(os.getenv("STATE_TTL_SECONDS", "43200"))
REDIS_URL = os.getenv("REDIS_URL", "").strip()
REDIS_PREFIX = os.getenv("STATE_REDIS_PREFIX", "on:conv:")
_REDIS_CLIENT = None
AI_RATE_LIMIT_COOLDOWN_SECONDS = int(os.getenv("AI_RATE_LIMIT_COOLDOWN_SECONDS", "60"))

COUNTRY_ALIASES = {
    "uae": "UAE",
    "emirates": "UAE",
    "emirats": "UAE",
    "dubai": "UAE",
    "ksa": "KSA",
    "saudi": "KSA",
    "saoodi": "KSA",
    "saudii": "KSA",
    "sudia": "KSA",
    "saudi arabia": "KSA",
    "egypt": "Egypt",
    "egyptt": "Egypt",
    "egpyt": "Egypt",
    "eg": "Egypt",
}

CATEGORY_PRODUCTS = {
    "protein": [
        "Gold Standard 100% Whey",
        "Serious Mass (Gainer)",
        "Gold Standard 100% Isolate",
        "Platinum HydroWhey",
    ],
    "energy": [
        "Essential Amin.O. Energy",
    ],
    "pre-workout": [
        "Gold Standard Pre-Workout",
    ],
    "recovery": [
        "Gold Standard 100% Casein",
        "Glutamine Powder",
        "BCAA 5000",
    ],
    "vitamins": [
        "Opti-Men",
        "Opti-Women",
        "Fish Oil Softgels",
        "Micronized Creatine Powder",
    ],
}

CATEGORY_ALIASES = {
    "protein": "protein",
    "proteins": "protein",
    "whey": "protein",
    "energy": "energy",
    "amino": "energy",
    "aminos": "energy",
    "pre workout": "pre-workout",
    "pre-workout": "pre-workout",
    "recovery": "recovery",
    "vitamin": "vitamins",
    "vitamins": "vitamins",
    "health": "vitamins",
}
PRODUCT_TO_CATEGORY = {
    product: category
    for category, products in CATEGORY_PRODUCTS.items()
    for product in products
}


class ConversationState:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.mode = "discovery"  # discovery | transaction
        self.pending_intent = None
        self.selected_category = None
        self.selected_product = None
        self.selected_pack = None
        self.selected_country = None
        self.last_asked = None
        self.last_product_mentioned = None
        self.last_pack_mentioned = None
        self.last_country_mentioned = None
        self.last_priced_product = None
        self.last_priced_pack = None
        self.last_priced_country = None
        self.requested_both_packs = False

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "mode": self.mode,
            "pending_intent": self.pending_intent,
            "selected_category": self.selected_category,
            "selected_product": self.selected_product,
            "selected_pack": self.selected_pack,
            "selected_country": self.selected_country,
            "last_asked": self.last_asked,
            "last_product_mentioned": self.last_product_mentioned,
            "last_pack_mentioned": self.last_pack_mentioned,
            "last_country_mentioned": self.last_country_mentioned,
            "last_priced_product": self.last_priced_product,
            "last_priced_pack": self.last_priced_pack,
            "last_priced_country": self.last_priced_country,
            "requested_both_packs": self.requested_both_packs,
        }

    @classmethod
    def from_dict(cls, payload: dict):
        user_id = payload.get("user_id") or "default"
        state = cls(user_id)
        for key, value in payload.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state


# Configure OpenAI (primary) and Gemini (fallback)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
MODEL_CANDIDATES = [
    os.getenv("GEMINI_MODEL", "").strip(),
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]
MODEL_CANDIDATES = [m for m in MODEL_CANDIDATES if m]
WORKING_MODEL = None
AI_DISABLED = False
AI_DISABLED_UNTIL = 0


def _generate_with_openai(contents: str, config: dict):
    if not OPENAI_API_KEY:
        return None
    system_instruction = config.get("system_instruction") or (
        "You are a concise assistant. Return only the requested output."
    )
    temperature = float(config.get("temperature", 0.2))
    max_tokens = int(config.get("max_output_tokens", 220))
    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": contents},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if config.get("response_mime_type") == "application/json":
        body["response_format"] = {"type": "json_object"}
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=20,
    )
    if resp.status_code == 429:
        raise RuntimeError("429")
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text[:240]}")
    data = resp.json()
    text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    return SimpleNamespace(text=text)


def _ensure_state_db() -> None:
    with DB_LOCK:
        conn = sqlite3.connect(STATE_DB_PATH)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    user_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    history_json TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


def _get_redis_client():
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT
    if not REDIS_URL or redis_lib is None:
        return None
    try:
        _REDIS_CLIENT = redis_lib.from_url(REDIS_URL, decode_responses=True)
        _REDIS_CLIENT.ping()
        return _REDIS_CLIENT
    except Exception as e:
        logger.error(f"Redis unavailable, fallback to sqlite: {e}")
        _REDIS_CLIENT = None
        return None


def _load_user_context(user_id: str):
    if user_id in USER_STATES and user_id in CONVERSATION_HISTORY:
        return USER_STATES[user_id], CONVERSATION_HISTORY[user_id]

    r = _get_redis_client()
    if r is not None:
        key = f"{REDIS_PREFIX}{user_id}"
        try:
            raw = r.get(key)
            if raw:
                payload = json.loads(raw)
                state = ConversationState.from_dict(payload.get("state", {"user_id": user_id}))
                history = payload.get("history", [])
                if not isinstance(history, list):
                    history = []
                USER_STATES[user_id] = state
                CONVERSATION_HISTORY[user_id] = history[-MAX_HISTORY_TURNS:]
                return state, CONVERSATION_HISTORY[user_id]
        except Exception as e:
            logger.error(f"Redis read failed for {user_id}: {e}")

    _ensure_state_db()
    with DB_LOCK:
        conn = sqlite3.connect(STATE_DB_PATH)
        try:
            row = conn.execute(
                "SELECT state_json, history_json, updated_at FROM conversations WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        finally:
            conn.close()

    if not row:
        state = ConversationState(user_id)
        history = []
    else:
        updated_at = int(row[2] or 0)
        is_stale = STATE_TTL_SECONDS > 0 and (int(time.time()) - updated_at) > STATE_TTL_SECONDS
        if is_stale:
            state = ConversationState(user_id)
            history = []
        else:
            state = ConversationState.from_dict(json.loads(row[0]))
            history = json.loads(row[1])
            if not isinstance(history, list):
                history = []

    USER_STATES[user_id] = state
    CONVERSATION_HISTORY[user_id] = history[-MAX_HISTORY_TURNS:]
    return state, history


def _save_user_context(user_id: str, state: ConversationState, history: list) -> None:
    USER_STATES[user_id] = state
    CONVERSATION_HISTORY[user_id] = history[-MAX_HISTORY_TURNS:]

    r = _get_redis_client()
    if r is not None:
        key = f"{REDIS_PREFIX}{user_id}"
        payload = {
            "state": state.to_dict(),
            "history": history[-10:],
            "updated_at": int(time.time()),
        }
        try:
            r.set(key, json.dumps(payload, ensure_ascii=True), ex=STATE_TTL_SECONDS)
            return
        except Exception as e:
            logger.error(f"Redis write failed for {user_id}: {e}")

    _ensure_state_db()
    with DB_LOCK:
        conn = sqlite3.connect(STATE_DB_PATH)
        try:
            conn.execute(
                """
                INSERT INTO conversations (user_id, state_json, history_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                  state_json = excluded.state_json,
                  history_json = excluded.history_json,
                  updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    json.dumps(state.to_dict(), ensure_ascii=True),
                    json.dumps(history[-10:], ensure_ascii=True),
                    int(time.time()),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def _safety_guard_response(message: str, state: ConversationState, entities: dict, response_text: str) -> str:
    if state.pending_intent != "price":
        return response_text

    text = response_text.lower()
    looks_like_discovery = "main categories" in text or "which category do you want" in text
    if not looks_like_discovery:
        return response_text

    if not state.selected_product:
        return _ask_for_missing(state, "product")
    if _needs_pack(state.selected_product) and not state.selected_pack and not state.requested_both_packs:
        return _ask_for_missing(state, "pack")
    if not state.selected_country:
        return _ask_for_missing(state, "country")
    return response_text


def _semantic_guard_response(message: str, state: ConversationState, entities: dict, response_text: str) -> str:
    text = response_text.lower()

    # 1) Listing request must not drift into pricing.
    if _is_category_listing_request(message):
        if "price in" in text or "which country" in text:
            return _handle_discovery(state)

    # 2) If we have enough slots for price and response doesn't answer price, enforce deterministic resolution.
    if (entities.get("intent") == "price" or state.pending_intent == "price") and state.selected_product:
        country = state.selected_country or entities.get("country")
        if country and "price in" not in text and "do not have" not in text:
            if state.requested_both_packs and _needs_pack(state.selected_product):
                packs = PRICING_DATA.get(state.selected_product, {}).get("packs", {})
                p2 = packs.get("2LB", {}).get(country)
                p5 = packs.get("5LB", {}).get(country)
                if p2 and p5:
                    facts = (
                        f"{state.selected_product} prices in {country}: 2LB is {p2} and 5LB is {p5}. "
                        "If you want, I can also share another country."
                    )
                    return _grounded_reply(facts)
            resolved = _resolve_price(state.selected_product, country, state.selected_pack)
            if resolved.get("status") == "ok":
                pack_text = f" ({resolved['pack']})" if resolved.get("pack") else ""
                facts = (
                    f"{state.selected_product}{pack_text} price in {country} is {resolved['value']}. "
                    "If you want, I can also share prices for another country."
                )
                return _grounded_reply(facts)

    # 3) Pronoun follow-up should not bounce to generic category lists.
    if _is_pronoun_reference(message) and state.last_product_mentioned:
        if "main categories" in text or "choose one category" in text:
            return _handle_product_info(state.last_product_mentioned, state)

    return response_text


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


def _to_pack(raw: str) -> str:
    value = raw.upper().replace(" ", "")
    return value.replace("LBS", "LB")


def _extract_country(segment: str) -> Optional[str]:
    clean = _normalize(segment)
    for alias, canonical in COUNTRY_ALIASES.items():
        if alias in clean:
            return canonical

    # Fuzzy country detection for typos.
    alias_keys = list(COUNTRY_ALIASES.keys())
    for token in clean.split():
        if len(token) < 3:
            continue
        match = difflib.get_close_matches(token, alias_keys, n=1, cutoff=0.74)
        if match:
            return COUNTRY_ALIASES[match[0]]
    return None


def _extract_prices(price_blob: str) -> Dict[str, str]:
    prices = {}
    for part in [p.strip() for p in price_blob.split("/")]:
        m = re.match(r"^(AED|SAR)\s*([0-9][0-9,]*\.?[0-9]*)$", part, flags=re.IGNORECASE)
        if m:
            prices["UAE" if m.group(1).upper() == "AED" else "KSA"] = f"{m.group(1).upper()} {m.group(2)}"
            continue
        m = re.match(r"^([0-9][0-9,]*\.?[0-9]*)\s*LE$", part, flags=re.IGNORECASE)
        if m:
            prices["Egypt"] = f"{m.group(1)} LE"
    return prices


def _parse_pricing_kb(kb_text: str) -> Dict[str, dict]:
    products = {}
    in_pricing = False
    current_product = None

    for line in kb_text.splitlines():
        raw = line.strip()
        if raw == "=== SECTION: PRICING ===":
            in_pricing = True
            continue
        if in_pricing and raw.startswith("=== SECTION:"):
            break
        if not in_pricing or not raw:
            continue

        m_product = re.match(r"^\d+\.\s*(.+):$", raw)
        if m_product:
            current_product = m_product.group(1).strip()
            products[current_product] = {"prices": {}, "packs": {}, "note": None}
            continue

        if not current_product or not raw.startswith("-"):
            continue

        content = raw[1:].strip()
        note_match = re.search(r"\(([^)]*check[^)]*)\)", content, flags=re.IGNORECASE)
        if note_match:
            products[current_product]["note"] = note_match.group(1).strip()
            content = re.sub(r"\([^)]*\)", "", content).strip()

        m_pack = re.match(r"^([0-9]+\s*LB[S]?)\s*:\s*(.+)$", content, flags=re.IGNORECASE)
        if m_pack:
            pack = _to_pack(m_pack.group(1))
            prices = _extract_prices(m_pack.group(2).strip())
            products[current_product]["packs"][pack] = prices
            continue

        if ":" in content:
            content = content.split(":", 1)[1].strip()
        prices = _extract_prices(content)
        if prices:
            products[current_product]["prices"] = prices

    return products


def _parse_kb_sections(kb_text: str) -> Dict[str, str]:
    sections = {}
    current = None
    buf = []
    for line in kb_text.splitlines():
        line = line.rstrip()
        m = re.match(r"^=== SECTION:\s*(.+?)\s*===$", line)
        if m:
            if current:
                sections[current] = "\n".join(buf).strip()
            current = m.group(1).strip()
            buf = []
            continue
        if current:
            buf.append(line)
    if current:
        sections[current] = "\n".join(buf).strip()
    return sections


PRICING_DATA = _parse_pricing_kb(ON_KNOWLEDGE_BASE)
KB_SECTIONS = _parse_kb_sections(ON_KNOWLEDGE_BASE)
PRODUCTS = list(PRICING_DATA.keys())
PRODUCT_BY_NORMALIZED_NAME = {_normalize(p): p for p in PRODUCTS}
PRODUCT_ALIASES = {
    "whey": "Gold Standard 100% Whey",
    "gold standard whey": "Gold Standard 100% Whey",
    "gs whey": "Gold Standard 100% Whey",
    "serious mass": "Serious Mass (Gainer)",
    "gainer": "Serious Mass (Gainer)",
    "mass": "Serious Mass (Gainer)",
    "casein": "Gold Standard 100% Casein",
    "amino": "Essential Amin.O. Energy",
    "amino energy": "Essential Amin.O. Energy",
    "pre workout": "Gold Standard Pre-Workout",
    "creatine": "Micronized Creatine Powder",
    "isolate": "Gold Standard 100% Isolate",
    "opti men": "Opti-Men",
    "optimen": "Opti-Men",
    "opti women": "Opti-Women",
    "optiwomen": "Opti-Women",
    "fish oil": "Fish Oil Softgels",
    "glutamine": "Glutamine Powder",
    "bcaa": "BCAA 5000",
    "hydro": "Platinum HydroWhey",
    "hydrowhey": "Platinum HydroWhey",
    "platinum hydro whey": "Platinum HydroWhey",
}
INTENT_NOISE_WORDS = {
    "price",
    "cost",
    "how",
    "much",
    "show",
    "products",
    "product",
    "category",
    "categories",
    "what",
    "about",
    "and",
    "in",
    "for",
    "the",
    "a",
    "an",
    "me",
    "tell",
    "please",
    "need",
    "want",
    "know",
}
PRICE_KEYWORDS = ["price", "cost", "how much", "rate"]
DISCOVERY_KEYWORDS = ["browse", "products", "show products", "what do you have", "category", "categories"]
GREETING_KEYWORDS = ["hi", "hello", "hey"]
AUTH_KEYWORDS = ["original", "authentic", "sticker", "fake", "genuine"]
DIETARY_KEYWORDS = ["gluten", "vegan", "diet", "nutritionist", "isolate vs whey", "whey vs isolate"]
WHERE_BUY_KEYWORDS = ["where buy", "where to buy", "where can i find", "find your products", "available at", "where available"]
SMALLTALK_KEYWORDS = ["thanks", "thank you", "ok", "okay", "great", "awesome", "cool", "good morning", "good evening"]
RESPONSE_STYLE = os.getenv("RESPONSE_STYLE", "friendly_concise")


def _detect_pack(message: str) -> Optional[str]:
    m = re.search(r"\b([0-9]+\s*lb[s]?)\b", message, flags=re.IGNORECASE)
    return _to_pack(m.group(1)) if m else None


def _wants_both_packs(message: str) -> bool:
    msg = _normalize(message)
    if msg.strip() == "both":
        return True
    if "both" in msg and any(k in msg for k in ["price", "prices", "pack", "lb", "size", "2", "5"]):
        return True
    if "2lb" in msg and "5lb" in msg:
        return True
    if "2 lb" in msg and "5 lb" in msg:
        return True
    # typo-friendly for "prices" / "process" style misspellings
    if "both" in msg and difflib.get_close_matches("price", msg.split(), n=1, cutoff=0.6):
        return True
    return False


def _detect_country(message: str) -> Optional[str]:
    msg = _normalize(message)
    for alias, canonical in COUNTRY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", msg):
            return canonical
    return _extract_country(message)


def _contains_fuzzy_keyword(msg: str, keywords: list, cutoff: float = 0.78) -> bool:
    for kw in keywords:
        if kw in msg:
            return True

    single_word_keywords = [k for k in keywords if " " not in k]
    for token in msg.split():
        if len(token) < 3:
            continue
        if difflib.get_close_matches(token, single_word_keywords, n=1, cutoff=cutoff):
            return True
    return False


def _contains_dietary_marker(msg: str) -> bool:
    if any(k in msg for k in ["gluten", "vegan", "nutritionist", "whey vs isolate", "isolate vs whey"]):
        return True
    tokens = msg.split()
    markers = ["gluten", "vegan", "nutritionist"]
    for token in tokens:
        if len(token) < 4:
            continue
        if difflib.get_close_matches(token, markers, n=1, cutoff=0.72):
            return True
    return False


def _is_greeting(msg: str) -> bool:
    words = set(msg.split())
    if words.intersection({"hi", "hello", "hey", "hii", "heyy", "heyyy", "morning", "afternoon", "evening"}):
        return True
    if any(re.fullmatch(r"h+i+", w) for w in words):
        return True
    # Tolerate distorted greeting spellings like "helooo", "hyy", "helloooo".
    for w in words:
        if len(w) < 2:
            continue
        simplified = re.sub(r"(.)\1{2,}", r"\1\1", w)
        if simplified.startswith("hel"):
            return True
        if difflib.get_close_matches(w, ["hi", "hey", "hello"], n=1, cutoff=0.65):
            return True
        if difflib.get_close_matches(simplified, ["hi", "hey", "hello"], n=1, cutoff=0.65):
            return True
    return False


def _is_broad_listing_request(message: str) -> bool:
    msg = _normalize(message)
    if _detect_category(message):
        return False
    broad_patterns = [
        "what products",
        "list all products",
        "list products",
        "all products",
        "all your products",
        "show all products",
        "products you have",
    ]
    if any(p in msg for p in broad_patterns):
        return True

    tokens = set(msg.split())
    has_product = "product" in tokens or "products" in tokens
    has_listing_verb = bool(tokens.intersection({"list", "show", "have", "give", "share"}))
    has_broad_scope = bool(tokens.intersection({"all", "everything", "full"}))
    return has_product and (has_listing_verb or has_broad_scope)


def _is_category_listing_request(message: str) -> bool:
    msg = _normalize(message)
    has_list_verb = any(v in msg for v in ["list", "show", "give", "share"])
    has_category_word = any(w in msg for w in ["category", "categories", "product", "products"])
    has_category_value = _detect_category(message) is not None
    has_price_signal = _contains_fuzzy_keyword(msg, PRICE_KEYWORDS)
    return has_category_value and (has_list_verb or has_category_word) and not has_price_signal


def _is_affirmative(message: str) -> bool:
    msg = _normalize(message)
    affirm = {
        "yes",
        "yes please",
        "yep",
        "yeah",
        "sure",
        "ok",
        "okay",
        "please",
        "do it",
        "go ahead",
    }
    return msg in affirm


def _is_pronoun_reference(message: str) -> bool:
    msg = _normalize(message)
    return any(p in msg.split() for p in {"this", "that"}) or "this one" in msg or "that one" in msg


def _category_product_pool(state: Optional[ConversationState]) -> list:
    if not state or not state.selected_category:
        return PRODUCTS
    return CATEGORY_PRODUCTS.get(state.selected_category, PRODUCTS)


def _score_product_match(message: str, product: str) -> int:
    msg_tokens = [t for t in _normalize(message).split() if t and t not in INTENT_NOISE_WORDS]
    if not msg_tokens:
        return 0
    product_tokens = [t for t in _normalize(product).split() if t and t not in {"100"}]
    score = len(set(msg_tokens).intersection(set(product_tokens)))
    return score


def _detect_product(message: str, state: Optional[ConversationState] = None) -> Optional[str]:
    msg = _normalize(message)

    if msg in PRODUCT_BY_NORMALIZED_NAME:
        return PRODUCT_BY_NORMALIZED_NAME[msg]
    if msg in PRODUCT_ALIASES:
        return PRODUCT_ALIASES[msg]

    for alias, product in sorted(PRODUCT_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", msg):
            return product

    # Default for common shorthand.
    if "gold standard" in msg and not any(k in msg for k in ["isolate", "pre workout", "pre-workout", "casein"]):
        if state and state.selected_category and state.selected_category != "protein":
            return None
        return "Gold Standard 100% Whey"

    for normalized, product in PRODUCT_BY_NORMALIZED_NAME.items():
        if re.search(rf"\b{re.escape(normalized)}\b", msg):
            return product

    # Fuzzy alias matching for typos like "hidro", "isolatee", "serios mass".
    alias_keys = list(PRODUCT_ALIASES.keys()) + list(PRODUCT_BY_NORMALIZED_NAME.keys())
    phrase_candidates = [msg]
    tokens = msg.split()
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 5, len(tokens) + 1)):
            phrase_candidates.append(" ".join(tokens[i:j]))

    best_key = None
    best_ratio = 0.0
    for phrase in phrase_candidates:
        if len(phrase) < 3:
            continue
        match = difflib.get_close_matches(phrase, alias_keys, n=1, cutoff=0.78)
        if not match:
            continue
        ratio = difflib.SequenceMatcher(None, phrase, match[0]).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_key = match[0]

    if best_key:
        if best_key in PRODUCT_ALIASES:
            return PRODUCT_ALIASES[best_key]
        if best_key in PRODUCT_BY_NORMALIZED_NAME:
            return PRODUCT_BY_NORMALIZED_NAME[best_key]

    # Fuzzy token overlap for partial names like "gold standard cost".
    pool = _category_product_pool(state)
    scored = [(p, _score_product_match(message, p)) for p in pool]
    scored = [item for item in scored if item[1] > 0]
    if not scored:
        return None

    best_score = max(s for _, s in scored)
    best = [p for p, s in scored if s == best_score]
    if len(best) == 1:
        return best[0]

    # Tie-breaker by category order so replies stay deterministic.
    if state and state.selected_category in CATEGORY_PRODUCTS:
        for p in CATEGORY_PRODUCTS[state.selected_category]:
            if p in best:
                return p
    return None


def _has_explicit_product_mention(message: str) -> bool:
    msg = _normalize(message)
    if not msg:
        return False
    if msg in PRODUCT_ALIASES or msg in PRODUCT_BY_NORMALIZED_NAME:
        return True
    for alias in sorted(PRODUCT_ALIASES.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", msg):
            return True
    for normalized in sorted(PRODUCT_BY_NORMALIZED_NAME.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(normalized)}\b", msg):
            return True
    return False


def _is_followup_phrase(message: str) -> bool:
    msg = _normalize(message)
    return (
        "what about" in msg
        or "how about" in msg
        or re.search(r"\band\b", msg) is not None
        or msg in {"again", "same"}
    )


def _is_context_followup_request(message: str) -> bool:
    msg = _normalize(message)
    if not msg:
        return False
    followup_tokens = [
        "tell me more",
        "more",
        "what else",
        "show more",
        "about this",
        "about it",
        "details",
        "tell me on",
        "tell me about",
    ]
    if any(t in msg for t in followup_tokens):
        return True
    if msg in {"more", "details", "continue", "next"}:
        return True
    return False


def _detect_intent(message: str, state: ConversationState) -> Optional[str]:
    msg = _normalize(message)
    words = msg.split()
    is_short = len(words) <= 3

    product_guess = _detect_product(message, state)
    country_guess = _detect_country(message)
    pack_guess = _detect_pack(message)
    category_guess = _detect_category(message)

    # Strong dietary markers should win.
    if ("difference" in msg or "vs" in msg) and "whey" in msg and "isolate" in msg:
        return "dietary"
    if _contains_dietary_marker(msg):
        return "dietary"

    # Price signals first for transactional questions.
    if _contains_fuzzy_keyword(msg, PRICE_KEYWORDS):
        return "price"
    if _contains_fuzzy_keyword(msg, WHERE_BUY_KEYWORDS):
        return "where_to_buy"
    if category_guess and not _contains_fuzzy_keyword(msg, PRICE_KEYWORDS):
        return "discovery"
    if _is_affirmative(message):
        return "confirm"
    if _contains_fuzzy_keyword(msg, SMALLTALK_KEYWORDS):
        return "smalltalk"
    if _contains_fuzzy_keyword(msg, AUTH_KEYWORDS):
        return "authenticity"
    if _contains_fuzzy_keyword(msg, DIETARY_KEYWORDS, cutoff=0.82):
        return "dietary"
    if _wants_both_packs(message) and (state.pending_intent == "price" or state.selected_product):
        return "price"
    if product_guess and (country_guess or pack_guess):
        return "price"
    if re.search(r"\band\b", msg) and (country_guess or pack_guess):
        return "price"
    if any(k in msg for k in ["what about", "how about"]) and (
        product_guess or country_guess or pack_guess
    ):
        return "price"
    if msg in {"this", "that", "this one", "that one"} and state.pending_intent == "price":
        return "price"
    if msg in {"this", "that", "this one", "that one"} and state.last_priced_product:
        return "price"
    if _contains_fuzzy_keyword(msg, DISCOVERY_KEYWORDS):
        return "discovery"
    if (
        _is_context_followup_request(message)
        and state.selected_category
        and not product_guess
        and not country_guess
        and not pack_guess
        and state.pending_intent != "price"
    ):
        return "discovery"
    if _is_greeting(msg) and (is_short or len(words) <= 6):
        return "greeting"
    if (
        is_short
        and state.selected_category
        and not product_guess
        and not country_guess
        and not pack_guess
        and state.pending_intent != "price"
    ):
        # Keep short follow-ups inside currently selected category browsing.
        return "discovery"
    if is_short and state.pending_intent == "price":
        # Continue active pricing flow for short replies like "hydro", "uae", "2lb".
        return "price"
    if product_guess and state.selected_category and not category_guess:
        return "price"
    return None


def _detect_category(message: str) -> Optional[str]:
    msg = _normalize(message)
    for alias, category in sorted(CATEGORY_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", msg):
            return category

    # Fuzzy category detection for typos like "protien", "vitmins".
    alias_keys = list(CATEGORY_ALIASES.keys())
    for token in msg.split():
        if len(token) < 4:
            continue
        match = difflib.get_close_matches(token, alias_keys, n=1, cutoff=0.82)
        if match:
            return CATEGORY_ALIASES[match[0]]
    return None


def _ai_extract_first(message: str, state: ConversationState) -> dict:
    if not client or AI_DISABLED:
        return {}
    prompt = f"""Extract JSON only. Keys: intent, product, country, pack, category, both_packs, confidence.
message: "{message}"
memory:
- pending_intent: "{state.pending_intent or ''}"
- selected_category: "{state.selected_category or ''}"
- selected_product: "{state.selected_product or ''}"
- selected_pack: "{state.selected_pack or ''}"
- selected_country: "{state.selected_country or ''}"
- last_priced_product: "{state.last_priced_product or ''}"
- last_priced_pack: "{state.last_priced_pack or ''}"
- last_priced_country: "{state.last_priced_country or ''}"

Rules:
- Allowed intents: price, discovery, authenticity, greeting, dietary, where_to_buy, smalltalk, confirm, null.
- country must be one of UAE,KSA,Egypt,null.
- pack must be 2LB/5LB or null.
- both_packs must be true/false.
- confidence is 0.0 to 1.0.
- Use memory for short follow-ups like "this", "that", "and in ksa", "both prices".
- If unsure return null values and lower confidence."""
    try:
        response = _generate_with_fallback(
            prompt,
            {"temperature": 0.0, "response_mime_type": "application/json"},
        )
        return json.loads(response.text) if response and response.text else {}
    except Exception as e:
        logger.error(f"AI extraction failed: {e}")
        return {}


def _validate_ai_entities(ai_entities: dict, state: ConversationState) -> dict:
    allowed_intents = {"price", "discovery", "authenticity", "greeting", "dietary", "where_to_buy", "smalltalk", "confirm", None}
    intent = ai_entities.get("intent")
    if intent not in allowed_intents:
        intent = None

    country = ai_entities.get("country")
    if country not in {"UAE", "KSA", "Egypt"}:
        country = None

    pack = ai_entities.get("pack")
    pack = _to_pack(str(pack)) if pack else None
    if pack and pack not in {"2LB", "5LB"}:
        pack = None

    category = _detect_category(str(ai_entities.get("category") or ""))
    product = _detect_product(str(ai_entities.get("product") or ""), state)
    both_packs = bool(ai_entities.get("both_packs"))

    try:
        confidence = float(ai_entities.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "intent": intent,
        "product": product,
        "country": country,
        "pack": pack,
        "category": category,
        "both_packs": both_packs,
        "confidence": confidence,
    }


def extract_entities(message: str, state: ConversationState) -> dict:
    msg_norm = _normalize(message)
    deterministic = {
        "intent": _detect_intent(message, state),
        "product": _detect_product(message, state),
        "country": _detect_country(message),
        "pack": _detect_pack(message),
        "category": _detect_category(message),
        "both_packs": _wants_both_packs(message),
    }

    ai_raw = _ai_extract_first(message, state)
    ai = _validate_ai_entities(ai_raw, state) if ai_raw else {}
    high_conf = ai.get("confidence", 0.0) >= 0.55

    if high_conf:
        entities = {
            "intent": ai.get("intent") or deterministic["intent"],
            "product": ai.get("product") or deterministic["product"],
            "country": deterministic["country"] or ai.get("country"),
            "pack": deterministic["pack"] or ai.get("pack"),
            "category": ai.get("category") or deterministic["category"],
            "both_packs": ai.get("both_packs") or deterministic["both_packs"],
        }
    else:
        entities = {
            "intent": deterministic["intent"] or ai.get("intent"),
            "product": deterministic["product"] or ai.get("product"),
            "country": deterministic["country"] or ai.get("country"),
            "pack": deterministic["pack"] or ai.get("pack"),
            "category": deterministic["category"] or ai.get("category"),
            "both_packs": deterministic["both_packs"] or ai.get("both_packs"),
        }

    # Guard intent priority: explicit price wording must remain price.
    if _contains_fuzzy_keyword(msg_norm, PRICE_KEYWORDS) and entities.get("intent") != "price":
        entities["intent"] = "price"
    # Hard guard for Whey vs Isolate comparison question.
    if (("difference" in msg_norm or "vs" in msg_norm) and "whey" in msg_norm and "isolate" in msg_norm):
        entities["intent"] = "dietary"

    # Hard guard: category listing requests must stay in discovery flow.
    if _is_category_listing_request(message):
        entities["intent"] = "discovery"
        entities["product"] = None
        entities["country"] = None
        entities["pack"] = None
        entities["both_packs"] = False

    # Hard guard: category-only turns must never drift into product pricing.
    # Example: "I like explore in recovery" should list recovery products.
    if (
        deterministic.get("category")
        and not deterministic.get("product")
        and not deterministic.get("country")
        and not deterministic.get("pack")
        and not _contains_fuzzy_keyword(msg_norm, PRICE_KEYWORDS)
    ):
        entities["intent"] = "discovery"
        entities["product"] = None
        entities["country"] = None
        entities["pack"] = None
        entities["both_packs"] = False

    # If user is continuing category discussion, keep current selected category sticky.
    if (
        entities.get("intent") == "discovery"
        and not entities.get("category")
        and state.selected_category
        and _is_context_followup_request(message)
    ):
        entities["category"] = state.selected_category

    # Continuity guards: in active pricing, do not let AI invent slot switches.
    if state.pending_intent == "price":
        explicit_product = _has_explicit_product_mention(message)
        if not explicit_product and entities.get("product") and state.selected_product:
            entities["product"] = state.selected_product
        if state.last_asked == "pack" and deterministic.get("pack"):
            # User answered pack; country must not be inferred on this turn.
            entities["country"] = deterministic.get("country")
        if state.last_asked == "country" and deterministic.get("country") and state.selected_product:
            entities["intent"] = "price"

    # Country-only follow-up after a recent price should continue pricing.
    if (
        entities.get("country")
        and not _contains_fuzzy_keyword(msg_norm, WHERE_BUY_KEYWORDS)
        and (state.pending_intent == "price" or state.selected_product or state.last_priced_product)
    ):
        entities["intent"] = "price"

    # Pronoun follow-ups should map to the last referenced product.
    if _is_pronoun_reference(message) and not entities.get("product"):
        entities["product"] = (
            state.selected_product
            or state.last_priced_product
            or state.last_product_mentioned
        )

    if any(entities.values()):
        if entities.get("intent") == "price" and not entities.get("product"):
            msg = _normalize(message)
            if msg in {"this", "that", "this one", "that one"}:
                entities["product"] = state.selected_product or state.last_priced_product or state.last_product_mentioned
                entities["pack"] = entities.get("pack") or state.selected_pack or state.last_priced_pack or state.last_pack_mentioned
        return entities
    return deterministic


def _grounded_reply(facts: str, style_instruction: str = "") -> str:
    if not client:
        return facts
    style_hint = (
        "Friendly and natural tone. Keep it clear and helpful."
        if RESPONSE_STYLE == "friendly_concise"
        else "Answer naturally and clearly."
    )
    system = (
        "You are Optimum Nutrition Instagram DM assistant. "
        "Use ONLY provided facts. Do not add any new data. "
        "Prefer concise replies, but use up to 4 sentences when needed for clarity. "
        "Ask only one missing detail when needed."
    )
    prompt = f"Facts:\n{facts}\n\nTask:\n{style_instruction or style_hint}"
    try:
        response = _generate_with_fallback(
            prompt,
            {"system_instruction": system, "temperature": 0.2, "max_output_tokens": 220},
        )
        text = response.text.strip() if response and response.text else ""
        return text or facts
    except Exception as e:
        logger.error(f"Grounded generation failed: {e}")
        return facts


def _generate_with_fallback(contents: str, config: dict):
    global WORKING_MODEL
    global AI_DISABLED
    global AI_DISABLED_UNTIL
    if not OPENAI_API_KEY and not client:
        return None
    if AI_DISABLED and time.time() < AI_DISABLED_UNTIL:
        return None
    if AI_DISABLED and time.time() >= AI_DISABLED_UNTIL:
        AI_DISABLED = False

    if OPENAI_API_KEY:
        try:
            response = _generate_with_openai(contents, config)
            if response:
                return response
        except Exception as e:
            err = str(e)
            if "429" in err:
                AI_DISABLED = True
                AI_DISABLED_UNTIL = time.time() + AI_RATE_LIMIT_COOLDOWN_SECONDS
                logger.warning("OpenAI quota exhausted; temporary cooldown enabled for AI calls.")
                return None
            logger.error(f"OpenAI call failed: {e}")

    if not client:
        return None
    models = [WORKING_MODEL] + MODEL_CANDIDATES if WORKING_MODEL else MODEL_CANDIDATES
    tried = set()
    for model in models:
        if not model or model in tried:
            continue
        tried.add(model)
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            if response:
                WORKING_MODEL = model
                return response
        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err or "429" in err:
                AI_DISABLED = True
                AI_DISABLED_UNTIL = time.time() + AI_RATE_LIMIT_COOLDOWN_SECONDS
                logger.warning("Gemini quota exhausted; temporary cooldown enabled for AI calls.")
                return None
            if "NOT_FOUND" in err:
                continue
            logger.error(f"Model call failed on {model}: {e}")
            continue
    return None


def _retrieve_kb_sections(message: str, state: ConversationState, limit: int = 3) -> Dict[str, str]:
    context_seed = " ".join(
        [
            message or "",
            state.selected_category or "",
            state.selected_product or "",
            state.last_product_mentioned or "",
            state.pending_intent or "",
            state.last_asked or "",
        ]
    )
    msg_tokens = set([t for t in _normalize(context_seed).split() if len(t) > 2])
    scored = []
    for name, body in KB_SECTIONS.items():
        section_tokens = set([t for t in _normalize(name + " " + body).split() if len(t) > 2])
        score = len(msg_tokens.intersection(section_tokens))
        name_norm = _normalize(name)
        if state.pending_intent == "price" and "pricing" in name_norm:
            score += 2
        if state.selected_category and state.selected_category in name_norm:
            score += 2
        if state.selected_product and _normalize(state.selected_product) in _normalize(body):
            score += 2
        if score > 0:
            scored.append((name, body, score))
    scored.sort(key=lambda x: x[2], reverse=True)
    selected = scored[:limit]
    return {name: body for name, body, _ in selected}


def _ai_rag_reply(message: str, state: ConversationState, fallback: str) -> str:
    if not client:
        return fallback

    sections = _retrieve_kb_sections(message, state, limit=3)
    if not sections:
        return fallback
    section_text = "\n\n".join([f"[{k}]\n{v}" for k, v in sections.items()])
    history_hint = ", ".join(CONVERSATION_HISTORY.get(state.user_id, [])[-4:])
    prompt = (
        f"User message: {message}\n"
        f"Recent conversation: {history_hint}\n"
        f"Available KB sections:\n{section_text}\n\n"
        "Answer naturally using ONLY these sections. "
        "Use detail when needed (up to 4 sentences). "
        "If pricing is requested but missing required fields, ask only one missing field."
    )
    response = _generate_with_fallback(
        prompt,
        {
            "system_instruction": (
                "You are Optimum Nutrition assistant. Do not invent products, prices, or policies."
            ),
            "temperature": 0.15,
            "max_output_tokens": 120,
        },
    )
    if response and response.text:
        return response.text.strip()
    return fallback


def _ask_for_missing(state: ConversationState, missing_field: str) -> str:
    def pick(options):
        idx = len(CONVERSATION_HISTORY.get(state.user_id, [])) % max(len(options), 1)
        return options[idx]

    repeat = state.last_asked == missing_field
    state.last_asked = missing_field

    if missing_field == "product":
        if state.selected_category in CATEGORY_PRODUCTS:
            items = ", ".join(CATEGORY_PRODUCTS[state.selected_category])
            facts = pick([
                f"Please pick one product from {state.selected_category}: {items}. Send the product name and I’ll continue pricing.",
                f"From {state.selected_category}, choose one: {items}. I’ll continue right after that.",
            ])
            state.last_asked = "product_from_category"
            return _grounded_reply(facts)

        facts = pick([
            "I just need the product name to continue. You can send Gold Standard 100% Whey, Serious Mass, or Platinum HydroWhey.",
            "Please share the product name so I can continue. For example: Gold Standard 100% Whey, Serious Mass, or Platinum HydroWhey.",
        ])
        if not repeat:
            facts = pick([
                "Which product price do you want? You can choose Gold Standard 100% Whey, Serious Mass, or Platinum HydroWhey.",
                "Which product price should I check? Options: Gold Standard 100% Whey, Serious Mass, Platinum HydroWhey.",
            ])
        return _grounded_reply(facts)

    if missing_field == "pack":
        facts = pick([
            f"I need the pack size for {state.selected_product}: 2LB or 5LB.",
            f"For {state.selected_product}, which pack do you want: 2LB or 5LB?",
        ])
        if not repeat:
            facts = pick([
                f"For {state.selected_product}, which pack do you want: 2LB or 5LB?",
                f"Got it. Choose the pack for {state.selected_product}: 2LB or 5LB.",
            ])
        return _grounded_reply(facts)

    facts = pick([
        "I need your country to give exact pricing: UAE, KSA, or Egypt.",
        "To give the exact price, tell me your country: UAE, KSA, or Egypt.",
    ])
    if not repeat:
        facts = pick([
            "To give the exact price, which country are you in: UAE, KSA, or Egypt?",
            "Which country should I check pricing for: UAE, KSA, or Egypt?",
        ])
    return _grounded_reply(facts)


def _needs_pack(product: str) -> bool:
    product_data = PRICING_DATA.get(product, {})
    return len(product_data.get("packs", {})) > 1


def _resolve_price(product: str, country: str, pack: Optional[str]) -> dict:
    product_data = PRICING_DATA.get(product, {})
    packs = product_data.get("packs", {})
    prices = product_data.get("prices", {})
    note = product_data.get("note")

    if packs:
        if not pack:
            if len(packs) == 1:
                pack = next(iter(packs.keys()))
            else:
                return {"status": "missing_pack"}
        if pack not in packs:
            return {"status": "unknown_pack"}
        country_prices = packs[pack]
        if country not in country_prices:
            return {"status": "missing_country_price", "note": note}
        return {"status": "ok", "value": country_prices[country], "pack": pack, "note": note}

    if country not in prices:
        return {"status": "missing_country_price", "note": note}
    return {"status": "ok", "value": prices[country], "pack": None, "note": note}


def _handle_discovery(state: ConversationState) -> str:
    def pick(options):
        idx = len(CONVERSATION_HISTORY.get(state.user_id, [])) % max(len(options), 1)
        return options[idx]

    category = state.selected_category
    if category in CATEGORY_PRODUCTS:
        items = ", ".join(CATEGORY_PRODUCTS[category])
        facts = pick([
            f"Our {category} products are: {items}. Want details or price for one of these?",
            f"Our {category} products are: {items}. Which one should I help with?",
            f"Our {category} products are: {items}. Send one product name and I’ll continue.",
        ])
        state.last_asked = f"discovery_{category}"
        return _grounded_reply(facts)

    facts = pick([
        "Our main categories are Protein, Energy & Aminos, Pre-Workout, Recovery, and Vitamins/Health. Which category do you want?",
        "Our main categories are Protein, Energy & Aminos, Pre-Workout, Recovery, and Vitamins/Health. Which one should I open for you?",
        "Our main categories are Protein, Energy & Aminos, Pre-Workout, Recovery, and Vitamins/Health. Which category are you interested in?",
    ])
    if state.last_asked == "discovery_generic":
        facts = pick([
            "Please pick one category: Protein, Energy & Aminos, Pre-Workout, Recovery, or Vitamins/Health.",
            "Share one category to continue: Protein, Energy & Aminos, Pre-Workout, Recovery, or Vitamins/Health.",
        ])
    state.last_asked = "discovery_generic"
    return _grounded_reply(facts)


def _handle_authenticity() -> str:
    facts = (
        "Authentic ON products have an authenticity sticker with a code you can verify at originalon.com. "
        "If the sticker is missing, authenticity cannot be guaranteed."
    )
    return _grounded_reply(facts)


def _handle_dietary() -> str:
    facts = (
        "Gold Standard 100% Whey is certified gluten-free except Cookies and Cream flavor. "
        "We do not currently offer vegan protein."
    )
    return _grounded_reply(facts)


def _handle_whey_vs_isolate() -> str:
    facts = (
        "The Gold Standard Isolate offers much lower fat levels and higher protein levels in each serving, "
        "it's ultra-filtered Whey Protein."
    )
    return _grounded_reply(facts)


def _handle_where_to_buy() -> str:
    facts = (
        "UAE & KSA: available at www.sporter.com, www.amazon.ae, and Dr. Nutrition; also Life Pharmacies in UAE. "
        "Egypt: available through iFIT at www.ifit-eg.com and at El Ezaby, Khalil, Max Muscle, and Bodybuilding House."
    )
    return _grounded_reply(facts)


def _handle_smalltalk(message: str, state: ConversationState) -> str:
    msg = _normalize(message)
    if "good morning" in msg or "morning" in msg:
        fallback = "Good morning. Tell me the product or price you want and I will help right away."
    elif "good evening" in msg or "evening" in msg:
        fallback = "Good evening. Tell me the product or price you want and I will help right away."
    elif _is_greeting(msg):
        fallback = "Hello, welcome to Optimum Nutrition support. Tell me a product or ask for a price and I will help."
    elif "thank" in msg:
        fallback = "You are welcome. If you want, I can check another product price now."
    else:
        fallback = "Sure. Tell me what product or price details you want next."
    return _ai_rag_reply(message, state, fallback)


def _handle_product_info(product: str, state: ConversationState) -> str:
    # If user asks for product info, keep pricing continuation open for country follow-up.
    state.pending_intent = "price"
    state.mode = "transaction"
    state.selected_product = product
    state.last_product_mentioned = product
    state.last_asked = "country"
    category = PRODUCT_TO_CATEGORY.get(product, "products")
    pretty_category = "Vitamins/Health" if category == "vitamins" else category.title()
    facts = (
        f"{product} is in our {pretty_category} range. "
        "If you want, I can share price for UAE, KSA, or Egypt."
    )
    return _grounded_reply(facts)


def _handle_confirm(state: ConversationState) -> str:
    if state.last_priced_product:
        state.pending_intent = "price"
        state.mode = "transaction"
        state.selected_product = state.last_priced_product
        state.selected_pack = state.last_priced_pack
        state.selected_country = None
        previous = state.last_priced_country
        options = [c for c in ["UAE", "KSA", "Egypt"] if c != previous]
        if options:
            facts = f"Sure. Which country should I check next for {state.selected_product}: {', '.join(options)}?"
            state.last_asked = "country"
            return _grounded_reply(facts)
    return _handle_smalltalk("ok", state)


def _handle_unknown_query() -> str:
    facts = (
        "I may not have that specific information in this chat yet. "
        "For the latest details, please visit www.sporter.com (UAE/KSA) or www.ifit-eg.com (Egypt)."
    )
    return _grounded_reply(facts)


def _should_clear_pricing_context(intent: Optional[str]) -> bool:
    return intent in {"authenticity", "dietary", "discovery", "where_to_buy"}


def _clear_pricing_state(state: ConversationState) -> None:
    state.pending_intent = None
    state.selected_product = None
    state.selected_pack = None
    state.selected_country = None
    state.last_asked = None
    state.requested_both_packs = False
    state.mode = "discovery"


def _reset_session_state(state: ConversationState) -> None:
    state.mode = "discovery"
    state.pending_intent = None
    state.selected_category = None
    state.selected_product = None
    state.selected_pack = None
    state.selected_country = None
    state.last_asked = None
    state.last_product_mentioned = None
    state.last_pack_mentioned = None
    state.last_country_mentioned = None
    state.last_priced_product = None
    state.last_priced_pack = None
    state.last_priced_country = None
    state.requested_both_packs = False


def get_on_ai_response(message: str, user_id: str = "default") -> str:
    state, history = _load_user_context(user_id)

    def finish(text: str) -> str:
        safe = _safety_guard_response(message, state, entities, text)
        safe = _semantic_guard_response(message, state, entities, safe)
        _save_user_context(user_id, state, history)
        return safe

    history.append(message)
    if len(history) > 10:
        history = history[-10:]
        CONVERSATION_HISTORY[user_id] = history

    entities = extract_entities(message, state)
    logger.info(f"User {user_id} | State: {state.mode} | Slots: {entities}")

    explicit_product_mention = _has_explicit_product_mention(message)
    if entities.get("product"):
        previous_product = state.selected_product
        should_apply_product = (
            explicit_product_mention
            or not previous_product
            or state.pending_intent == "price"
            and previous_product is None
        )
        # Prevent AI-inferred product drift from replacing an explicit active product context.
        if should_apply_product:
            state.selected_product = entities["product"]
            if explicit_product_mention:
                state.last_product_mentioned = entities["product"]
            # Do not reset pack unless user explicitly switched product in this message.
            if explicit_product_mention and previous_product and entities["product"] != previous_product:
                state.selected_pack = None
    if entities.get("pack"):
        state.selected_pack = entities["pack"]
        state.last_pack_mentioned = entities["pack"]
    if entities.get("both_packs"):
        state.requested_both_packs = True
        if state.selected_product and _needs_pack(state.selected_product):
            state.selected_pack = "__BOTH__"
    if entities.get("country"):
        state.selected_country = entities["country"]
        state.last_country_mentioned = entities["country"]
    if entities.get("category") and not _is_pronoun_reference(message):
        state.selected_category = entities["category"]

    intent = entities.get("intent")
    if entities.get("country") and state.selected_product and intent not in {"authenticity", "dietary", "where_to_buy"}:
        intent = "price"
        entities["intent"] = "price"
    if (
        entities.get("country")
        and not entities.get("product")
        and not state.selected_product
        and state.selected_category in CATEGORY_PRODUCTS
        and len(CATEGORY_PRODUCTS[state.selected_category]) == 1
    ):
        state.selected_product = CATEGORY_PRODUCTS[state.selected_category][0]
        state.last_product_mentioned = state.selected_product
        intent = "price"
        entities["intent"] = "price"
    if intent == "discovery" and _is_broad_listing_request(message):
        state.selected_category = None

    # Guardrail: during an active pricing flow, short slot-only replies must stay in price intent
    # even if AI-first intent classification drifts.
    if state.pending_intent == "price":
        msg_norm = _normalize(message)
        if state.last_asked == "pack" and "both" in msg_norm:
            entities["both_packs"] = True
            state.requested_both_packs = True
            if state.selected_product and _needs_pack(state.selected_product):
                state.selected_pack = "__BOTH__"
        explicit_non_price = (
            intent in {"dietary", "authenticity", "where_to_buy"}
            or "gluten" in msg_norm
            or "vegan" in msg_norm
            or (("difference" in msg_norm or "vs" in msg_norm) and "whey" in msg_norm and "isolate" in msg_norm)
            or "authentic" in msg_norm
            or "original" in msg_norm
            or "sticker" in msg_norm
            or "where" in msg_norm and ("buy" in msg_norm or "find" in msg_norm)
        )
        expected_slot_filled = (
            (state.last_asked == "country" and entities.get("country"))
            or (state.last_asked == "pack" and (entities.get("pack") or entities.get("both_packs")))
            or (state.last_asked == "product" and entities.get("product"))
        )
        continuity_hint = (
            state.last_asked in {"pack", "country", "product"}
            and (
                entities.get("both_packs")
                or entities.get("country")
                or entities.get("pack")
                or "both" in msg_norm
            )
        )
        if (
            not explicit_non_price
            and (
            expected_slot_filled
            or continuity_hint
            or
            (
            entities.get("country")
            or entities.get("pack")
            or entities.get("product")
            or entities.get("both_packs")
            or msg_norm in {"this", "that", "this one", "that one"}
            )
            )
        ):
            intent = "price"
            entities["intent"] = "price"

    if _should_clear_pricing_context(intent):
        _clear_pricing_state(state)

    if intent == "price" or state.pending_intent == "price":
        state.mode = "transaction"
        state.pending_intent = "price"

        if not state.selected_product and state.last_priced_product:
            state.selected_product = state.last_priced_product
        if not state.selected_product and state.last_product_mentioned:
            state.selected_product = state.last_product_mentioned
        if (
            not state.selected_pack
            and state.selected_product == state.last_priced_product
            and state.last_priced_pack
            and _is_followup_phrase(message)
        ):
            state.selected_pack = state.last_priced_pack
        if (
            not state.selected_country
            and state.last_priced_country
            and state.selected_product == state.last_priced_product
            and (entities.get("pack") or entities.get("both_packs"))
            and (_is_followup_phrase(message) or entities.get("pack"))
        ):
            state.selected_country = state.last_priced_country

        if not state.selected_product:
            return finish(_ask_for_missing(state, "product"))

        if state.requested_both_packs and _needs_pack(state.selected_product):
            if not state.selected_country:
                return finish(_ask_for_missing(state, "country"))
            product_data = PRICING_DATA.get(state.selected_product, {})
            packs = product_data.get("packs", {})
            p2 = packs.get("2LB", {}).get(state.selected_country)
            p5 = packs.get("5LB", {}).get(state.selected_country)
            if p2 and p5:
                facts = (
                    f"{state.selected_product} prices in {state.selected_country}: "
                    f"2LB is {p2} and 5LB is {p5}. "
                    "If you want, I can also share another country."
                )
                state.last_priced_product = state.selected_product
                state.last_priced_pack = None
                state.last_priced_country = state.selected_country
                _clear_pricing_state(state)
                return finish(_grounded_reply(facts))

        if _needs_pack(state.selected_product) and not state.selected_pack:
            return finish(_ask_for_missing(state, "pack"))

        if not state.selected_country:
            return finish(_ask_for_missing(state, "country"))

        resolved = _resolve_price(state.selected_product, state.selected_country, state.selected_pack)
        if resolved["status"] == "missing_pack":
            return finish(_ask_for_missing(state, "pack"))
        if resolved["status"] == "unknown_pack":
            facts = f"{state.selected_product} pack was not found in our KB. Please choose 2LB or 5LB."
            return finish(_grounded_reply(facts))
        if resolved["status"] == "missing_country_price":
            note = resolved.get("note")
            if note:
                facts = (
                    f"I do not have a {state.selected_country} price for {state.selected_product} in the KB. "
                    f"Please check www.sporter.com or www.ifit-eg.com for the latest exact price."
                )
            else:
                facts = (
                    f"I do not have a {state.selected_country} price in the KB for {state.selected_product}. "
                    "Share another country (UAE, KSA, or Egypt) and I will check."
                )
            _clear_pricing_state(state)
            return finish(_grounded_reply(facts))

        pack_text = f" ({resolved['pack']})" if resolved.get("pack") else ""
        facts = (
            f"{state.selected_product}{pack_text} price in {state.selected_country} is {resolved['value']}. "
            "If you want, I can also share prices for another country."
        )
        state.last_priced_product = state.selected_product
        state.last_priced_pack = resolved.get("pack")
        state.last_priced_country = state.selected_country
        _clear_pricing_state(state)
        return finish(_grounded_reply(facts))

    if intent == "authenticity":
        return finish(_handle_authenticity())
    if intent == "dietary":
        msg_norm = _normalize(message)
        if ("difference" in msg_norm or "vs" in msg_norm) and "whey" in msg_norm and "isolate" in msg_norm:
            return finish(_handle_whey_vs_isolate())
        return finish(_handle_dietary())
    if intent == "where_to_buy":
        return finish(_handle_where_to_buy())
    if intent == "smalltalk":
        return finish(_handle_smalltalk(message, state))
    if intent == "confirm":
        return finish(_handle_confirm(state))
    if (
        entities.get("product")
        and intent not in {"price", "authenticity", "dietary", "where_to_buy"}
        and not _contains_fuzzy_keyword(_normalize(message), PRICE_KEYWORDS)
    ):
        return finish(_handle_product_info(entities["product"], state))
    if intent == "discovery":
        return finish(_handle_discovery(state))

    if intent == "greeting":
        # Demo behavior: pure greeting starts a fresh session context.
        if _is_greeting(_normalize(message)) and len(_normalize(message).split()) <= 4:
            _reset_session_state(state)
            history.clear()
            history.append(message)
            return finish(_handle_smalltalk(message, state))
        if state.pending_intent == "price":
            # Treat a pure greeting as a fresh turn and drop stale pricing slots.
            if not any(
                [entities.get("product"), entities.get("country"), entities.get("pack"), entities.get("category")]
            ):
                _clear_pricing_state(state)
                return finish(_handle_smalltalk(message, state))
            if not state.selected_product:
                return finish(_grounded_reply(
                    "Hi. To continue pricing, tell me the product name."
                ))
            if _needs_pack(state.selected_product) and not state.selected_pack:
                return finish(_grounded_reply(
                    f"Hi. To continue pricing for {state.selected_product}, tell me the pack size: 2LB or 5LB."
                ))
            if not state.selected_country:
                return finish(_grounded_reply(
                    "Hi. To continue pricing, tell me your country: UAE, KSA, or Egypt."
                ))
        return finish(_handle_smalltalk(message, state))

    if not intent and not any(
        [entities.get("product"), entities.get("country"), entities.get("pack"), entities.get("category")]
    ):
        return finish(_handle_unknown_query())

    return finish(_ai_rag_reply(message, state, _handle_discovery(state)))
