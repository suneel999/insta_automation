"""
Instagram AI Handler for Optimum Nutrition (ON)
Retrieval-grounded, memory-aware, and strict about using only KB facts.
"""

import os
import re
import json
import logging
import difflib
from typing import Dict, Optional
from google import genai
from dotenv import load_dotenv
from on_knowledge import ON_KNOWLEDGE_BASE

# Load local env
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logger = logging.getLogger(__name__)

# --- MEMORY STORE ---
USER_STATES = {}  # { user_id: ConversationState }
CONVERSATION_HISTORY = {}  # { user_id: [str] }

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


# Configure Gemini Client (optional for phrasing)
api_key = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=api_key) if api_key else None
MODEL_CANDIDATES = [
    os.getenv("GEMINI_MODEL", "").strip(),
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]
MODEL_CANDIDATES = [m for m in MODEL_CANDIDATES if m]
WORKING_MODEL = None
AI_DISABLED = False


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


PRICING_DATA = _parse_pricing_kb(ON_KNOWLEDGE_BASE)
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
    if words.intersection({"hi", "hello", "hey", "hii", "heyy", "heyyy"}):
        return True
    return False


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


def _detect_intent(message: str, state: ConversationState) -> Optional[str]:
    msg = _normalize(message)
    words = msg.split()
    is_short = len(words) <= 3

    product_guess = _detect_product(message, state)
    country_guess = _detect_country(message)
    pack_guess = _detect_pack(message)

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
    if _is_greeting(msg) and (is_short or len(words) <= 6):
        return "greeting"
    if is_short and state.pending_intent == "price":
        # Continue active pricing flow for short replies like "hydro", "uae", "2lb".
        return "price"
    if product_guess and (state.selected_category or (state.last_asked or "").startswith("discovery_")):
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
        match = difflib.get_close_matches(token, alias_keys, n=1, cutoff=0.76)
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
- Allowed intents: price, discovery, authenticity, greeting, dietary, null.
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
    allowed_intents = {"price", "discovery", "authenticity", "greeting", "dietary", "where_to_buy", None}
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
            "country": ai.get("country") or deterministic["country"],
            "pack": ai.get("pack") or deterministic["pack"],
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
    system = (
        "You are Optimum Nutrition Instagram DM assistant. "
        "Use ONLY provided facts. Do not add any new data. "
        "Max 2 sentences. Ask only one missing detail when needed."
    )
    prompt = f"Facts:\n{facts}\n\nTask:\n{style_instruction or 'Answer naturally and clearly.'}"
    try:
        response = _generate_with_fallback(
            prompt,
            {"system_instruction": system, "temperature": 0.1, "max_output_tokens": 120},
        )
        text = response.text.strip() if response and response.text else ""
        return text or facts
    except Exception as e:
        logger.error(f"Grounded generation failed: {e}")
        return facts


def _generate_with_fallback(contents: str, config: dict):
    global WORKING_MODEL
    global AI_DISABLED
    if not client or AI_DISABLED:
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
                logger.warning("Gemini quota exhausted; switching to deterministic KB replies.")
                return None
            if "NOT_FOUND" in err:
                continue
            logger.error(f"Model call failed on {model}: {e}")
            continue
    return None


def _ask_for_missing(state: ConversationState, missing_field: str) -> str:
    repeat = state.last_asked == missing_field
    state.last_asked = missing_field

    if missing_field == "product":
        if state.selected_category in CATEGORY_PRODUCTS:
            items = ", ".join(CATEGORY_PRODUCTS[state.selected_category])
            facts = (
                f"Please pick one product from {state.selected_category}: {items}. "
                "Send the product name and I will continue pricing."
            )
            state.last_asked = "product_from_category"
            return _grounded_reply(facts)

        facts = (
            "I need the product name to continue. "
            "You can send one: Gold Standard 100% Whey, Serious Mass, or Platinum HydroWhey."
        )
        if not repeat:
            facts = (
                "Which product price do you want? "
                "You can choose Gold Standard 100% Whey, Serious Mass, or Platinum HydroWhey."
            )
        return _grounded_reply(facts)

    if missing_field == "pack":
        facts = f"I need the pack size for {state.selected_product}: 2LB or 5LB."
        if not repeat:
            facts = f"For {state.selected_product}, which pack do you want: 2LB or 5LB?"
        return _grounded_reply(facts)

    facts = "I need your country to give exact pricing: UAE, KSA, or Egypt."
    if not repeat:
        facts = "To give the exact price, which country are you in: UAE, KSA, or Egypt?"
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
    category = state.selected_category
    if category in CATEGORY_PRODUCTS:
        items = ", ".join(CATEGORY_PRODUCTS[category])
        facts = (
            f"Our {category} products are: {items}. "
            "Tell me one product name if you want details or price."
        )
        state.last_asked = f"discovery_{category}"
        return _grounded_reply(facts)

    facts = (
        "Our main categories are Protein, Energy & Aminos, Pre-Workout, Recovery, and Vitamins/Health. "
        "Which category do you want to see products for?"
    )
    if state.last_asked == "discovery_generic":
        facts = "Please choose one category: Protein, Energy & Aminos, Pre-Workout, Recovery, or Vitamins/Health."
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
        "Gold Standard Isolate has lower fat and higher protein per serving than Gold Standard Whey. "
        "Gold Standard Whey is certified gluten-free except Cookies and Cream flavor."
    )
    return _grounded_reply(facts)


def _handle_where_to_buy() -> str:
    facts = (
        "UAE & KSA: available at www.sporter.com, www.amazon.ae, and Dr. Nutrition; also Life Pharmacies in UAE. "
        "Egypt: available through iFIT at www.ifit-eg.com and at El Ezaby, Khalil, Max Muscle, and Bodybuilding House."
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


def get_on_ai_response(message: str, user_id: str = "default") -> str:
    if user_id not in USER_STATES:
        USER_STATES[user_id] = ConversationState(user_id)
        CONVERSATION_HISTORY[user_id] = []

    state = USER_STATES[user_id]
    history = CONVERSATION_HISTORY[user_id]
    history.append(message)
    if len(history) > 10:
        CONVERSATION_HISTORY[user_id] = history[-10:]

    entities = extract_entities(message, state)
    logger.info(f"User {user_id} | State: {state.mode} | Slots: {entities}")

    if entities.get("product"):
        state.selected_product = entities["product"]
        state.last_product_mentioned = entities["product"]
        if entities["product"] != state.last_priced_product:
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
    if entities.get("category"):
        state.selected_category = entities["category"]

    intent = entities.get("intent")

    # Guardrail: during an active pricing flow, short slot-only replies must stay in price intent
    # even if AI-first intent classification drifts.
    if state.pending_intent == "price":
        msg_norm = _normalize(message)
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
        if (
            not explicit_non_price
            and intent not in {"discovery"}
            and (
            entities.get("country")
            or entities.get("pack")
            or entities.get("product")
            or entities.get("both_packs")
            or msg_norm in {"this", "that", "this one", "that one"}
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
        if not state.selected_pack and state.selected_product == state.last_priced_product and state.last_priced_pack:
            state.selected_pack = state.last_priced_pack
        if (
            not state.selected_country
            and state.last_priced_country
            and state.selected_product == state.last_priced_product
            and (entities.get("pack") or entities.get("both_packs"))
        ):
            state.selected_country = state.last_priced_country

        if not state.selected_product:
            return _ask_for_missing(state, "product")

        if state.requested_both_packs and _needs_pack(state.selected_product):
            if not state.selected_country:
                return _ask_for_missing(state, "country")
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
                return _grounded_reply(facts)

        if _needs_pack(state.selected_product) and not state.selected_pack:
            return _ask_for_missing(state, "pack")

        if not state.selected_country:
            return _ask_for_missing(state, "country")

        resolved = _resolve_price(state.selected_product, state.selected_country, state.selected_pack)
        if resolved["status"] == "missing_pack":
            return _ask_for_missing(state, "pack")
        if resolved["status"] == "unknown_pack":
            facts = f"{state.selected_product} pack was not found in our KB. Please choose 2LB or 5LB."
            return _grounded_reply(facts)
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
            return _grounded_reply(facts)

        pack_text = f" ({resolved['pack']})" if resolved.get("pack") else ""
        facts = (
            f"{state.selected_product}{pack_text} price in {state.selected_country} is {resolved['value']}. "
            "If you want, I can also share prices for another country."
        )
        state.last_priced_product = state.selected_product
        state.last_priced_pack = resolved.get("pack")
        state.last_priced_country = state.selected_country
        _clear_pricing_state(state)
        return _grounded_reply(facts)

    if intent == "authenticity":
        return _handle_authenticity()
    if intent == "dietary":
        msg_norm = _normalize(message)
        if ("difference" in msg_norm or "vs" in msg_norm) and "whey" in msg_norm and "isolate" in msg_norm:
            return _handle_whey_vs_isolate()
        return _handle_dietary()
    if intent == "where_to_buy":
        return _handle_where_to_buy()
    if intent == "discovery":
        return _handle_discovery(state)

    if intent == "greeting":
        if state.pending_intent == "price":
            if not state.selected_product:
                return _grounded_reply(
                    "Hi. To continue pricing, tell me the product name."
                )
            if _needs_pack(state.selected_product) and not state.selected_pack:
                return _grounded_reply(
                    f"Hi. To continue pricing for {state.selected_product}, tell me the pack size: 2LB or 5LB."
                )
            if not state.selected_country:
                return _grounded_reply(
                    "Hi. To continue pricing, tell me your country: UAE, KSA, or Egypt."
                )
        return _grounded_reply(
            "Hello, welcome to Optimum Nutrition support. Tell me a product or ask for a price and I will help."
        )

    return _handle_discovery(state)
