"""
Instagram AI Handler for Optimum Nutrition (ON) - ROBUST PRODUCTION VERSION
Features: Deterministic State Machine, Slot Filling, and Anti-Truncation Logic.
"""

import os
import logging
import hashlib
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from on_knowledge import ON_KNOWLEDGE_BASE

# Load local env
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

logger = logging.getLogger(__name__)

# --- MEMORY & CACHE ---
USER_STATES = {}  # { user_id: state_dict }
CONVERSATION_HISTORY = {} # { user_id: [history] }

# Map categories to products
CATEGORY_MAP = {
    "protein": ["Gold Standard 100% Whey", "Gold Standard Isolate", "Platinum HydroWhey"],
    "gainer": ["Serious Mass"],
    "energy": ["Essential Amin.O. Energy"],
    "pre_workout": ["Gold Standard Pre-Workout"],
    "recovery": ["Gold Standard 100% Casein", "Glutamine Powder", "BCAA 5000"],
    "vitamins": ["Opti-Men", "Opti-Women", "Fish Oil Softgels", "Micronized Creatine Powder"]
}

# Configure Gemini Client
api_key = os.getenv('GEMINI_API_KEY', '')
client = None
if api_key:
    client = genai.Client(api_key=api_key)

class ConversationState:
    def __init__(self, user_id):
        self.user_id = user_id
        self.pending_intent = None
        self.selected_category = None
        self.selected_product = None
        self.selected_country = None

def extract_entities(message: str) -> dict:
    """
    Use AI to extract structured categories and intent from the message.
    """
    prompt = f"""Extract fields from this message into JSON:
- intent: ("price", "authenticity", "where_to_buy", "dietary", "greeting", or null)
- product: (Full name from KB or null)
- country: ("UAE", "KSA", "Egypt", or null)

Rules:
1. If user asks "how much", "cost", "price", intent is "price".
2. If user mentions "Dubai", "UAE", "KSA", "Saudi", "Egypt", set country.
3. If user says "hi", "hello", intent is "greeting".

Message: "{message}"
JSON:"""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={"temperature": 0.0, "response_mime_type": "application/json"}
        )
        if response and response.text:
            return json.loads(response.text)
    except Exception as e:
        logger.error(f"Extraction Error: {e}")
    return {}

def get_on_ai_response(message: str, user_id: str = "default") -> str:
    if not client:
        return "Please check our official website for the most updated information."

    # 1. Init State/History
    if user_id not in USER_STATES:
        USER_STATES[user_id] = ConversationState(user_id)
        CONVERSATION_HISTORY[user_id] = []
    
    state = USER_STATES[user_id]
    hist = CONVERSATION_HISTORY[user_id]

    # 2. Extract
    entities = extract_entities(message)
    logger.info(f"Entities for {user_id}: {entities}")

    # 3. Handle Logic
    if entities.get('intent'): state.pending_intent = entities['intent']
    if entities.get('product'): state.selected_product = entities['product']
    if entities.get('country'): state.selected_country = entities['country']

    # Determine Instruction
    instruction = ""
    if state.pending_intent == "greeting":
        instruction = "Greet the user warmly using the WELCOME section. Ask how you can help."
        state.pending_intent = None
    elif state.pending_intent == "price":
        if not state.selected_product:
            instruction = "The user wants pricing but hasn't picked a product. Ask: 'Which product would you like to know the price of?' and list 3 popular options from the KB."
        elif not state.selected_country:
            instruction = f"User wants price for {state.selected_product}. We need the country. Ask: 'To give you the correct price for {state.selected_product}, which country are you in: UAE, KSA, or Egypt?'"
        else:
            instruction = f"Provide the exact price for {state.selected_product} in {state.selected_country} from the PRICING section. Do NOT ask more questions. Use full sentences."
            state.pending_intent = None # Fulfilled
    else:
        # Default Overview
        instruction = "Use the PRODUCT OVERVIEW to list our main categories (Protein, Energy, Vitamins). Ask which one they are interested in. Do NOT repeat the greeting if already done."

    # 4. Generate Final Response
    system_instruction = f"""You are the official Optimum Nutrition support.
TASK: {instruction}

STRICT RULES:
- ALWAYS COMPLETE YOUR SENTENCES. Do not cut off early.
- Use ONLY the Knowledge Base below.
- Do not greet twice.
- Max 2 sentences.

KNOWLEDGE BASE:
{ON_KNOWLEDGE_BASE}
"""

    try:
        # Update history window
        hist.append(types.Content(role="user", parts=[types.Part(text=message)]))
        if len(hist) > 6: CONVERSATION_HISTORY[user_id] = hist[-6:]

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=hist,
            config={
                "system_instruction": system_instruction,
                "temperature": 0.1,
                "max_output_tokens": 400
            }
        )
        
        if response and response.text:
            ai_text = response.text.strip()
            # Safety log
            logger.info(f"AI Final Output: {ai_text}")
            hist.append(types.Content(role="model", parts=[types.Part(text=ai_text)]))
            return ai_text
            
    except Exception as e:
        logger.error(f"Gen Error: {e}")
        
    return "Please check our official website for the most updated information."
