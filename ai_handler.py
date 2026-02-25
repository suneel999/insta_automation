"""
Instagram AI Handler for Optimum Nutrition (ON) - PRODUCTION GRADE STATE MACHINE
Uses a deterministic backend state machine for multi-turn conversations and Slot Filling.
AI is used as an Entity Extractor and Response Formatter.
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

# --- CONFIG & MAPPINGS ---
RESPONSE_CACHE = {}
USER_STATES = {}  # { user_id: state_dict }

# Mapping categories to products in the KB
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
        self.selected_pack = None
        self.selected_country = None
        self.last_question = None

    def to_dict(self):
        return vars(self)

    def is_price_ready(self):
        # Pricing requires: Product and Country
        # (Pack is optional if only one available or user clarifies)
        return self.selected_product is not None and self.selected_country is not None

def extract_entities(message: str) -> dict:
    """
    Step 1: Use AI to extract structured data from the message.
    """
    prompt = f"""Analyze this user message for an Optimum Nutrition support bot.
Extract the following fields into a valid JSON object:
- intent: ("price", "authenticity", "where_to_buy", "dietary", "greeting", or null)
- category: ("protein", "gainer", "energy", "pre_workout", "recovery", "vitamins", or null)
- product: (The full name of the product from the list below, or null)
- pack_size: (e.g. "2LB", "5LB", "10LB", "100 softgels", or null)
- country: ("UAE", "KSA", "Egypt", or null)

Available Products (MUST use these names exactly): 
Gold Standard 100% Whey, Serious Mass, Gold Standard Isolate, Platinum HydroWhey, 
Essential Amin.O. Energy, Gold Standard Pre-Workout, Gold Standard 100% Casein, 
Glutamine Powder, BCAA 5000, Opti-Men, Opti-Women, Fish Oil Softgels, Micronized Creatine Powder.

Context Extraction Rules:
1. If user asks "price", "cost", "how much", intent is "price".
2. If user mentions "UAE", "Dubai", "KSA", "Saudi", "Egypt", extract country.
3. Map "Whey", "Isolate" to category "protein" IF product name isn't fully clear.

Message: "{message}"
JSON:"""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',  # Consistent with generation model
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

    # 1. State Retrieval
    if user_id not in USER_STATES:
        USER_STATES[user_id] = ConversationState(user_id)
    state = USER_STATES[user_id]

    # 2. Extract Entities
    entities = extract_entities(message)
    logger.info(f"Entities extracted for {user_id}: {entities}")

    # 3. State Update Logic
    # Update intent only if a new clear intent is found, otherwise persist
    if entities.get('intent'):
        state.pending_intent = entities['intent']
    
    # Update slots
    if entities.get('category'): state.selected_category = entities['category']
    if entities.get('product'): state.selected_product = entities['product']
    if entities.get('pack_size'): state.selected_pack = entities['pack_size']
    if entities.get('country'): state.selected_country = entities['country']

    # 4. Deterministic Flow Control
    instruction = ""
    decision = "ask_clarification"

    # GREETING
    if state.pending_intent == "greeting":
        instruction = "Respond using the WELCOME/GREETING section. Ask how you can help."
        state.pending_intent = None # Clear after greeting

    # AUTHENTICITY
    elif state.pending_intent == "authenticity":
        instruction = "Answer using the AUTHENTICITY CHECK or NO STICKER POLICY sections. Max 2 sentences."
        state.pending_intent = None

    # DIETARY
    elif state.pending_intent == "dietary":
        instruction = "Answer using the DIETARY / COMPLIANCE section only. Do not give medical advice."
        state.pending_intent = None

    # PRICING (The complex flow)
    elif state.pending_intent == "price":
        # 4a. Handle Category Mapping
        if state.selected_category and not state.selected_product:
            products_in_cat = CATEGORY_MAP.get(state.selected_category, [])
            if len(products_in_cat) == 1:
                state.selected_product = products_in_cat[0]
            else:
                instruction = f"We have multiple products in the {state.selected_category} category: {', '.join(products_in_cat)}. Ask which one they want the price for."
                state.last_question = "product"
        
        # 4b. Missing Product
        if not state.selected_product:
            instruction = "Politely ask which product they would like to know the price of."
            state.last_question = "product"
        
        # 4c. Missing Country
        elif not state.selected_country:
            instruction = f"User wants the price for {state.selected_product}. Ask for their country (UAE, KSA, or Egypt) to show the correct price."
            state.last_question = "country"
        
        # 4d. Ready to give Price!
        else:
            instruction = f"Provide the exact price for {state.selected_product} in {state.selected_country} from the PRICING section. Max 2 sentences. Include the website disclaimer."
            decision = "provide_answer"
            # Clear state after fulfilling intent
            state.pending_intent = None
            state.selected_product = None
            state.selected_category = None
            state.selected_country = None

    # FALLBACK
    else:
        instruction = "Use the PRODUCT OVERVIEW to list what we sell and ask which one they need details for. If outside scope, use the website fallback."

    # 5. Response Formatting (AI Step)
    system_instruction = f"""You are the official Optimum Nutrition (ON) support assistant.
Your goal is to follow the INSTRUCTION below using ONLY the provided Knowledge Base.

KB SECTIONS:
{ON_KNOWLEDGE_BASE}

STRICT RESPONSE RULES:
- Never generate partial or cut-off sentences.
- Never guess or invent prices.
- Be professional and concise. Max 2 sentences.
- INSTRUCTION: {instruction}
"""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=message,
            config={
                "system_instruction": system_instruction,
                "temperature": 0.0,
                "max_output_tokens": 150
            }
        )
        if response and response.text:
            return response.text.strip()
    except Exception as e:
        logger.error(f"Response Generation Error: {e}")
        
    return "Please check our official website for the most updated information."
