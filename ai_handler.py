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

# --- MEMORY & CACHE ---
RESPONSE_CACHE = {}
USER_STATES = {}  # { user_id: state_dict }
CONVERSATION_HISTORY = {} # { user_id: [history] }

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

def extract_entities(message: str) -> dict:
    prompt = f"""Analyze this user message for an Optimum Nutrition support bot.
Extract the following fields into a valid JSON object:
- intent: ("price", "authenticity", "where_to_buy", "dietary", "greeting", or null)
- category: ("protein", "gainer", "energy", "pre_workout", "recovery", "vitamins", or null)
- product: (The full name or null)
- pack_size: (null or value)
- country: ("UAE", "KSA", "Egypt", or null)

Rules:
1. If the message is "Price" or "What is the cost?", intent MUST be "price".
2. If it's a greeting, intent is "greeting".
3. Extract ANY mentioned country.

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

    # 1. State & History Retrieval
    if user_id not in USER_STATES:
        USER_STATES[user_id] = ConversationState(user_id)
        CONVERSATION_HISTORY[user_id] = []
    
    state = USER_STATES[user_id]
    hist = CONVERSATION_HISTORY[user_id]

    # 2. Extract Entities
    entities = extract_entities(message)
    logger.info(f"Parsed for {user_id}: {entities}")

    # 3. State Update Logic
    if entities.get('intent'): state.pending_intent = entities['intent']
    if entities.get('category'): state.selected_category = entities['category']
    if entities.get('product'): state.selected_product = entities['product']
    if entities.get('country'): state.selected_country = entities['country']

    # 4. Flow Control (Instruction Generation)
    instruction = ""
    
    if state.pending_intent == "greeting":
        instruction = "GREET the user warmly using the WELCOME/GREETING section. Ask how you can help."
        state.pending_intent = None 
    elif state.pending_intent == "price":
        if not state.selected_product:
            instruction = "The user wants pricing. IDENTIFY the product first. Ask: 'Which product would you like to know the price of?' and list a few examples like Whey or Serious Mass."
        elif not state.selected_country:
            instruction = f"Target Product: {state.selected_product}. We need the location. ASK: 'Which country are you in (UAE, KSA, or Egypt) to show the correct price?'"
        else:
            instruction = f"Target: {state.selected_product} in {state.selected_country}. PROVIDE the exact price from the PRICING section. Do not ask more questions."
            state.pending_intent = None # Reset after fulfillment
    else:
        # General Help/Fallback
        instruction = "Use the PRODUCT OVERVIEW to briefly list categories. Ask which specific product they need details for. DO NOT repeat the greeting if the history shows you already said hello."

    # 5. Response Generation
    system_instruction = f"""### YOUR CURRENT TASK (PRIORITY): 
{instruction}

### CORE RULES:
1. Use ONLY the Knowledge Base below.
2. DO NOT REPEAT the welcome/greeting if history shows you already greeted the user.
3. NEVER generate cut-off sentences. Max 2 sentences.
4. If you don't have the info, say "Please check our official website for more details."

### KNOWLEDGE BASE:
{ON_KNOWLEDGE_BASE}
"""

    try:
        # Update history
        hist.append(types.Content(role="user", parts=[types.Part(text=message)]))
        if len(hist) > 6: hist = hist[-6:]

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=hist,
            config={
                "system_instruction": system_instruction,
                "temperature": 0.0,
                "max_output_tokens": 150
            }
        )
        
        if response and response.text:
            ai_text = response.text.strip()
            hist.append(types.Content(role="model", parts=[types.Part(text=ai_text)]))
            return ai_text
            
    except Exception as e:
        logger.error(f"Gen Error: {e}")
        
    return "Please check our official website for the most updated information."
