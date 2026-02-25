"""
Instagram AI Handler for Optimum Nutrition (ON) - FAIL-SAFE PRODUCTION VERSION
Fixed: Overuse of website fallback, loop prevention, and helpful discovery mode.
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

# --- CONFIG & FALLBACKS ---
SAFE_DEFAULT = "I can help with product information, prices, availability, and authenticity. What would you like to know?"
WEBSITE_FALLBACK = "Please check our official website for the most updated information."

USER_STATES = {}  # { user_id: ConversationState }
CONVERSATION_HISTORY = {} # { user_id: [history_items] }

class ConversationState:
    def __init__(self, user_id):
        self.user_id = user_id
        self.mode = "discovery" # discovery | transaction
        self.pending_intent = None
        self.selected_category = None
        self.selected_product = None
        self.selected_country = None

# Configure Gemini Client
api_key = os.getenv('GEMINI_API_KEY', '')
client = None
if api_key:
    client = genai.Client(api_key=api_key)
else:
    logger.error("CRITICAL: GEMINI_API_KEY not found!")

def extract_entities(message: str) -> dict:
    if not client: return {}
    
    prompt = f"""Extract fields from this message into JSON:
- intent: (price, discovery, authenticity, greeting, dietary, or null)
- product: (Full name or null)
- country: (UAE, KSA, Egypt, or null)

Rules:
1. Intent "price" if asking for cost/how much.
2. Intent "discovery" if browsing or asking about "products" or "what do you have".
3. Intent "greeting" for hello/hi.

Message: "{message}"
JSON:"""

    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
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
        return WEBSITE_FALLBACK

    # 1. State/History Retrieval
    if user_id not in USER_STATES:
        USER_STATES[user_id] = ConversationState(user_id)
        CONVERSATION_HISTORY[user_id] = []
    
    state = USER_STATES[user_id]
    hist = CONVERSATION_HISTORY[user_id]

    # 2. Extract Entities
    entities = extract_entities(message)
    logger.info(f"User {user_id} | Intent: {entities.get('intent')} | Mode: {state.mode}")

    # 3. Logics & Mode Switching
    if entities.get('intent') == "price":
        state.mode = "transaction"
    
    if entities.get('product'): state.selected_product = entities['product']
    if entities.get('country'): state.selected_country = entities['country']

    # 4. Deterministic Instruction Selection
    instruction = ""
    
    # GREETING
    if entities.get('intent') == "greeting":
        instruction = "GREET the user using the WELCOME section. Ask how you can help. DO NOT mention the website."
    
    # PRICING (Transaction Mode)
    elif state.mode == "transaction":
        if not state.selected_product:
            instruction = "Price Mode: We need a product name. Ask: 'Which product would you like to know the price of?' and list examples (Whey, Serious Mass, Amin.O. Energy)."
        elif not state.selected_country:
            instruction = f"Price Mode for {state.selected_product}: Ask: 'To give you the correct price, which country are you in: UAE, KSA, or Egypt?'"
        else:
            instruction = f"Provide exactly the price for {state.selected_product} in {state.selected_country} from the KB PRICING section. End the price session after this."
            state.mode = "discovery" # Reset flow
    
    # DISCOVERY / BROWSING
    elif entities.get('intent') == "discovery" or state.mode == "discovery":
        # Check if they asked for a specific category
        cat_info = "categories like Protein, Energy, and Vitamins"
        if entities.get('product'):
            instruction = f"The user asked about {entities.get('product')}. Use the KB to describe its features briefly and ask if they want its price."
        else:
            instruction = f"DISCOVERY: Use the PRODUCT OVERVIEW to list our main products (Whey, Serious Mass, Amin.O. Energy, Vitamins). Ask which one they want details for. NEVER use the website fallback here."

    # DEFAULT FALLBACK (If everything else fails to map)
    if not instruction:
        return SAFE_DEFAULT

    # 5. Final Response Generation
    system_instruction = f"""### YOUR TASK (PRIORITY): 
{instruction}

### STRICT RULES:
1. USE ONLY the Knowledge Base below.
2. DO NOT use the website fallback for Greetings or Discovery.
3. If info is truly missing for a PRICE request, say "Unfortunately, that product is unavailable." 
4. If the user query is vague, use the SAFE DEFAULT below.
5. Max 2 sentences. ALWAYS finish your sentence.

SAFE DEFAULT: {SAFE_DEFAULT}

### KNOWLEDGE BASE:
{ON_KNOWLEDGE_BASE}
"""

    try:
        hist.append(types.Content(role="user", parts=[types.Part(text=message)]))
        if len(hist) > 6: hist = hist[-6:]

        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=hist,
            config={
                "system_instruction": system_instruction,
                "temperature": 0.1,
                "max_output_tokens": 300
            }
        )
        
        if response and response.text:
            ai_text = response.text.strip()
            # Loop check
            if len(hist) > 2 and ai_text == hist[-2].parts[0].text:
                logger.warning("Repetition detected. Forcing safe default.")
                return SAFE_DEFAULT
                
            logger.info(f"Final AI for {user_id}: {ai_text}")
            hist.append(types.Content(role="model", parts=[types.Part(text=ai_text)]))
            return ai_text
            
    except Exception as e:
        logger.error(f"Gen Error: {e}")
        
    return SAFE_DEFAULT
