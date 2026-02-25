"""
Instagram AI Handler for Optimum Nutrition (ON) - TWO-MODE PRODUCTION ARCHITECTURE
Separates Discovery (Product Browsing) from Transaction (Statedful Pricing).
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

# --- MEMORY STORE ---
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
        self.last_asked_slot = None

# Configure Gemini Client
api_key = os.getenv('GEMINI_API_KEY', '')
client = None
if api_key:
    client = genai.Client(api_key=api_key)

def extract_entities(message: str) -> dict:
    """
    Step 1: Extract intent and slots from the user message.
    """
    prompt = f"""Analyze this message for an Optimum Nutrition support bot.
Extract the following fields into JSON:
- intent: ("price", "discovery", "authenticity", "greeting", or null)
- product: (Full name from KB if mentioned, or null)
- category: ("protein", "gainer", "energy", "pre_workout", "vitamins", or null)
- country: ("UAE", "KSA", "Egypt", or null)

Rules:
1. Intent is "price" if user asks "cost", "how much", "price", or mentions a currency.
2. Intent is "discovery" if user asks "what products", "tell me about", "what categories", or mentions a product without asking for price.
3. Map "Dubai/Abu Dhabi" to "UAE", "Saudi/Riyadh" to "KSA".

Message: "{message}"
JSON:"""

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
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
        CONVERSATION_HISTORY[user_id] = []
    
    state = USER_STATES[user_id]
    hist = CONVERSATION_HISTORY[user_id]

    # 2. Extract Entities
    entities = extract_entities(message)
    logger.info(f"System State for {user_id}: Mode={state.mode} | Extracted={entities}")

    # 3. Logic: Mode Switching & Slot Filling
    # Switch to Transaction Mode if price is mentioned
    if entities.get('intent') == "price":
        state.mode = "transaction"
        state.pending_intent = "price"
    
    # Update slots (Slots persist in both modes)
    if entities.get('product'): state.selected_product = entities['product']
    if entities.get('category'): state.selected_category = entities['category']
    if entities.get('country'): state.selected_country = entities['country']

    # 4. Mode-Based Instruction Generation
    instruction = ""
    
    # MODE A: TRANSACTION (Price Logic)
    if state.mode == "transaction":
        if not state.selected_product:
            instruction = "We are in PRICING mode. No product identified yet. Ask: 'Which specific product would you like the price for?' List 3 popular options (Whey, Serious Mass, Amin.O. Energy)."
            state.last_asked_slot = "product"
        elif not state.selected_country:
            instruction = f"We have the product: {state.selected_product}. We need the location. Ask: 'To give you the correct price for {state.selected_product}, which country are you in: UAE, KSA, or Egypt?'"
            state.last_asked_slot = "country"
        else:
            instruction = f"PRICING FULFILLMENT: Provide the exact price for {state.selected_product} in {state.selected_country} from the KB. After answering, clear the price intent."
            # Clear transactional state after this response
            state.mode = "discovery"
            state.pending_intent = None
            state.selected_product = None
            state.selected_country = None

    # MODE B: DISCOVERY (Browsing/Greeting)
    else:
        if entities.get('intent') == "greeting":
            instruction = "Greeting: Use the WELCOME section. Ask how you can help."
        elif entities.get('intent') == "authenticity":
            instruction = "Authenticity: Explain the sticker and 6-digit code policy clearly."
        else:
            # General Browsing (RAG-like)
            instruction = "DISCOVERY MODE: The user is browsing. Briefly describe our main categories (Protein, Energy, Vitamins). If they mentioned a category, list the products in it. Do NOT ask for country or price info here."

    # 5. Final Generator Prompt
    system_instruction = f"""You are the official Optimum Nutrition support assistant.
TASK MISSION: {instruction}

STRICT OPERATIONAL RULES:
- If in DISCOVERY mode, be descriptive and helpful. Do not mention prices.
- If in TRANSACTION mode, be precise and data-driven.
- ALWAYS complete your sentences. Ending with a period or question mark is mandatory.
- Use ONLY the Knowledge Base below.
- Max 2 sentences total.

=== KNOWLEDGE BASE ===
{ON_KNOWLEDGE_BASE}
"""

    try:
        # History window management
        hist.append(types.Content(role="user", parts=[types.Part(text=message)]))
        if len(hist) > 6: 
            CONVERSATION_HISTORY[user_id] = hist[-6:]
            hist = CONVERSATION_HISTORY[user_id]

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=hist,
            config={
                "system_instruction": system_instruction,
                "temperature": 0.1,
                "max_output_tokens": 400
            }
        )
        
        if response and response.text:
            ai_text = response.text.strip()
            # Safety check for truncation
            if not ai_text.endswith(('.', '?', '!')):
                logger.warning(f"Response might be truncated: {ai_text}")
            
            logger.info(f"AI Final Output for {user_id}: {ai_text}")
            hist.append(types.Content(role="model", parts=[types.Part(text=ai_text)]))
            return ai_text
            
    except Exception as e:
        logger.error(f"Generation error: {e}")
        
    return "Please check our official website for the most updated information."
