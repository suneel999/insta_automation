"""
Instagram AI Handler for Optimum Nutrition (ON) - ROBUST RECOVERY VERSION
Fixed: Switched to stable models, added deeper error logging, and refined mode extraction.
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

# Configure Gemini Client
api_key = os.getenv('GEMINI_API_KEY', '')
client = None
if api_key:
    client = genai.Client(api_key=api_key)
else:
    logger.error("CRITICAL: GEMINI_API_KEY not found in environment!")

def extract_entities(message: str) -> dict:
    """
    Step 1: Extract intent and slots using the most STABLE model.
    """
    if not client:
        return {}

    prompt = f"""Extract intent and products from this message into JSON format.
Fields: intent (price/discovery/authenticity/greeting), product (name), country (UAE/KSA/Egypt).

Rules:
- intent is "price" if asking for cost/how much.
- intent is "discovery" if browsing/asking about products.
- intent is "greeting" if saying hi/hello.

Message: "{message}"
JSON:"""

    try:
        # Using 1.5-flash for maximum reliability in JSON mode
        response = client.models.generate_content(
            model='gemini-1.5-flash', 
            contents=prompt,
            config={"temperature": 0.0, "response_mime_type": "application/json"}
        )
        if response and response.text:
            return json.loads(response.text)
    except Exception as e:
        logger.error(f"Extraction Error (Model might be down/quota): {e}")
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
    logger.info(f"User {user_id} State: {state.mode} | Input Intent: {entities.get('intent')}")

    # 3. Logic: Mode Switching
    if entities.get('intent') == "price":
        state.mode = "transaction"
    
    if entities.get('product'): state.selected_product = entities['product']
    if entities.get('country'): state.selected_country = entities['country']

    # 4. Mode-Based Instruction
    instruction = ""
    if state.mode == "transaction":
        if not state.selected_product:
            instruction = "Price mode: We need a product. Ask: 'Which product would you like to know the price of?'"
        elif not state.selected_country:
            instruction = f"Price mode for {state.selected_product}: We need the location. Ask: 'Which country are you in (UAE, KSA, or Egypt) to provide the correct price?'"
        else:
            instruction = f"Provide the exact price for {state.selected_product} in {state.selected_country} from the KB. End the price intent after this."
            state.mode = "discovery" # Reset after answering
    else:
        # Discovery / Greeting
        if entities.get('intent') == "greeting":
            instruction = "Greeting: Use the WELCOME section. Ask how you can help today."
        else:
            instruction = "Discovery: Use the PRODUCT OVERVIEW to list categories and ask which one they need details for."

    # 5. Final Generator
    system_instruction = f"""You are the official Optimum Nutrition support.
TASK MISSION: {instruction}

RULES:
- Use ONLY the KB below.
- Do NOT repeat greetings.
- Max 2 sentences.
- Always finish your sentence.

KNOWLEDGE BASE:
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
                "max_output_tokens": 400
            }
        )
        
        if response and response.text:
            ai_text = response.text.strip()
            logger.info(f"AI for {user_id}: {ai_text}")
            hist.append(types.Content(role="model", parts=[types.Part(text=ai_text)]))
            return ai_text
            
    except Exception as e:
        logger.error(f"Generation Error (Model/Quota): {e}")
        
    return "Please check our official website for the most updated information."
