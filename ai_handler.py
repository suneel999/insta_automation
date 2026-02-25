"""
Instagram AI Handler for Optimum Nutrition (ON) - PERSONA-DRIVEN ROBUST VERSION
Implements the "Expert Assistant" persona with strict slot-filling and natural continuation.
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
        self.last_asked = None

# Configure Gemini Client
api_key = os.getenv('GEMINI_API_KEY', '')
client = None
if api_key:
    client = genai.Client(api_key=api_key)

def extract_entities(message: str, history=None) -> dict:
    """
    Extracts slots and intent. Context-aware to handle 'short replies'.
    """
    if not client: return {}
    
    # We pass history to extraction to resolve "this one" or "hydro"
    context = ""
    if history:
        # Just the last 2 turns for context
        context = "\n".join([f"{c.role}: {c.parts[0].text}" for c in history[-2:]])

    prompt = f"""Context:
{context}

Message: "{message}"

Extract JSON:
- intent: (price, discovery, authenticity, greeting, dietary, or null)
- product: (Full name or null)
- country: (UAE, KSA, Egypt, or null)

Rules:
1. If user says 'price', 'cost', 'how much', or 'this' after a cost question, intent is 'price'.
2. If message is short (e.g., 'hydro', 'whey', 'uae'), map it to the correct field.
3. If mentioning a category or browsing, intent is 'discovery'.

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
        return "Please check our official website for the most updated information."

    # 1. State/History Retrieval
    if user_id not in USER_STATES:
        USER_STATES[user_id] = ConversationState(user_id)
        CONVERSATION_HISTORY[user_id] = []
    
    state = USER_STATES[user_id]
    hist = CONVERSATION_HISTORY[user_id]

    # 2. Extract (Context-Aware)
    entities = extract_entities(message, hist)
    logger.info(f"User {user_id} | State: {state.mode} | Intent: {entities.get('intent')} | Slots: {entities}")

    # 3. Logic: Slot Filling & Continuity
    # Persist intent if it's already transaction mode and message is a short reply
    if entities.get('intent') == "price":
        state.mode = "transaction"
        state.pending_intent = "price"
    
    # Short reply handling: if we were waiting for a product and user says "hydro", extract handles it.
    if entities.get('product'): state.selected_product = entities['product']
    if entities.get('country'): state.selected_country = entities['country']

    # 4. Instruction Generation
    instruction = ""
    
    if entities.get('intent') == "greeting":
        instruction = "GREET the user warmly. Use the WELCOME/GREETING section from the KB."
        state.mode = "discovery" # Reset to discovery on new greeting
    
    elif state.mode == "transaction" or state.pending_intent == "price":
        if not state.selected_product:
            instruction = "We are in pricing mode. WE NEED THE PRODUCT. Ask: 'Which product would you like to know the price of?' List 2-3 specific options like Gold Standard Whey or Serious Mass."
            state.last_asked = "product"
        elif not state.selected_country:
            instruction = f"We have the product: {state.selected_product}. WE NEED THE COUNTRY. Ask: 'To give you the exact price, which country are you in: UAE, KSA, or Egypt?'"
            state.last_asked = "country"
        else:
            instruction = f"FULFILL PRICE: Provide the exact price for {state.selected_product} in {state.selected_country} from the PRICING section. Add the website disclaimer. Then reset the pricing flow."
            # Reset after fulfillment
            state.pending_intent = None
            state.selected_product = None
            state.selected_country = None
            state.mode = "discovery"

    else:
        # Discovery / General Help
        if entities.get('intent') == "authenticity":
            instruction = "Explain the AUTHENTICITY policy and the 6-digit code. Max 2 sentences."
        else:
            instruction = "DISCOVERY: Help the user browse products. List our main categories (Protein, Energy, Vitamins) and specific products in them. Ask what they are interested in."

    # 5. Final Generation with the "Expert Assistant" Persona
    system_instruction = f"""You are the official Optimum Nutrition (ON) support assistant. 
You must follow the TASK MISSION below using ONLY the provided Knowledge Base.

### RULES:
- Always use conversation memory to understand follow-up questions.
- Treat short replies like "price", "this", "that", "hydro" as continuations.
- If information is missing, ask ONLY for the missing piece.
- NEVER reset context unless the user clearly changes topic.
- NEVER invent prices, availability, or products.
- Website fallback is LAST resort only when data truly does not exist in the KB.
- Never repeat the same question.
- Never send generic help messages.
- Max 2 sentences per reply.

### TASK MISSION:
{instruction}

### KNOWLEDGE BASE:
{ON_KNOWLEDGE_BASE}
"""

    try:
        hist.append(types.Content(role="user", parts=[types.Part(text=message)]))
        if len(hist) > 8: 
            CONVERSATION_HISTORY[user_id] = hist[-8:]
            hist = CONVERSATION_HISTORY[user_id]

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
            logger.info(f"AI Response for {user_id}: {ai_text}")
            hist.append(types.Content(role="model", parts=[types.Part(text=ai_text)]))
            return ai_text
            
    except Exception as e:
        logger.error(f"Gen Error: {e}")
        
    return "Please check our official website for the most updated information."
