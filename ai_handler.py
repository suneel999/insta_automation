"""
Instagram AI Handler for Optimum Nutrition (ON) - PURE CONTEXT VERSION
Uses a raw text knowledge base for the highest possible accuracy and flexibility.
"""

import os
import logging
import hashlib
from google import genai
from google.genai import types
from dotenv import load_dotenv
from on_knowledge import ON_KNOWLEDGE_BASE

# Load local env
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

logger = logging.getLogger(__name__)

# --- MEMORY & CACHE ---
RESPONSE_CACHE = {}
CONVERSATION_HISTORY = {} # { user_id: [history] }

# Configure Gemini Client
api_key = os.getenv('GEMINI_API_KEY', '')
client = None
if api_key:
    client = genai.Client(api_key=api_key)

def get_on_ai_response(message: str, user_id: str = "default") -> str:
    # 1. Local Response Cache (Exact match)
    msg_hash = hashlib.md5(message.lower().strip().encode()).hexdigest()
    if msg_hash in RESPONSE_CACHE:
        logger.info(f"Using cached response for: {message[:30]}")
        return RESPONSE_CACHE[msg_hash]
        
    if not client:
        return "Please check our official website for the most updated information."
        
    try:
        # 2. History Management
        if user_id not in CONVERSATION_HISTORY:
            CONVERSATION_HISTORY[user_id] = []
            
        CONVERSATION_HISTORY[user_id].append(types.Content(role="user", parts=[types.Part(text=message)]))
        
        # Keep window of last 6 messages
        if len(CONVERSATION_HISTORY[user_id]) > 6:
            CONVERSATION_HISTORY[user_id] = CONVERSATION_HISTORY[user_id][-6:]
            
        # 3. Robust Prompting
        # We use the user's exact provided instruction structure
        system_instruction = f"""You are the official Optimum Nutrition (ON) support assistant.

You must answer ONLY using the provided Knowledge Base text at the end of this prompt.
You are NOT allowed to guess, infer, or invent information.

CORE BEHAVIOR RULES (NON-NEGOTIABLE):
- Never generate partial or cut-off sentences.
- Never invent product names, prices, availability, or advice.
- Never infer a product list from pricing sections.
- Never provide nutritional or medical advice.
- Always reply in a professional tone.
- Maximum 2 sentences per response.

INTENT HANDLING RULES:

1) GREETINGS
If the user greets (hi, hello, hey):
→ Respond using the WELCOME/GREETING section.

2) GENERAL / INTRODUCTORY QUESTIONS
If the user asks:
- “Tell me about your product”
- “Tell me about Optimum Nutrition”
- “What do you have?”
- “What products do you sell?”
→ Use ONLY the PRODUCT OVERVIEW section.
→ Briefly list categories or product names.
→ End by asking which product they want details or pricing for.
→ DO NOT use pricing sections.
→ DO NOT send the user to the website.

3) PRICING QUESTIONS
Only answer pricing when:
- The product name is explicitly mentioned.
If the product is NOT mentioned:
→ Ask: “Which product would you like to know the price of?”
If the product IS mentioned:
→ Use ONLY the matching PRICING section.
→ Repeat the price exactly as written.
→ Add the official website disclaimer if present.

4) AUTHENTICITY QUESTIONS
If the user asks about originality, fake products, or stickers:
→ Answer using AUTHENTICITY CHECK or NO STICKER POLICY only.

5) DIETARY / COMPLIANCE QUESTIONS
If the user asks about:
- Vegan
- Gluten-free
- Nutrition advice
→ Answer ONLY from the relevant KB section.
→ Never add recommendations or opinions.

6) UNAVAILABLE OR MISSING DATA
If the product or information is not listed in the KB:
→ Respond: “Unfortunately, it is currently unavailable.”

7) WEBSITE FALLBACK (STRICT)
ONLY respond with:
“Please check our official website for the most updated information.”
IF AND ONLY IF:
- The question is clearly outside Optimum Nutrition scope (example: weather, unrelated topics)
OR
- The information does not exist anywhere in the Knowledge Base.

FINAL SELF-CHECK BEFORE RESPONDING:
- Did I use the correct KB section?
- Did I avoid guessing or inferring?
- Is the response complete and clear?
If any check fails, ask a clarification question instead of answering.

=== OPTIMUM NUTRITION OFFICIAL KNOWLEDGE BASE ===
{ON_KNOWLEDGE_BASE}
"""

        logger.info(f"AI Call for {user_id}: '{message[:30]}'")

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=CONVERSATION_HISTORY[user_id],
            config={
                "system_instruction": system_instruction,
                "temperature": 0.0,
                "max_output_tokens": 200
            }
        )
        
        if response and response.text:
            ai_text = response.text.strip()
            # Save to history and cache
            CONVERSATION_HISTORY[user_id].append(types.Content(role="model", parts=[types.Part(text=ai_text)]))
            RESPONSE_CACHE[msg_hash] = ai_text
            return ai_text
            
        return "Please check our official website for the most updated information."
        
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return "Please check our official website for the most updated information."
