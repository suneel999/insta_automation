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
        # We put the entire text KB in the system instruction
        system_instruction = f"""You are the official automated assistant for Optimum Nutrition (ON). 

STRICT GUIDELINES:
1. Use ONLY the 'ON OFFICIAL Q&A' below to answer questions.
2. If the user asks for a price of any product (e.g. Fish Oil, Whey), provide the EXACT answer from the text.
3. If they ask about authenticity or where to buy, use the text below.
4. If a question is NOT answered in the text below, say: "Please check our official website for the most updated information."
5. Be direct and concise. Max 2 sentences. No emojis or filler words.

=== ON OFFICIAL Q&A ===
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
