"""
Instagram Message Processor
Orchestrates incoming Instagram messages and routes them to the AI handler.
"""

import logging
from client import send_text_message
from ai_handler import get_on_ai_response

logger = logging.getLogger(__name__)

def process_instagram_message(sender_id: str, message_text: str):
    """
    Main entry point for Instagram webhook events.
    1. Logs interaction
    2. Gets AI response from Gemini using ON KB (with history support)
    3. Sends response back to Instagram
    """
    logger.info(f"Processing Instagram message from {sender_id}: {message_text[:50]}")
    
    # Get response from specialized AI - now passing sender_id for context memory
    ai_response = get_on_ai_response(message_text, user_id=sender_id)
    
    # Send message back
    success = send_text_message(sender_id, ai_response)
    
    if success:
        logger.info(f"Successfully replied to Instagram user {sender_id}")
    else:
        logger.error(f"Failed to reply to Instagram user {sender_id}")
    
    return success
