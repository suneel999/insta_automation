"""
Instagram API Client
Handles sending messages and quick replies to Instagram via Graph API.
"""

import os
import requests
import logging
from dotenv import load_dotenv

# Load local env
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

logger = logging.getLogger(__name__)

GRAPH_API_URL = "https://graph.instagram.com/v24.0"

def get_token():
    return os.getenv('INSTAGRAM_USER_ACCESS_TOKEN', '').strip()

def send_text_message(recipient_id: str, text: str) -> bool:
    """Send a simple text message."""
    token = get_token()
    if not token:
        logger.error("INSTAGRAM_USER_ACCESS_TOKEN not set")
        return False
        
    url = f"{GRAPH_API_URL}/me/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "recipient": {"id": str(recipient_id)},
        "message": {"text": text}
    }
    
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            logger.info(f"Instagram text sent to {recipient_id}")
            return True
        logger.error(f"Instagram API error: {r.status_code} - {r.text}")
        return False
    except Exception as e:
        logger.error(f"Instagram send failed: {e}")
        return False

def send_quick_replies(recipient_id: str, text: str, quick_replies: list) -> bool:
    """
    Send text with quick reply buttons.
    Instagram limit: 13 buttons, 20 chars max title.
    """
    token = get_token()
    if not token:
        return False
        
    url = f"{GRAPH_API_URL}/me/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Prune and validate quick replies
    valid_qr = []
    for qr in quick_replies[:13]:
        title = str(qr.get('title', ''))[:20]
        payload = str(qr.get('payload', title))
        if title:
            valid_qr.append({
                "content_type": "text",
                "title": title,
                "payload": payload
            })
            
    payload = {
        "recipient": {"id": str(recipient_id)},
        "message": {
            "text": text,
            "quick_replies": valid_qr
        }
    }
    
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            logger.info(f"Instagram quick replies sent to {recipient_id}")
            return True
        logger.error(f"Instagram API error: {r.status_code} - {r.text}")
        return False
    except Exception as e:
        logger.error(f"Instagram QR send failed: {e}")
        return False
