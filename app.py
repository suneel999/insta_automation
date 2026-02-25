"""
Standalone Instagram Application for Optimum Nutrition (ON)
Completely separate from the WhatsApp Clinic Bot.
"""

import os
import json
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load from the LOCAL .env file in the instagram folder
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from processor import process_instagram_message

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    """Instagram Webhook Endpoint"""
    
    # Verification (GET)
    if request.method == 'GET':
        mode = request.args.get('hub.mode')
        token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')
        
        verify_token = os.getenv('INSTAGRAM_VERIFY_TOKEN')
        
        if mode == 'subscribe' and token == verify_token:
            logger.info("INSTAGRAM WEBHOOK VERIFIED")
            return challenge, 200
        return 'Forbidden', 403

    # Event Handling (POST)
    data = request.get_json()
    if not data:
        return jsonify({"status": "error"}), 400

    if data.get('object') == 'instagram':
        for entry in data.get('entry', []):
            for messaging_event in entry.get('messaging', []):
                # Skip if it's not a message or it's an echo
                if 'message' not in messaging_event:
                    continue
                
                message = messaging_event['message']
                if message.get('is_echo'):
                    continue
                
                sender_id = messaging_event['sender']['id']
                text = message.get('text', '')
                
                if text:
                    process_instagram_message(sender_id, text)

    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8001))
    logger.info(f"Starting Instagram Automation on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
