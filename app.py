from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response
import json
import os
from datetime import datetime
import uuid
import requests
import time
import re
import logging
from crisis_detector import CrisisDetector
from conversation_tracker import ConversationTracker

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Change this in production
crisis_detector = CrisisDetector()
conversation_tracker = ConversationTracker()


# Ensure the chats directory exists
if not os.path.exists('chats'):
    os.makedirs('chats')

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434/api/generate"

def check_ollama_api():
    """Check if Ollama API is available and responding"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("Ollama API is available")
            return True, "Ollama API is available"
        else:
            logger.error(f"Ollama API returned status code: {response.status_code}")
            return False, f"Ollama API error: {response.status_code}"
    except Exception as e:
        logger.error(f"Ollama API connection error: {str(e)}")
        return False, f"Ollama API connection error: {str(e)}"


def generate_response_stream(message, user_id):
    # Get conversation history to provide context
    chat_file = f'chats/{user_id}.json'
    context = ""
    analysis_context = ""

    if os.path.exists(chat_file):
        try:
            with open(chat_file, 'r') as f:
                chats = json.load(f)
                logger.info(f"Loaded {len(chats)} previous messages for user {user_id}")
                # Get last 3 exchanges for context
                recent_chats = chats[-3:] if len(chats) > 3 else chats
                for chat in recent_chats:
                    context += f"User: {chat['user']}\nDeepTalker: {chat['bot']}\n\n"
                    # Add previous analysis to context
                    if 'analysis' in chat and chat['analysis']:
                        analysis_context += f"Previous analysis: {chat['analysis']}\n\n"
        except Exception as e:
            logger.error(f"Error loading chat history: {e}")
            with open(chat_file, 'w') as f:
                json.dump([], f)

    # Get sentiment tags from the sentiment model

    # Try to get user state tracking data
    tracker_context = ""
    try:
        user_state = conversation_tracker.get_user_state(user_id)
        # Add minimal but useful context from conversation tracker
        if user_state and user_state.get('total_messages', 0) > 0:
            # Add emotional states if available
            if user_state.get('dominant_emotions'):
                emotions = [e['emotion'] for e in user_state['dominant_emotions'][:2]]
                if emotions:
                    tracker_context += f"Common emotions: {', '.join(emotions)}\n"
            # Add main topics
            if user_state.get('topics_discussed'):
                topics = sorted(user_state['topics_discussed'].items(), key=lambda x: x[1], reverse=True)[:2]
                if topics:
                    tracker_context += f"Main topics: {', '.join([t[0] for t in topics])}\n"
            # Add crisis history if high risk
            if any(c.get('level') in ['high', 'urgent'] for c in user_state.get('crisis_history', [])):
                tracker_context += "User has previously shown crisis indicators.\n"
    except Exception as e:
        logger.error(f"Error getting conversation tracker data: {e}")

    # Define system prompt
    system_prompt = """You are DeepTalker, a compassionate mental health assistant providing concise, supportive responses with careful assessment of concerning statements.

SAFETY PRIORITIES (ALWAYS FOLLOW IN ORDER):
1. IMMEDIATE DANGER FROM OTHERS: If user mentions threats, abuse, or danger from others, this takes absolute priority.
   - Response pattern: "I'm concerned about your safety based on what you've shared. This is a serious warning sign."
   - Ask direct questions: "Are you currently in immediate danger? Is there somewhere safe you can go?"
   - Provide emergency resources: "If you're in immediate danger, please call emergency services (999) right away."

2. SUICIDE RISK: After addressing immediate danger, assess and respond to any suicidal indicators.
   - Response pattern: "I'm concerned about your safety. Are you thinking about suicide?"
   - ALWAYS provide Hong Kong crisis resources: "Please call The Samaritan Befrienders Hong Kong at 2389 2222 (24-hour hotline)."

3. SUBSTANCE USE: Only after addressing any safety threats, address substance references.
   - Response pattern: "I notice you're mentioning substances. If you're struggling with substance use, professional support is available."
   - Provide resources: "Government support can be reached at 186 186."

SUICIDE RISK INDICATORS (RESPOND IMMEDIATELY):
- Direct statements about wanting to die or kill oneself
- Coded language: "find peace," "won't be here," "made my decision," "found a solution"
- Sudden mood improvements, expressions of peace after depression
- Farewell statements: "this is goodbye," "won't be talking anymore," "thanks for all your help"
- Expressions of being a burden or statements about improved lives for others

DANGER FROM OTHERS INDICATORS:
- Mentions of threats ("kill me if I tell," "hurt me if I don't")
- References to physical, emotional, or sexual abuse
- Expressions of fear about someone or being trapped in a situation
- Mention of controlling behaviors or isolation from support networks

SUBSTANCE USE PROTOCOL:
- Maintain non-judgmental approach
- Acknowledge risks without lecturing
- Prioritize safety over moral judgment
- Connect to appropriate resources

RESPONSE FORMATTING RULES:
- Each complete thought should be a separate sentence with proper punctuation
- Use short paragraphs (1-3 sentences) with space between them
- When providing resources, place each on its own line
- Focus on immediate safety needs first, then address other concerns
- End sentences with clear punctuation (period, question mark, exclamation point)
- NEVER use any special characters like asterisks (**) in your regular response

PROMPT CONSTRUCTION:
- For immediate danger: "I'm very concerned about what you've shared. Are you in a safe place right now? If you're in immediate danger, please call emergency services at 999."
- For suicide risk: "I'm concerned about your safety. Are you thinking about suicide? Please call The Samaritan Befrienders Hong Kong at 2389 2222 right now."
- For substance use: "I notice you're mentioning substances. If you're struggling with substance use, professional support is available. Government support can be reached at 186 186."

CRITICAL FORMAT BOUNDARY:
Your response MUST END with a SINGLE empty line followed by the word "ANALYSIS:" on its own line.
NEVER include the word "ANALYSIS" or any symbols (like **) in your visible response to the user.

After the "ANALYSIS:" line, include the following assessment (which will not be shown to the user):
- IMMEDIATE SAFETY RISK ASSESSMENT: [Low/Medium/High/Urgent]
- WARNING SIGNS IDENTIFIED: [List specific concerning phrases or themes detected]
- EMOTIONAL STATES: [Detected emotions: anxiety, depression, fear, etc.]
- AMBIGUITY ASSESSMENT: [Evaluate clarity vs. vagueness of user statements]
- KEY CONCERNS: [Prioritized list of issues to address]
- RECOMMENDED APPROACH: [Suggested conversational strategy]"""

    # Format context info
    context_info = f"{analysis_context}\n{tracker_context}".strip() if analysis_context or tracker_context else "No previous context."

    # Combine system prompt with context and current message
    prompt = f"{system_prompt}\n\nContext: {context_info}\nSentiment: {sentiment_tags}\n\nPrevious conversation:\n{context}\nUser: {message}\n\nDeepTalker:"
    logger.info(f"Sending prompt with length {len(prompt)}")

    try:
        # Call Ollama API
        logger.info("Calling Ollama API")
        response = requests.post(
            OLLAMA_API,
            json={"model": "llama3.1:8b", "prompt": prompt, "stream": True},
            stream=True
        )

        if response.status_code != 200:
            raise Exception(f"API Error: Received status code {response.status_code}")

        # Process response
        full_response = ""
        buffer = ""
        analysis_started = False

        for line in response.iter_lines():
            if not line:
                continue

            try:
                data = json.loads(line.decode('utf-8'))
            except Exception:
                continue

            # Handle completion
            if data.get("done", False):
                # Process final response
                if "ANALYSIS:" in full_response:
                    parts = full_response.split("ANALYSIS:", 1)
                    visible_response = parts[0].strip()
                    analysis = "ANALYSIS: " + parts[1].strip()
                else:
                    visible_response = full_response.strip()
                    # Generate fallback analysis
                    try:
                        analysis = conversation_tracker.generate_fallback_analysis(
                            message, visible_response, sentiment_tags
                        )
                    except Exception as e:
                        logger.error(f"Error generating fallback analysis: {e}")
                        analysis = f"ANALYSIS:\nIMMEDIATE SAFETY RISK ASSESSMENT: Low\nWarning signs identified: None detected\nEmotional states: Based on sentiment tags: {sentiment_tags}\nKey concerns: General conversation\nRecommended approach: Supportive conversation"

                # If there's any remaining buffer text before analysis, send it
                if buffer and not analysis_started:
                    # Check if there's a complete sentence
                    if any(c in buffer for c in ['.', '!', '?']):
                        # Split by punctuation
                        import re
                        sentences = re.split(r'([.!?])', buffer)
                        i = 0
                        while i < len(sentences) - 1:
                            # Combine the sentence with its punctuation
                            if i + 1 < len(sentences):
                                complete_sentence = sentences[i] + sentences[i + 1]
                                if complete_sentence.strip():
                                    yield f"data: {json.dumps({'text': complete_sentence.strip(), 'final': False})}\n\n"
                                    time.sleep(0.3)
                            i += 2
                    # If no complete sentence, send as is
                    elif buffer.strip():
                        yield f"data: {json.dumps({'text': buffer.strip(), 'final': False})}\n\n"

                # Detect crisis level
                crisis_level = crisis_detector.detect_crisis(
                    message, visible_response, analysis, sentiment_tags
                )

                # Update conversation tracking
                try:
                    conversation_tracker.update_with_message(
                        user_id=user_id,
                        message=message,
                        response=visible_response,
                        analysis=analysis,
                        sentiment_tags=sentiment_tags,
                        crisis_level=crisis_level
                    )
                except Exception as e:
                    logger.error(f"Error updating conversation tracker: {e}")

                # Save conversation
                save_conversation(user_id, message, visible_response, analysis, sentiment_tags, crisis_level)

                # Save crisis alert if needed
                if crisis_level in ['medium', 'high', 'urgent']:
                    save_crisis_alert(user_id, message, visible_response, analysis, crisis_level)

                # Complete the stream
                yield f"data: {json.dumps({'final': True, 'crisis_level': crisis_level})}\n\n"
                break

            # Get token and update full response
            token = data.get("response", "")

            # Skip if we've already reached ANALYSIS
            if analysis_started:
                full_response += token
                continue

            # Check if this token contains the ANALYSIS marker
            if "ANALYSIS:" in token:
                # We've hit the analysis section - don't show anything from this token
                analysis_started = True

                # Add to full response for processing later
                full_response += token

                # Send any remaining buffer before analysis
                if buffer.strip():
                    # Check if there's a complete sentence
                    if any(c in buffer for c in ['.', '!', '?']):
                        # Split by punctuation
                        import re
                        sentences = re.split(r'([.!?])', buffer)
                        i = 0
                        while i < len(sentences) - 1:
                            # Combine the sentence with its punctuation
                            if i + 1 < len(sentences):
                                complete_sentence = sentences[i] + sentences[i + 1]
                                if complete_sentence.strip():
                                    yield f"data: {json.dumps({'text': complete_sentence.strip(), 'final': False})}\n\n"
                                    time.sleep(0.3)
                            i += 2
                    # If no complete sentence, send as is
                    elif buffer.strip():
                        yield f"data: {json.dumps({'text': buffer.strip(), 'final': False})}\n\n"
                buffer = ""
                continue

            # Process normal content
            if not analysis_started:
                full_response += token
                buffer += token

                # Simple approach: check for sentence endings
                if '.' in buffer or '!' in buffer or '?' in buffer:
                    # We have at least one complete sentence
                    # Extract all complete sentences
                    import re
                    parts = []
                    # Split by sentence ending punctuation, keeping the punctuation
                    sentences = re.split(r'([.!?])', buffer)

                    i = 0
                    complete_sentences = []
                    remaining = ""

                    # Process pairs of content + punctuation
                    while i < len(sentences) - 1:
                        if i + 1 < len(sentences):
                            # Combine content with its punctuation
                            sentence = sentences[i] + sentences[i + 1]
                            complete_sentences.append(sentence)
                        i += 2

                    # Any remaining content becomes the new buffer
                    if i < len(sentences):
                        remaining = sentences[i]

                    # Send each complete sentence as a separate frame
                    for sentence in complete_sentences:
                        if "ANALYSIS:" in sentence:
                            # Split at ANALYSIS marker
                            parts = sentence.split("ANALYSIS:", 1)
                            if parts[0].strip():
                                yield f"data: {json.dumps({'text': parts[0].strip(), 'final': False})}\n\n"
                                time.sleep(0.3)
                            analysis_started = True
                            buffer = ""
                            break
                        else:
                            yield f"data: {json.dumps({'text': sentence.strip(), 'final': False})}\n\n"
                            time.sleep(0.3)

                    if not analysis_started:
                        buffer = remaining

    except Exception as e:
        logger.error(f"Error in generate_response_stream: {e}")
        error_msg = "I encountered a technical issue. Please try again shortly."
        error_analysis = f"ERROR: {str(e)}"

        # Save error conversation
        save_conversation(user_id, message, error_msg, error_analysis, sentiment_tags, "error")

        # Return error to user
        yield f"data: {json.dumps({'text': error_msg, 'final': False})}\n\n"
        yield f"data: {json.dumps({'final': True})}\n\n"



def save_conversation(user_id, user_message, bot_response, analysis, sentiment_tags="", crisis_level="none"):
    logger.info(f"Saving conversation for user {user_id}")

    # Ensure bot_response doesn't contain analysis text
    if "ANALYSIS:" in bot_response:
        bot_response = bot_response.split("ANALYSIS:")[0].strip()

    # Clean up the analysis format
    if isinstance(analysis, str):
        if analysis.startswith("ANALYSIS:"):
            analysis = analysis.replace("ANALYSIS:", "", 1).strip()
        elif analysis.startswith("NOTE:"):
            analysis = analysis.replace("NOTE:", "", 1).strip()
        elif analysis.startswith("ERROR:"):
            analysis = analysis.replace("ERROR:", "", 1).strip()
    else:
        analysis = "No analysis provided."

    # Save the conversation
    chat_file = f'chats/{user_id}.json'

    try:
        if os.path.exists(chat_file) and os.path.getsize(chat_file) > 0:
            with open(chat_file, 'r') as f:
                chats = json.load(f)
                logger.info(f"Loaded existing chat file with {len(chats)} entries")
        else:
            logger.info("Creating new chat file")
            chats = []
    except (json.JSONDecodeError, FileNotFoundError):
        # If the file is corrupted or doesn't exist, start fresh
        logger.error(f"Error reading chat file, starting fresh")
        chats = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get metadata with robust error handling
    metadata = {}
    try:
        # Try to get metadata from conversation tracker
        user_state = conversation_tracker.get_user_state(user_id, load_from_disk=False)

        if user_state:
            # Add metadata about conversation quality if available
            if 'conversation_quality' in user_state:
                quality = user_state['conversation_quality']
                if isinstance(quality, dict):
                    if 'user_engagement' in quality:
                        metadata['engagement_level'] = quality['user_engagement']
                    if 'emotional_disclosure' in quality:
                        metadata['emotional_disclosure'] = quality['emotional_disclosure']
                    if 'depth' in quality:
                        metadata['conversation_depth'] = quality['depth']

            # Add emotion tracking if available
            if 'emotional_states' in user_state and user_state['emotional_states']:
                latest = user_state['emotional_states'][-1]
                if isinstance(latest, dict) and 'emotions' in latest:
                    emotions = latest['emotions']
                    if isinstance(emotions, dict):
                        metadata['detected_emotions'] = list(emotions.keys())
    except Exception as e:
        logger.error(f"Error getting conversation metadata: {e}")
        # Continue without metadata if there's an error

    # Create new entry
    new_entry = {
        'user': user_message,
        'bot': bot_response,
        'analysis': analysis,
        'sentiment_tags': sentiment_tags,
        'timestamp': timestamp,
        'crisis_level': crisis_level
    }

    # Only add metadata if we have some
    if metadata:
        new_entry['metadata'] = metadata

    # Add to chat history
    chats.append(new_entry)

    # Save updated chats
    try:
        with open(chat_file, 'w') as f:
            json.dump(chats, f, indent=2)
        logger.info(f"Successfully saved chat file with {len(chats)} entries")

        # Verify the save worked
        if os.path.exists(chat_file) and os.path.getsize(chat_file) > 0:
            try:
                with open(chat_file, 'r') as f:
                    test_read = json.load(f)
                logger.info(f"Verification successful: read {len(test_read)} entries")
            except Exception as e:
                logger.error(f"Verification failed: {e}")
    except Exception as e:
        logger.error(f"Error saving chat file: {e}")


def save_crisis_alert(user_id, message, bot_response, analysis, crisis_level):
    """Save a crisis alert when high or urgent crisis level is detected"""
    logger.info(f"Saving crisis alert for user {user_id} with level {crisis_level}")

    # Only save alerts for medium, high, or urgent crisis levels
    if crisis_level not in ['medium', 'high', 'urgent']:
        return

    # Ensure the crisis_alerts directory exists
    if not os.path.exists('crisis_alerts'):
        os.makedirs('crisis_alerts')

    # Create unique ID for this alert
    alert_id = f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    alert_file = f"crisis_alerts/{alert_id}.json"

    # Format the alert data
    alert_data = {
        'user_id': user_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'level': crisis_level,
        'message': message,
        'response': bot_response,
        'analysis': analysis,
        'status': 'new',  # new, reviewed, resolved, escalated
        'notes': ''
    }

    # Save the alert
    try:
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)
        logger.info(f"Successfully saved crisis alert to {alert_file}")
    except Exception as e:
        logger.error(f"Error saving crisis alert: {e}")


@app.route('/')
def home():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

    # Check if Ollama API is available
    api_available, api_message = check_ollama_api()

    return render_template('index.html', api_available=api_available, api_message=api_message)

@app.route('/new_chat')
def new_chat():
    """Start a new chat session"""
    session['user_id'] = str(uuid.uuid4())
    logger.info(f"Created new chat session with user_id: {session['user_id']}")
    return redirect(url_for('home'))


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    user_id = session.get('user_id', str(uuid.uuid4()))
    logger.info(f"Chat request from user {user_id}: {user_message[:30]}...")

    def generate():
        return generate_response_stream(user_message, user_id)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/get_chat_history')
def get_chat_history():
    user_id = session.get('user_id')
    if not user_id:
        logger.warning("No user_id in session for chat history request")
        return jsonify([])

    chat_file = f'chats/{user_id}.json'
    if os.path.exists(chat_file):
        try:
            with open(chat_file, 'r') as f:
                chats = json.load(f)
            logger.info(f"Retrieved {len(chats)} chat entries for user {user_id}")

            # Strip private analysis from the returned data but keep crisis level
            for chat in chats:
                if 'analysis' in chat:
                    del chat['analysis']
                if 'sentiment_tags' in chat:
                    del chat['sentiment_tags']
            return jsonify(chats)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse chat file: {e}")
            return jsonify([])

    logger.info(f"No chat history found for user {user_id}")
    return jsonify([])


# New endpoint to get active crisis alerts
@app.route('/api/crisis_alerts', methods=['GET'])
def get_crisis_alerts():
    """Get list of all crisis alerts"""
    # In a real app, you might want to add authentication here

    alerts = []
    if os.path.exists('crisis_alerts'):
        for filename in os.listdir('crisis_alerts'):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join('crisis_alerts', filename), 'r') as f:
                        alert = json.load(f)

                        # Generate an ID for the alert based on filename
                        # Strip the .json and use that as the ID
                        alert_id = filename.replace('.json', '')

                        alerts.append({
                            'id': alert_id,
                            'timestamp': alert.get('timestamp', 'Unknown'),
                            'level': alert.get('level', 'unknown'),
                            'user_id': alert.get('user_id', 'unknown'),
                            'message_preview': alert.get('message', '')[:100] + '...' if len(
                                alert.get('message', '')) > 100 else alert.get('message', ''),
                            'status': alert.get('status', 'new')
                        })
                except Exception as e:
                    logger.error(f"Error reading crisis alert {filename}: {e}")

    # Sort by timestamp (newest first)
    alerts.sort(key=lambda x: x['timestamp'], reverse=True)

    return jsonify(alerts)


# Mark a crisis alert as reviewed
@app.route('/crisis_alerts/<alert_id>/update', methods=['POST'])
def update_crisis_alert(alert_id):
    status = request.json.get('status', '')
    notes = request.json.get('notes', '')

    # Find the alert file - handle different possible filename formats
    alert_file = None
    if os.path.exists('crisis_alerts'):
        for filename in os.listdir('crisis_alerts'):
            # Check if the alert_id is contained in the filename
            if alert_id in filename and filename.endswith('.json'):
                alert_file = os.path.join('crisis_alerts', filename)
                break

    if not alert_file:
        return jsonify({"error": "Alert not found"}), 404

    try:
        with open(alert_file, 'r') as f:
            alert = json.load(f)

        # Update the alert
        alert['status'] = status
        alert['notes'] = notes
        alert['reviewed_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert['reviewed_by'] = session.get('user_id', 'unknown')

        with open(alert_file, 'w') as f:
            json.dump(alert, f, indent=2)

        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error updating crisis alert: {e}")
        return jsonify({"error": str(e)}), 500


# Add route to restore conversation from a specific chat file
@app.route('/restore_chat/<filename>')
def restore_chat(filename):
    # Security check - only allow filenames that are valid UUIDs
    try:
        uuid_obj = uuid.UUID(filename.replace('.json', ''))
        valid_filename = f"{uuid_obj}.json"
    except ValueError:
        return jsonify({"error": "Invalid filename"}), 400

    chat_file = f'chats/{valid_filename}'

    if not os.path.exists(chat_file):
        return jsonify({"error": "Chat history not found"}), 404

    try:
        with open(chat_file, 'r') as f:
            chats = json.load(f)

        # Set this as the active session
        session['user_id'] = filename.replace('.json', '')
        logger.info(f"Restored chat session from {filename} with {len(chats)} messages")

        return jsonify({"success": True, "message_count": len(chats)})
    except Exception as e:
        logger.error(f"Error restoring chat: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/conversation_summary/<user_id>', methods=['GET'])
def get_conversation_summary(user_id):
    # In a real app, you would add authentication here
    try:
        summary = conversation_tracker.generate_conversation_summary(user_id)
        return jsonify({"summary": summary})
    except Exception as e:
        logger.error(f"Error generating conversation summary: {e}")
        return jsonify({"error": str(e)}), 500
@app.route('/available_chats')
def available_chats():
    chat_files = []

    if os.path.exists('chats'):
        for filename in os.listdir('chats'):
            if filename.endswith('.json'):
                try:
                    chat_path = os.path.join('chats', filename)
                    with open(chat_path, 'r') as f:
                        chats = json.load(f)

                    # Extract relevant summary info
                    first_message = chats[0]['user'] if chats else "No messages"
                    last_message = chats[-1]['user'] if chats else "No messages"
                    timestamp = chats[-1]['timestamp'] if chats else "Unknown"

                    # Check if any messages have crisis flags
                    has_crisis = any(chat.get('crisis_level') in ['high', 'urgent', 'medium'] for chat in chats)

                    chat_files.append({
                        'id': filename.replace('.json', ''),
                        'filename': filename,
                        'message_count': len(chats),
                        'first_message': first_message[:50] + ('...' if len(first_message) > 50 else ''),
                        'last_message': last_message[:50] + ('...' if len(last_message) > 50 else ''),
                        'last_updated': timestamp,
                        'has_crisis': has_crisis
                    })
                except Exception as e:
                    logger.error(f"Error processing chat file {filename}: {e}")

    return jsonify(chat_files)


@app.route('/stats')
def stats():
    # Access to the stats page is public in this application
    return render_template('stats.html')


@app.route('/services')
def services():
    # Access to the services page is public in this application
    return render_template('services.html')


# Add crisis monitoring dashboard
@app.route('/crisis_monitor')
def crisis_monitor():
    # In a real app, you would add authentication here
    return render_template('crisis_monitor.html')


# Add a debug route to test conversation history and analysis
@app.route('/debug_analysis')
def debug_analysis():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "No active user session"}), 400

    chat_file = f'chats/{user_id}.json'
    if not os.path.exists(chat_file):
        return jsonify({"error": "No chat history found"}), 404

    try:
        with open(chat_file, 'r') as f:
            chats = json.load(f)

        # Extract analyses for debugging
        analyses = []
        for idx, chat in enumerate(chats):
            analyses.append({
                "index": idx,
                "user_message": chat['user'],
                "has_analysis": 'analysis' in chat,
                "analysis_length": len(chat.get('analysis', '')) if 'analysis' in chat else 0,
                "analysis_sample": chat.get('analysis', '')[:100] + '...' if 'analysis' in chat and len(
                    chat.get('analysis', '')) > 100 else chat.get('analysis', ''),
                "sentiment_tags": chat.get('sentiment_tags', 'None'),
                "crisis_level": chat.get('crisis_level', 'none')
            })

        return jsonify({
            "total_messages": len(chats),
            "messages_with_analysis": sum(1 for a in analyses if a["has_analysis"]),
            "crisis_messages": sum(1 for a in analyses if a.get("crisis_level") in ["medium", "high", "urgent"]),
            "analyses": analyses
        })
    except Exception as e:
        logger.error(f"Error in debug_analysis: {e}")
        return jsonify({"error": str(e)}), 500


# Add these new imports at the top of app.py
from collections import Counter
from operator import itemgetter
from flask import send_from_directory


# Add these new routes to app.py

@app.route('/crisis_dashboard')
def crisis_dashboard():
    """Dashboard to monitor crisis alerts and high-risk conversations"""
    # In a real app, you would add authentication here
    return render_template('crisis_dashboard.html')


@app.route('/api/crisis_overview')
def get_crisis_overview():
    """Get overview statistics about crisis alerts"""
    # In a real app, add authentication check here

    # Statistics we'll gather
    stats = {
        'total_alerts': 0,
        'new_alerts': 0,
        'urgent_alerts': 0,
        'high_alerts': 0,
        'medium_alerts': 0,
        'alerts_by_day': {},
        'recent_alerts': []
    }

    if os.path.exists('crisis_alerts'):
        alerts = []
        # Gather all alerts
        for filename in os.listdir('crisis_alerts'):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join('crisis_alerts', filename), 'r') as f:
                        alert = json.load(f)
                        alerts.append(alert)

                        # Update basic counts
                        stats['total_alerts'] += 1

                        if alert.get('status') == 'new':
                            stats['new_alerts'] += 1

                        level = alert.get('level', 'unknown')
                        if level == 'urgent':
                            stats['urgent_alerts'] += 1
                        elif level == 'high':
                            stats['high_alerts'] += 1
                        elif level == 'medium':
                            stats['medium_alerts'] += 1

                        # Track by day
                        timestamp = alert.get('timestamp', '')
                        if timestamp:
                            day = timestamp.split(' ')[0]  # Get just the date part
                            stats['alerts_by_day'][day] = stats['alerts_by_day'].get(day, 0) + 1
                except Exception as e:
                    logger.error(f"Error reading crisis alert {filename}: {e}")

        # Get 5 most recent alerts for the overview
        alerts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        for alert in alerts[:5]:
            stats['recent_alerts'].append({
                'timestamp': alert.get('timestamp', 'Unknown'),
                'level': alert.get('level', 'unknown'),
                'message_preview': alert.get('message', '')[:100] + '...' if len(
                    alert.get('message', '')) > 100 else alert.get('message', ''),
                'status': alert.get('status', 'new'),
                'user_id': alert.get('user_id', 'unknown')
            })

    return jsonify(stats)


@app.route('/api/at_risk_users')
def get_at_risk_users():
    """Get users who have shown crisis indicators"""
    # In a real app, add authentication check here

    at_risk_users = []
    high_risk_users = []

    if os.path.exists('chats'):
        for filename in os.listdir('chats'):
            if filename.endswith('.json'):
                try:
                    user_id = filename.replace('.json', '')
                    chat_path = os.path.join('chats', filename)

                    with open(chat_path, 'r') as f:
                        chats = json.load(f)

                    # Check for crisis messages
                    crisis_messages = [chat for chat in chats if
                                       chat.get('crisis_level') in ['medium', 'high', 'urgent']]

                    if crisis_messages:
                        # Get last message timestamp
                        last_timestamp = chats[-1].get('timestamp', 'Unknown')

                        # Get highest crisis level
                        crisis_levels = [chat.get('crisis_level') for chat in crisis_messages]
                        highest_level = max(crisis_levels,
                                            key=lambda x: {'urgent': 3, 'high': 2, 'medium': 1}.get(x, 0))

                        # Count crisis messages by level
                        level_counts = Counter(crisis_levels)

                        user_data = {
                            'user_id': user_id,
                            'message_count': len(chats),
                            'crisis_message_count': len(crisis_messages),
                            'highest_crisis_level': highest_level,
                            'urgent_count': level_counts.get('urgent', 0),
                            'high_count': level_counts.get('high', 0),
                            'medium_count': level_counts.get('medium', 0),
                            'last_message_time': last_timestamp,
                            'last_message': chats[-1].get('user', '')[:100] + '...' if len(
                                chats[-1].get('user', '')) > 100 else chats[-1].get('user', '')
                        }

                        # Separate into high risk (urgent or high) and medium risk
                        if highest_level in ['urgent', 'high']:
                            high_risk_users.append(user_data)
                        else:
                            at_risk_users.append(user_data)

                except Exception as e:
                    logger.error(f"Error processing chat file {filename}: {e}")

    # Sort by last message time (most recent first)
    high_risk_users.sort(key=itemgetter('last_message_time'), reverse=True)
    at_risk_users.sort(key=itemgetter('last_message_time'), reverse=True)

    return jsonify({
        'high_risk_users': high_risk_users,
        'at_risk_users': at_risk_users
    })


@app.route('/api/user_timeline/<user_id>')
def get_user_timeline(user_id):
    """Get detailed timeline for a specific user"""
    # In a real app, add authentication check here

    # Security check - only allow valid UUIDs
    try:
        uuid_obj = uuid.UUID(user_id)
    except ValueError:
        return jsonify({"error": "Invalid user ID"}), 400

    chat_file = f'chats/{user_id}.json'
    if not os.path.exists(chat_file):
        return jsonify({"error": "User chat history not found"}), 404

    try:
        with open(chat_file, 'r') as f:
            chats = json.load(f)

        # For better frontend rendering, process the chat data
        timeline_data = []
        for idx, chat in enumerate(chats):
            entry = {
                'id': idx,
                'timestamp': chat.get('timestamp', 'Unknown'),
                'user_message': chat.get('user', ''),
                'bot_response': chat.get('bot', ''),
                'crisis_level': chat.get('crisis_level', 'none'),
                'analysis': chat.get('analysis', '') if 'analysis' in chat else '',
                'sentiment_tags': chat.get('sentiment_tags', [])
            }
            timeline_data.append(entry)

        # Get user summary from conversation tracker
        user_summary = {}
        try:
            user_state = conversation_tracker.get_user_state(user_id)
            if user_state:
                # Add useful summary info
                if 'dominant_emotions' in user_state:
                    user_summary['emotions'] = [e['emotion'] for e in user_state['dominant_emotions'][:5]]

                if 'topics_discussed' in user_state:
                    topics = sorted(user_state['topics_discussed'].items(), key=lambda x: x[1], reverse=True)[:5]
                    user_summary['topics'] = [t[0] for t in topics]

                if 'conversation_quality' in user_state:
                    user_summary['conversation_quality'] = user_state['conversation_quality']
        except Exception as e:
            logger.error(f"Error getting user summary: {e}")

        return jsonify({
            'user_id': user_id,
            'message_count': len(chats),
            'timeline': timeline_data,
            'user_summary': user_summary
        })

    except Exception as e:
        logger.error(f"Error processing user timeline: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/crisis_alerts/details/<alert_id>')
def get_crisis_alert_details(alert_id):
    """Get detailed information for a specific crisis alert"""
    # In a real app, add authentication check here

    # Find the alert file - handle different possible filename formats
    alert_file = None
    if os.path.exists('crisis_alerts'):
        for filename in os.listdir('crisis_alerts'):
            # Check if the alert_id is contained in the filename
            if alert_id in filename and filename.endswith('.json'):
                alert_file = os.path.join('crisis_alerts', filename)
                break

    if not alert_file:
        return jsonify({"error": "Alert not found"}), 404

    try:
        with open(alert_file, 'r') as f:
            alert = json.load(f)

        # Get related chat history
        user_id = alert.get('user_id', '')
        related_messages = []

        if user_id and os.path.exists(f'chats/{user_id}.json'):
            try:
                with open(f'chats/{user_id}.json', 'r') as f:
                    chats = json.load(f)

                # Find index of the crisis message
                crisis_time = alert.get('timestamp', '')
                crisis_message = alert.get('message', '')

                for idx, chat in enumerate(chats):
                    if chat.get('timestamp') == crisis_time and chat.get('user') == crisis_message:
                        # Get messages before and after (context)
                        start_idx = max(0, idx - 3)
                        end_idx = min(len(chats), idx + 3)
                        context_chats = chats[start_idx:end_idx]

                        for c in context_chats:
                            related_messages.append({
                                'timestamp': c.get('timestamp', ''),
                                'user_message': c.get('user', ''),
                                'bot_response': c.get('bot', ''),
                                'is_crisis_message': c.get('timestamp') == crisis_time,
                                'crisis_level': c.get('crisis_level', 'none')
                            })
                        break
            except Exception as e:
                logger.error(f"Error getting related messages: {e}")

        return jsonify({
            'alert': alert,
            'related_messages': related_messages
        })

    except Exception as e:
        logger.error(f"Error getting crisis alert details: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/download/chat/<user_id>')
def download_chat_history(user_id):
    """Download a user's chat history as a JSON file"""
    # In a real app, add authentication check here

    try:
        uuid_obj = uuid.UUID(user_id)
    except ValueError:
        return jsonify({"error": "Invalid user ID"}), 400

    chat_file = f'chats/{user_id}.json'
    if not os.path.exists(chat_file):
        return jsonify({"error": "User chat history not found"}), 404

    return send_from_directory('chats', f'{user_id}.json', as_attachment=True)


@app.route('/user_profile/<user_id>')
def user_profile(user_id):
    """Page for viewing a specific user's profile and chat history"""
    # In a real app, add authentication check here

    # Validate user ID
    try:
        uuid_obj = uuid.UUID(user_id)
    except ValueError:
        return "Invalid user ID", 400

    return render_template('user_profile.html', user_id=user_id)


if __name__ == '__main__':
    logger.info("Starting DeepTalker app")
    app.run(debug=True)