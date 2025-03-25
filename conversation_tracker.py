# Add this to a new file named conversation_tracker.py

import logging
import json
import os
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
import re
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class ConversationTracker:
    def __init__(self):
        """Initialize the conversation tracker"""
        self.user_states = {}  # Track state by user_id

        # Initialize topic categories and emotional indicators
        self.topics = {
            'family': ['family', 'parent', 'mother', 'father', 'mom', 'dad', 'sister', 'brother', 'daughter', 'son',
                       'child', 'children'],
            'relationships': ['relationship', 'partner', 'boyfriend', 'girlfriend', 'spouse', 'husband', 'wife',
                              'dating', 'marriage', 'divorce'],
            'work': ['work', 'job', 'career', 'coworker', 'boss', 'workplace', 'office', 'unemployed', 'fired',
                     'promotion', 'stress'],
            'education': ['school', 'college', 'university', 'class', 'student', 'study', 'exam', 'teacher',
                          'professor', 'homework', 'grades'],
            'health': ['health', 'sick', 'illness', 'doctor', 'hospital', 'pain', 'medication', 'disease', 'treatment',
                       'diagnosis', 'symptom'],
            'mental_health': ['anxiety', 'depression', 'stress', 'therapy', 'therapist', 'psychiatrist', 'medication',
                              'mental health', 'breakdown', 'overwhelmed'],
            'grief_loss': ['grief', 'loss', 'death', 'died', 'funeral', 'bereavement', 'mourning', 'passed away',
                           'miss', 'gone'],  # Renamed to avoid conflict
            'identity': ['identity', 'self-esteem', 'confidence', 'who am i', 'purpose', 'meaning', 'future', 'goal',
                         'direction'],
            'trauma': ['trauma', 'abuse', 'assault', 'violence', 'accident', 'ptsd', 'flashback', 'trigger',
                       'nightmare'],
            'substance': ['alcohol', 'drug', 'substance', 'addiction', 'recovery', 'sober', 'relapse', 'withdrawal',
                          'drinking']
        }

        self.emotional_indicators = {
            'joy': ['happy', 'joy', 'excited', 'pleased', 'delighted', 'cheerful', 'content', 'glad', 'thrilled',
                    'elated'],
            'sadness': ['sad', 'unhappy', 'depressed', 'gloomy', 'heartbroken', 'down', 'blue', 'upset', 'miserable'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated', 'rage', 'hate', 'resent',
                      'bitter'],
            'fear': ['afraid', 'scared', 'fearful', 'terrified', 'anxious', 'worried', 'panic', 'dread', 'frightened',
                     'nervous'],
            'disgust': ['disgusted', 'repulsed', 'revolted', 'sickened', 'appalled', 'horrified', 'loathing'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'startled', 'unexpected'],
            'shame': ['ashamed', 'guilty', 'embarrassed', 'humiliated', 'remorseful', 'regretful', 'shameful'],
            'hopelessness': ['hopeless', 'helpless', 'worthless', 'despair', 'desperate', 'defeated', 'given up'],
            'loneliness': ['lonely', 'alone', 'isolated', 'abandoned', 'rejected', 'unwanted', 'disconnected',
                           'solitary'],
            'gratitude': ['grateful', 'thankful', 'appreciative', 'blessed', 'fortunate', 'appreciate', 'thanks'],
            'hope': ['hopeful', 'optimistic', 'encouraged', 'positive', 'motivated', 'inspired', 'determined'],
            'grief_emotion': ['grief', 'grieving', 'mourning', 'bereavement', 'missing', 'loss']
            # Added as separate emotion category
        }

        # Ensure the storage directory exists
        if not os.path.exists('user_states'):
            os.makedirs('user_states')

    def get_user_state(self, user_id, load_from_disk=True):
        """
        Get current state for user, create if not exists

        Args:
            user_id: Unique user identifier
            load_from_disk: Whether to load state from disk if not in memory

        Returns:
            dict: User state dictionary
        """
        # Check if we already have the state in memory
        if user_id not in self.user_states:
            # Try to load from disk if requested
            if load_from_disk:
                state_path = f'user_states/{user_id}.json'
                if os.path.exists(state_path):
                    try:
                        with open(state_path, 'r') as f:
                            self.user_states[user_id] = json.load(f)
                            logger.info(f"Loaded user state from disk for user {user_id}")
                    except Exception as e:
                        logger.error(f"Error loading user state: {e}")
                        # Create new state if loading fails
                        self._initialize_user_state(user_id)
                else:
                    # Create new state if file doesn't exist
                    self._initialize_user_state(user_id)
            else:
                # Create new state without checking disk
                self._initialize_user_state(user_id)

        return self.user_states[user_id]

    def _initialize_user_state(self, user_id):
        """Create a new user state with default values"""
        self.user_states[user_id] = {
            'user_id': user_id,
            'first_session': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'last_session': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'session_count': 1,
            'total_messages': 0,
            'emotional_states': [],
            'dominant_emotions': [],
            'topics_discussed': defaultdict(int),
            'crisis_history': [],
            'key_concerns': set(),
            'protective_factors': set(),
            'progress_indicators': {
                'emotional_improvement': 0,
                'engagement_level': 0,
                'coping_strategies_mentioned': 0,
                'help_seeking_behavior': 0
            },
            'risk_factors': {
                'suicidal_ideation': False,
                'self_harm': False,
                'isolation': False,
                'hopelessness': False,
                'substance_abuse': False
            },
            'conversation_quality': {
                'user_engagement': 0,  # 0-100 scale
                'depth': 0,  # 0-100 scale
                'emotional_disclosure': 0  # 0-100 scale
            }
        }
        logger.info(f"Initialized new user state for user {user_id}")

    def save_user_state(self, user_id):
        """Save user state to disk"""
        if user_id in self.user_states:
            state_path = f'user_states/{user_id}.json'
            try:
                # Convert defaultdict to regular dict for serialization
                state_copy = self.user_states[user_id].copy()
                if 'topics_discussed' in state_copy and isinstance(state_copy['topics_discussed'], defaultdict):
                    state_copy['topics_discussed'] = dict(state_copy['topics_discussed'])

                # Convert sets to lists for serialization
                if 'key_concerns' in state_copy and isinstance(state_copy['key_concerns'], set):
                    state_copy['key_concerns'] = list(state_copy['key_concerns'])

                if 'protective_factors' in state_copy and isinstance(state_copy['protective_factors'], set):
                    state_copy['protective_factors'] = list(state_copy['protective_factors'])

                with open(state_path, 'w') as f:
                    json.dump(state_copy, f, indent=2)
                logger.info(f"Saved user state to disk for user {user_id}")
            except Exception as e:
                logger.error(f"Error saving user state: {e}")

    def update_with_message(self, user_id, message, response, analysis, sentiment_tags, crisis_level):
        """
        Update user state with new message information

        Args:
            user_id: Unique user identifier
            message: User's message text
            response: Bot's response text
            analysis: Analysis text provided by the LLM
            sentiment_tags: Sentiment tags from classifier
            crisis_level: Crisis level detected (none, low, medium, high, urgent)

        Returns:
            dict: Updated user state
        """
        # Get current state
        state = self.get_user_state(user_id)

        # Update basic counters
        state['total_messages'] += 1
        state['last_session'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Process and tokenize message
        message_lower = message.lower()
        tokens = word_tokenize(message_lower)

        # Extract emotions
        current_emotions = self._extract_emotions(message_lower)

        # Log emotions for tracking emotional trajectory
        emotion_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'emotions': current_emotions,
            'message_idx': state['total_messages'],
            'crisis_level': crisis_level
        }
        state['emotional_states'].append(emotion_entry)

        # Limit the history to the last 20 entries
        if len(state['emotional_states']) > 20:
            state['emotional_states'] = state['emotional_states'][-20:]

        # Update dominant emotions (aggregate of last 5 messages)
        recent_emotions = [e['emotions'] for e in state['emotional_states'][-5:]]
        all_emotions = []
        for emotion_set in recent_emotions:
            all_emotions.extend(emotion_set.keys())

        # Count occurrences of each emotion
        emotion_counts = Counter(all_emotions)

        # Get the 3 most common emotions with their counts
        dominant = emotion_counts.most_common(3)
        state['dominant_emotions'] = [{'emotion': e, 'count': c} for e, c in dominant]

        # Extract topics from message
        detected_topics = self._extract_topics(message_lower)
        for topic in detected_topics:
            state['topics_discussed'][topic] += 1

        # Track crisis information if applicable
        if crisis_level not in ['none', 'low']:
            crisis_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'message': message,
                'level': crisis_level,
                'message_idx': state['total_messages']
            }
            state['crisis_history'].append(crisis_entry)

            # Update risk factors based on crisis content
            if crisis_level in ['high', 'urgent']:
                # Perform more detailed risk analysis
                if any(term in message_lower for term in ['suicide', 'kill myself', 'end my life', 'better off dead']):
                    state['risk_factors']['suicidal_ideation'] = True

                if any(term in message_lower for term in
                       ['cut', 'harm myself', 'hurt myself', 'self-harm', 'burning', 'hitting']):
                    state['risk_factors']['self_harm'] = True

                if any(term in message_lower for term in
                       ['alone', 'lonely', 'no one cares', 'no support', 'by myself']):
                    state['risk_factors']['isolation'] = True

                if any(term in message_lower for term in
                       ['hopeless', 'no point', 'never get better', 'pointless', 'no future']):
                    state['risk_factors']['hopelessness'] = True

                if any(term in message_lower for term in ['drunk', 'drinking', 'alcohol', 'drugs', 'high', 'wasted']):
                    state['risk_factors']['substance_abuse'] = True

        # Extract key concerns from the analysis if available
        if analysis:
            # Look for key concerns in the analysis
            concern_match = re.search(r'Key concerns:?\s*([^.]*)', analysis)
            if concern_match:
                concerns = concern_match.group(1).split(',')
                for concern in concerns:
                    clean_concern = concern.strip()
                    if clean_concern and not clean_concern.lower() in ['none', 'n/a', 'not applicable']:
                        state['key_concerns'].add(clean_concern)

            # Look for protective factors
            protective_match = re.search(r'Protective factors identified:?\s*([^.]*)', analysis)
            if protective_match:
                factors = protective_match.group(1).split(',')
                for factor in factors:
                    clean_factor = factor.strip()
                    if clean_factor and not clean_factor.lower() in ['none', 'n/a', 'not applicable',
                                                                     'no protective factors']:
                        state['protective_factors'].add(clean_factor)

        # Update conversation quality metrics
        self._update_conversation_quality(state, message, response)

        # Update progress indicators
        self._update_progress_indicators(state, message, response, analysis)

        # Save the updated state
        self.save_user_state(user_id)

        return state

    def _extract_emotions(self, text):
        """
        Extract emotions from text based on emotional indicators

        Returns:
            dict: Dictionary of emotions and their strength (count of indicators)
        """
        emotions = {}

        for emotion, indicators in self.emotional_indicators.items():
            # Count occurrences of emotion indicators
            count = 0
            for indicator in indicators:
                if indicator in text:
                    count += 1

            if count > 0:
                emotions[emotion] = count

        return emotions

    def _extract_topics(self, text):
        """Extract topics from text based on topic indicators"""
        detected_topics = []

        for topic, indicators in self.topics.items():
            # Check if any indicator is present in the text
            if any(indicator in text for indicator in indicators):
                detected_topics.append(topic)

        return detected_topics

    def _update_conversation_quality(self, state, message, response):
        """Update conversation quality metrics"""
        # Simple heuristics for conversation quality
        # Message length as a proxy for engagement (capped at 100)
        message_length = min(len(message.split()), 100)
        engagement_score = (message_length / 100) * 100

        # Emotional disclosure - number of emotional terms
        emotion_terms = sum(1 for emotion, terms in self.emotional_indicators.items()
                            for term in terms if term in message.lower())
        emotion_score = min(emotion_terms * 20, 100)  # Scale up, cap at 100

        # Depth - presence of introspective terms or questions
        introspection_terms = ['feel', 'think', 'realize', 'understand', 'question', 'wonder',
                               'confused', 'uncertain', 'meaning', 'purpose', 'learned', 'why']
        depth_indicators = sum(1 for term in introspection_terms if term in message.lower())
        question_count = message.count('?')
        depth_score = min((depth_indicators * 15) + (question_count * 10), 100)  # Scale up, cap at 100

        # Update with moving average (70% previous, 30% new)
        state['conversation_quality']['user_engagement'] = (
                state['conversation_quality']['user_engagement'] * 0.7 + engagement_score * 0.3
        )
        state['conversation_quality']['emotional_disclosure'] = (
                state['conversation_quality']['emotional_disclosure'] * 0.7 + emotion_score * 0.3
        )
        state['conversation_quality']['depth'] = (
                state['conversation_quality']['depth'] * 0.7 + depth_score * 0.3
        )

    def _update_progress_indicators(self, state, message, response, analysis):
        """Update progress indicators based on message content and analysis"""
        message_lower = message.lower()

        # Check for emotional improvement indicators
        improvement_indicators = [
            'better', 'improving', 'progress', 'hope', 'hopeful', 'positive',
            'getting better', 'improved', 'feel good', 'happier', 'healing',
            'learning', 'growth', 'moving forward', 'optimistic'
        ]

        improvement_mentions = sum(1 for term in improvement_indicators if term in message_lower)

        # Check for coping strategy mentions
        coping_indicators = [
            'therapy', 'therapist', 'counseling', 'meditation', 'exercise', 'journal',
            'self-care', 'breathing', 'grounding', 'mindfulness', 'support group',
            'coping', 'techniques', 'strategies', 'skills', 'routine', 'sleep',
            'healthy', 'boundaries', 'saying no', 'taking time'
        ]

        coping_mentions = sum(1 for term in coping_indicators if term in message_lower)

        # Check for help-seeking behavior
        help_indicators = [
            'appointment', 'doctor', 'therapist', 'counselor', 'psychologist', 'psychiatrist',
            'medication', 'prescription', 'professional', 'help', 'clinic', 'treatment',
            'program', 'support', 'resources', 'referral', 'hotline', 'call',
            'reach out', 'talked to', 'asking for help'
        ]

        help_mentions = sum(1 for term in help_indicators if term in message_lower)

        # Update progress indicators with gradual changes
        # Each mention adds 5 points, up to max of 100, decays by 2 points per message
        state['progress_indicators']['emotional_improvement'] = min(
            max(state['progress_indicators']['emotional_improvement'] - 2, 0) + (improvement_mentions * 5),
            100
        )

        state['progress_indicators']['coping_strategies_mentioned'] = min(
            max(state['progress_indicators']['coping_strategies_mentioned'] - 2, 0) + (coping_mentions * 5),
            100
        )

        state['progress_indicators']['help_seeking_behavior'] = min(
            max(state['progress_indicators']['help_seeking_behavior'] - 2, 0) + (help_mentions * 5),
            100
        )

        # Update engagement based on message frequency and length
        # This is a simple heuristic - could be more sophisticated
        engagement_boost = min(len(message.split()) / 10, 5)  # Cap at 5 points per message
        state['progress_indicators']['engagement_level'] = min(
            max(state['progress_indicators']['engagement_level'] - 1, 0) + engagement_boost,
            100
        )

    def generate_conversation_summary(self, user_id):
        """
        Generate a summary of the conversation history

        Args:
            user_id: Unique user identifier

        Returns:
            str: Summary text
        """
        state = self.get_user_state(user_id)

        # Calculate total session duration
        first_session = datetime.strptime(state['first_session'], "%Y-%m-%d %H:%M:%S")
        last_session = datetime.strptime(state['last_session'], "%Y-%m-%d %H:%M:%S")
        duration_days = (last_session - first_session).days

        # Format dominant emotions
        emotions_text = ", ".join([f"{e['emotion']} ({e['count']})" for e in state['dominant_emotions']]) if state[
            'dominant_emotions'] else "None detected"

        # Format top topics
        topics = sorted(state['topics_discussed'].items(), key=lambda x: x[1], reverse=True)
        topics_text = ", ".join([f"{topic} ({count})" for topic, count in topics[:5]]) if topics else "None detected"

        # Format crisis history
        crisis_count = len(state['crisis_history'])
        last_crisis = state['crisis_history'][-1]['timestamp'] if crisis_count > 0 else "None"
        highest_level = max([c['level'] for c in state['crisis_history']],
                            key=lambda x: {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'urgent': 4}.get(x,
                                                                                                         0)) if crisis_count > 0 else "None"

        # Format risk factors
        active_risks = [risk for risk, active in state['risk_factors'].items() if active]
        risks_text = ", ".join(active_risks) if active_risks else "None detected"

        # Format progress indicators
        progress = state['progress_indicators']

        summary = f"""CONVERSATION SUMMARY FOR USER {user_id}:

Total duration: {duration_days} days ({state['first_session']} to {state['last_session']})
Total messages: {state['total_messages']}
Session count: {state['session_count']}

EMOTIONAL PROFILE:
- Dominant emotions: {emotions_text}
- Emotional disclosure level: {state['conversation_quality']['emotional_disclosure']:.1f}/100
- Emotional improvement indicators: {progress['emotional_improvement']:.1f}/100

TOPICS & CONCERNS:
- Most discussed topics: {topics_text}
- Key concerns: {', '.join(state['key_concerns']) if state['key_concerns'] else 'None identified'}
- Conversation depth: {state['conversation_quality']['depth']:.1f}/100

RISK ASSESSMENT:
- Crisis events: {crisis_count}
- Most recent crisis: {last_crisis}
- Highest crisis level: {highest_level}
- Active risk factors: {risks_text}

PROGRESS INDICATORS:
- Engagement level: {progress['engagement_level']:.1f}/100
- Coping strategies mentioned: {progress['coping_strategies_mentioned']:.1f}/100
- Help-seeking behavior: {progress['help_seeking_behavior']:.1f}/100

PROTECTIVE FACTORS:
- {', '.join(state['protective_factors']) if state['protective_factors'] else 'None identified'}
"""
        return summary

    def extract_analysis_from_bot_response(self, response):
        """
        Extract the analysis section from a bot response

        Args:
            response: Full bot response text

        Returns:
            tuple: (visible_response, analysis)
        """
        if "ANALYSIS:" in response:
            parts = response.split("ANALYSIS:", 1)
            visible_response = parts[0].strip()
            analysis = "ANALYSIS: " + parts[1].strip()
            return visible_response, analysis
        else:
            return response, None

    def generate_fallback_analysis(self, message, response, sentiment_tags):
        """
        Generate a fallback analysis when model doesn't provide one

        Args:
            message: User's message
            response: Bot's response
            sentiment_tags: Sentiment tags

        Returns:
            str: Generated analysis
        """
        # Extract emotions
        emotions = self._extract_emotions(message.lower())
        emotions_text = ", ".join(emotions.keys()) if emotions else "None detected"

        # Extract topics
        topics = self._extract_topics(message.lower())
        topics_text = ", ".join(topics) if topics else "General conversation"

        # Detect crisis indicators
        crisis_indicators = [
            "suicide", "kill myself", "die", "end my life", "hurt myself",
            "self-harm", "cut myself", "overdose", "pills", "gun",
            "hopeless", "worthless", "burden", "don't want to live"
        ]

        risk_level = "Low"
        message_lower = message.lower()

        if any(indicator in message_lower for indicator in crisis_indicators):
            risk_level = "Medium"

            # Check for more severe indicators
            urgent_indicators = [
                "going to kill myself", "today", "tonight", "now", "goodbye",
                "farewell", "suicide note", "final message", "end it all",
                "pills ready", "gun loaded", "about to jump"
            ]

            if any(indicator in message_lower for indicator in urgent_indicators):
                risk_level = "High"

        # Check if the bot response indicates crisis
        if "concerned about your safety" in response.lower() or "988" in response or "crisis" in response.lower():
            risk_level = "High"

        # Generate the analysis
        analysis = f"""ANALYSIS:
IMMEDIATE SAFETY RISK ASSESSMENT: {risk_level}
Warning signs identified: {'Crisis language detected in message' if risk_level != 'Low' else 'No explicit warning signs'}
Protective factors identified: None explicitly mentioned
Emotional states: {emotions_text}
Key concerns: {topics_text}
Risk factors: {'Crisis language present' if risk_level != 'Low' else 'No significant risk factors detected'}
Interpretation of sentiment tags: {sentiment_tags}
Recommended approach: {'Focus on safety and emotional support' if risk_level != 'Low' else 'Continue supportive conversation'}
Changes from previous states: Insufficient context to determine
Progress/regression: Insufficient context to determine
"""
        return analysis