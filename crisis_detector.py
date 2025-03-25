import logging
import os
import json
from datetime import datetime
import uuid
import re
import numpy as np
from flask import session
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Configure logging
logger = logging.getLogger(__name__)


class CrisisDetector:
    def __init__(self, model_path=None):
        """
        Initialize the crisis detector with risk lexicons and optional model

        Args:
            model_path: Path to a trained ML model file (optional)
        """
        # Ensure the crisis_alerts directory exists
        if not os.path.exists('crisis_alerts'):
            os.makedirs('crisis_alerts')

        # Load risk lexicons from files or create default ones
        self.load_risk_lexicons()

        # Initialize feature extraction
        self.stop_words = set(stopwords.words('english'))

        # Initialize model if provided
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded ML model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load ML model: {e}")

        # Initialize the crisis level thresholds
        self.risk_thresholds = {
            'urgent': 0.85,
            'high': 0.65,
            'medium': 0.4,
            'low': 0.15
        }

    def load_risk_lexicons(self):
        """Load risk lexicons from files or use defaults"""
        # Define paths to lexicon files
        lexicon_dir = os.path.join(os.path.dirname(__file__), 'lexicons')
        if not os.path.exists(lexicon_dir):
            os.makedirs(lexicon_dir)

        # File paths
        urgent_path = os.path.join(lexicon_dir, 'urgent_risk.json')
        high_path = os.path.join(lexicon_dir, 'high_risk.json')
        medium_path = os.path.join(lexicon_dir, 'medium_risk.json')
        crisis_response_path = os.path.join(lexicon_dir, 'crisis_responses.json')
        protective_factors_path = os.path.join(lexicon_dir, 'protective_factors.json')

        # Try to load lexicons from files, or use defaults
        try:
            if os.path.exists(urgent_path):
                with open(urgent_path, 'r') as f:
                    self.urgent_risk_phrases = json.load(f)
            else:
                # Default urgent risk phrases (imminent danger)
                self.urgent_risk_phrases = {
                    "immediate_intent": [
                        "going to kill myself", "about to end it", "this is goodbye",
                        "killing myself tonight", "ending it all today", "final message",
                        "pulling the trigger", "jumping off", "hanging myself today",
                        "taking all my pills now", "slitting my wrists", "this is the end",
                        "no longer alive after", "suicide note", "no one will find me"
                    ],
                    "specific_plan": [
                        "bought a gun", "have the pills ready", "tied the noose",
                        "standing on the bridge", "sharp blade", "collected enough pills",
                        "found the tallest building", "ready to jump", "loaded gun",
                        "poison ready", "written my note", "making preparations"
                    ],
                    "imminent_timeline": [
                        "tonight", "in the morning", "few hours", "when everyone's asleep",
                        "after this call", "right after", "as soon as", "once I hang up",
                        "before dawn", "before anyone notices", "in a few minutes"
                    ]
                }

                with open(urgent_path, 'w') as f:
                    json.dump(self.urgent_risk_phrases, f, indent=2)

            if os.path.exists(high_path):
                with open(high_path, 'r') as f:
                    self.high_risk_phrases = json.load(f)
            else:
                # Default high risk phrases (clear intent but not immediate)
                self.high_risk_phrases = {
                    "suicidal_intent": [
                        "want to die", "kill myself", "end my life", "suicide", "rather be dead",
                        "don't want to live", "no reason to live", "better off dead",
                        "planning to end", "looking up ways to die", "research methods",
                        "my suicide", "how to commit suicide", "ending things", "take my life"
                    ],
                    "means_access": [
                        "have pills", "got a gun", "rope", "lethal", "weapon", "sharp knife",
                        "my medication", "stockpiling", "gathering pills", "access to firearms",
                        "ammunition", "roof access", "high place", "bridge near", "train tracks"
                    ],
                    "preparation": [
                        "writing letters", "giving away", "saying goodbye", "final arrangements",
                        "delete my accounts", "putting affairs in order", "last wishes",
                        "will and testament", "insurance policy", "setting things up"
                    ],
                    "hopelessness": [
                        "no way out", "can't go on", "no future", "never get better",
                        "trapped", "unbearable", "can't handle this anymore", "no escape",
                        "no point anymore", "suffering too much", "too painful to continue"
                    ]
                }

                with open(high_path, 'w') as f:
                    json.dump(self.high_risk_phrases, f, indent=2)

            if os.path.exists(medium_path):
                with open(medium_path, 'r') as f:
                    self.medium_risk_phrases = json.load(f)
            else:
                # Default medium risk phrases (distress without specific intent)
                self.medium_risk_phrases = {
                    "passive_ideation": [
                        "wish I could disappear", "if I didn't wake up", "life is too hard",
                        "what's the point", "tired of living", "don't care if I die",
                        "wouldn't mind dying", "sometimes think about death",
                        "death would be easier", "don't care what happens to me"
                    ],
                    "emotional_distress": [
                        "hopeless", "worthless", "burden", "can't take it", "overwhelmed",
                        "desperate", "miserable", "can't cope", "falling apart",
                        "nothing helps", "despair", "at the end of my rope"
                    ],
                    "isolation": [
                        "nobody cares", "all alone", "no one would miss me", "no one understands",
                        "no support", "abandoned", "rejected", "isolated", "by myself",
                        "no friends", "no one to talk to", "no one notices"
                    ],
                    "burdensomeness": [
                        "everyone would be better off", "burden to others", "dragging others down",
                        "making things worse", "causing problems", "waste of space",
                        "drain on everyone", "holding others back", "doing them a favor"
                    ]
                }

                with open(medium_path, 'w') as f:
                    json.dump(self.medium_risk_phrases, f, indent=2)

            if os.path.exists(crisis_response_path):
                with open(crisis_response_path, 'r') as f:
                    self.crisis_response_indicators = json.load(f)
            else:
                # Default crisis response indicator phrases
                self.crisis_response_indicators = [
                    "concerned about your safety", "worried about you",
                    "are you planning to harm yourself", "are you thinking of suicide",
                    "are you in immediate danger", "having thoughts of suicide",
                    "please call 988", "call the crisis line", "suicide & crisis lifeline",
                    "emergency services", "go to emergency room", "nearest hospital",
                    "text home to 741741", "crisis text line", "call 911"
                ]

                with open(crisis_response_path, 'w') as f:
                    json.dump(self.crisis_response_indicators, f, indent=2)

            if os.path.exists(protective_factors_path):
                with open(protective_factors_path, 'r') as f:
                    self.protective_factors = json.load(f)
            else:
                # Protective factors that might reduce risk
                self.protective_factors = {
                    "social_support": [
                        "my family", "my friends", "therapist", "partner", "wife", "husband",
                        "children", "kids", "wouldn't do that to them", "people who care",
                        "people counting on me", "wouldn't hurt my family", "promised someone"
                    ],
                    "treatment": [
                        "medication helps", "therapy", "counseling", "doctor", "psychiatrist",
                        "treatment", "getting help", "mental health", "crisis team", "support group"
                    ],
                    "coping": [
                        "trying to cope", "working through", "one day at a time", "still fighting",
                        "not giving up", "holding on", "staying strong", "trying my best",
                        "coping skills", "safety plan", "reasons to live", "survive this"
                    ],
                    "future_oriented": [
                        "hoping things improve", "tomorrow", "future", "get through this",
                        "better days ahead", "want to see", "looking forward to", "goals",
                        "plans for", "want to finish", "want to live", "not ready to die"
                    ]
                }

                with open(protective_factors_path, 'w') as f:
                    json.dump(self.protective_factors, f, indent=2)

        except Exception as e:
            logger.error(f"Error loading risk lexicons: {e}")
            # Create fallback simple dictionaries if loading fails
            self.urgent_risk_phrases = {"immediate_intent": ["kill myself now", "about to end it"]}
            self.high_risk_phrases = {"suicidal_intent": ["want to die", "kill myself"]}
            self.medium_risk_phrases = {"passive_ideation": ["wish I could disappear"]}
            self.crisis_response_indicators = ["concerned about your safety"]
            self.protective_factors = {"social_support": ["my family"]}

    def extract_features(self, message, response, analysis):
        """
        Extract numerical features from message content for risk assessment

        Returns:
            dict: Feature dictionary
        """
        features = {}

        # Preprocess text
        message_lower = message.lower()
        response_lower = response.lower()
        analysis_lower = analysis.lower() if analysis else ""

        # Tokenize
        message_tokens = word_tokenize(message_lower)
        message_words = [w for w in message_tokens if w.isalnum() and w not in self.stop_words]

        # 1. Pattern counts from lexicons
        features['urgent_risk_count'] = 0
        features['high_risk_count'] = 0
        features['medium_risk_count'] = 0
        features['protective_factors_count'] = 0

        # Count matches in each risk category
        for category, phrases in self.urgent_risk_phrases.items():
            for phrase in phrases:
                if phrase in message_lower:
                    features['urgent_risk_count'] += 1

        for category, phrases in self.high_risk_phrases.items():
            for phrase in phrases:
                if phrase in message_lower:
                    features['high_risk_count'] += 1

        for category, phrases in self.medium_risk_phrases.items():
            for phrase in phrases:
                if phrase in message_lower:
                    features['medium_risk_count'] += 1

        for category, phrases in self.protective_factors.items():
            for phrase in phrases:
                if phrase in message_lower:
                    features['protective_factors_count'] += 1

        # 2. Response pattern analysis
        features['crisis_response_count'] = sum(1 for phrase in self.crisis_response_indicators
                                                if phrase in response_lower)

        # 3. Message structure features
        features['message_length'] = len(message_words)
        features['word_count'] = len(message_words)
        features['question_marks'] = message.count('?')
        features['exclamation_marks'] = message.count('!')

        # 4. Lexical features
        pronoun_counts = Counter([word for word in message_words
                                  if word.lower() in ['i', 'me', 'my', 'mine', 'myself']])
        features['first_person_pronouns'] = sum(pronoun_counts.values())

        # 5. Sentiment words (simple approach)
        negative_emotions = ['sad', 'angry', 'depressed', 'hopeless', 'worthless', 'tired',
                             'exhausted', 'pain', 'hurt', 'suffering', 'miserable', 'afraid']
        features['negative_emotion_words'] = sum(1 for word in message_words
                                                 if word in negative_emotions)

        # 6. Analysis-based features
        if "risk assessment: urgent" in analysis_lower or "immediate safety risk assessment: urgent" in analysis_lower:
            features['analysis_urgent_risk'] = 1
        else:
            features['analysis_urgent_risk'] = 0

        if "risk assessment: high" in analysis_lower or "immediate safety risk assessment: high" in analysis_lower:
            features['analysis_high_risk'] = 1
        else:
            features['analysis_high_risk'] = 0

        if "risk assessment: medium" in analysis_lower or "immediate safety risk assessment: medium" in analysis_lower:
            features['analysis_medium_risk'] = 1
        else:
            features['analysis_medium_risk'] = 0

        # 7. Temporal indicators (looking for imminent language)
        temporal_words = ['now', 'tonight', 'today', 'immediately', 'soon', 'going to', 'about to']
        features['temporal_urgency'] = sum(1 for word in temporal_words if word in message_lower)

        return features

    def calculate_risk_score(self, features):
        """
        Calculate a risk score based on extracted features

        Returns:
            float: Risk score between 0 and 1
        """
        # If we have a trained ML model, use it
        if self.model:
            try:
                # Convert features to the format expected by the model
                feature_vector = [features[k] for k in sorted(features.keys())]
                # Get prediction probability
                risk_score = self.model.predict_proba([feature_vector])[0][1]  # Assuming binary classification
                return float(risk_score)
            except Exception as e:
                logger.error(f"Error using ML model: {e}")
                # Fall back to rule-based method if model fails

        # Rule-based scoring as fallback or if no model is available
        score = 0.0

        # Urgent risk factors (strongest weight)
        score += min(features['urgent_risk_count'] * 0.25, 0.6)  # Cap at 0.6

        # High risk factors
        score += min(features['high_risk_count'] * 0.15, 0.4)  # Cap at 0.4

        # Medium risk factors
        score += min(features['medium_risk_count'] * 0.08, 0.3)  # Cap at 0.3

        # Response patterns from bot
        score += min(features['crisis_response_count'] * 0.1, 0.3)  # Cap at 0.3

        # Analysis-based boosting
        if features['analysis_urgent_risk'] == 1:
            score += 0.4
        elif features['analysis_high_risk'] == 1:
            score += 0.25
        elif features['analysis_medium_risk'] == 1:
            score += 0.15

        # Temporal urgency
        score += min(features['temporal_urgency'] * 0.1, 0.3)

        # Message structure
        if features['message_length'] > 200 and features['negative_emotion_words'] > 3:
            score += 0.05  # Long, emotional messages might indicate crisis

        # Protective factors (reduce score)
        score -= min(features['protective_factors_count'] * 0.05, 0.2)  # Cap reduction at 0.2

        # Ensure score is between 0 and 1
        return max(0.0, min(score, 1.0))

    def detect_crisis(self, message, response, analysis, sentiment_tags):
        """
        Comprehensive crisis detection based on extracted features and risk scoring

        Returns:
            str: Crisis level - "urgent", "high", "medium", "low", or "none"
        """
        # Extract feature set
        features = self.extract_features(message, response, analysis)

        # Calculate risk score
        risk_score = self.calculate_risk_score(features)

        # Log the risk assessment details
        logger.info(f"Risk score: {risk_score:.2f}, Features: {features}")

        # Determine crisis level based on score
        if risk_score >= self.risk_thresholds['urgent']:
            crisis_level = "urgent"
            logger.critical(f"URGENT CRISIS DETECTED (score: {risk_score:.2f}): {message[:100]}")
            self.create_crisis_alert(crisis_level, message, analysis, features, risk_score)

        elif risk_score >= self.risk_thresholds['high']:
            crisis_level = "high"
            logger.error(f"HIGH RISK CRISIS DETECTED (score: {risk_score:.2f}): {message[:100]}")
            self.create_crisis_alert(crisis_level, message, analysis, features, risk_score)

        elif risk_score >= self.risk_thresholds['medium']:
            crisis_level = "medium"
            logger.warning(f"MEDIUM RISK CRISIS DETECTED (score: {risk_score:.2f}): {message[:100]}")
            self.create_crisis_alert(crisis_level, message, analysis, features, risk_score)

        elif risk_score >= self.risk_thresholds['low']:
            crisis_level = "low"
            logger.info(f"LOW RISK DETECTED (score: {risk_score:.2f}): {message[:100]}")

        else:
            crisis_level = "none"

        # Additional check for specific sentiment tags
        if sentiment_tags:
            if isinstance(sentiment_tags, str):
                sentiment_list = [sentiment_tags.lower()]
            else:
                sentiment_list = [tag.lower() for tag in sentiment_tags]

            high_risk_sentiments = ["suicidal", "suicide", "self-harm", "severe depression"]

            # Upgrade risk level if high-risk sentiment is detected
            if any(s in sentiment for sentiment in sentiment_list for s in
                   high_risk_sentiments) and crisis_level == "none":
                crisis_level = "medium"
                logger.warning(f"MEDIUM RISK DETECTED from sentiment tags: {sentiment_tags}")
                self.create_crisis_alert(crisis_level, message,
                                         f"Crisis detected from sentiment tags: {sentiment_tags}",
                                         features, 0.45)

        return crisis_level

    def create_crisis_alert(self, level, message, analysis, features=None, risk_score=None):
        """Create a crisis alert file for urgent/high/medium risk situations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alert_id = str(uuid.uuid4())[:8]
        alert_file = f"crisis_alerts/{timestamp}_{alert_id}_{level}.json"

        # Get user_id from Flask session if available
        try:
            from flask import session
            user_id = session.get('user_id', 'unknown')
        except (ImportError, RuntimeError):
            user_id = 'unknown'

        alert_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "message": message,
            "analysis": analysis,
            "user_id": user_id,
            "status": "new"
        }

        # Add risk assessment details if available
        if features and risk_score:
            alert_data["risk_assessment"] = {
                "risk_score": risk_score,
                "features": features
            }

        try:
            with open(alert_file, 'w') as f:
                json.dump(alert_data, f, indent=2)
            logger.info(f"Crisis alert created: {alert_file}")
        except Exception as e:
            logger.error(f"Failed to create crisis alert: {e}")

    def train_model(self, training_data_path):
        """
        Train a machine learning model for crisis detection

        Args:
            training_data_path: Path to labeled training data

        Returns:
            bool: Success status
        """
        try:
            import pandas as pd
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            import pickle

            # Load training data
            data = pd.read_csv(training_data_path)

            # Prepare features and labels
            X = data.drop('crisis_level', axis=1)
            y = data['crisis_level']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate
            accuracy = model.score(X_test, y_test)
            logger.info(f"Model trained with accuracy: {accuracy:.2f}")

            # Save model
            model_path = os.path.join(os.path.dirname(__file__), 'crisis_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # Update current model
            self.model = model

            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False