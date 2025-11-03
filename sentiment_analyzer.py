"""
Sentiment and Mood Analyzer
Analyzes message content to detect sentiment and overall mood.
"""

import re
from typing import List, Tuple, Dict
from whatsapp_parser import Message
from collections import Counter


class SentimentAnalyzer:
    """Analyzes sentiment and mood from messages."""
    
    # Positive indicators
    POSITIVE_WORDS = {
        'great', 'awesome', 'amazing', 'excellent', 'fantastic', 'wonderful',
        'love', 'happy', 'excited', 'good', 'nice', 'cool', 'perfect',
        'thanks', 'thank you', 'appreciate', 'wonderful', 'brilliant',
        'fun', 'enjoy', 'yay', 'yes!', 'absolutely', 'definitely',
        'yes', 'sure', 'glad', 'pleased', 'delighted', 'thrilled',
        'best', 'favorite', 'beautiful', 'lovely', 'sweet', 'cute'
    }
    
    # Negative indicators
    NEGATIVE_WORDS = {
        'bad', 'terrible', 'awful', 'hate', 'worst', 'sad', 'angry',
        'mad', 'frustrated', 'disappointed', 'sorry', 'apologize',
        'no', "can't", "won't", 'worse', 'horrible', 'annoying',
        'stress', 'stressed', 'worried', 'concerned', 'tired',
        'sick', 'hurt', 'upset', 'unhappy', 'miserable'
    }
    
    # Excitement indicators
    EXCITEMENT_INDICATORS = {
        '!', 'wow', 'omg', 'yay', 'woohoo', 'hooray', 'finally',
        'can\'t wait', "can't wait", 'excited', 'so excited',
        'awesome', 'amazing', 'fantastic', 'incredible'
    }
    
    # Question patterns (showing engagement)
    QUESTION_PATTERNS = re.compile(r'\?+')
    
    # Emoji patterns (basic detection)
    EMOJI_PATTERNS = re.compile(
        r'[\U0001F600-\U0001F64F]|'  # Emoticons
        r'[\U0001F300-\U0001F5FF]|'  # Misc Symbols
        r'[\U0001F680-\U0001F6FF]|'  # Transport
        r'[\U0001F1E0-\U0001F1FF]|'  # Flags
        r'[\U00002702-\U000027B0]|'  # Dingbats
        r'[\U000024C2-\U0001F251]',
        flags=re.UNICODE
    )
    
    def analyze_message(self, message: Message) -> Dict[str, float]:
        """Analyze a single message for sentiment and mood indicators."""
        if message.is_system:
            return {'sentiment': 0.0, 'excitement': 0.0, 'engagement': 0.0}
        
        content = message.content.lower()
        
        # Sentiment analysis
        positive_count = sum(1 for word in self.POSITIVE_WORDS if word in content)
        negative_count = sum(1 for word in self.NEGATIVE_WORDS if word in content)
        
        # Normalize by message length (words)
        words = len(content.split())
        sentiment = 0.0
        if words > 0:
            sentiment = (positive_count - negative_count) / max(words, 1)
            sentiment = max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]
        
        # Excitement level (based on punctuation, capitalization, excitement words)
        exclamation_count = content.count('!')
        caps_ratio = sum(1 for c in message.content if c.isupper()) / max(len(message.content), 1)
        excitement_words = sum(1 for word in self.EXCITEMENT_INDICATORS if word in content)
        
        excitement = min(1.0, (exclamation_count * 0.1 + caps_ratio * 0.5 + excitement_words * 0.2))
        
        # Engagement (questions, emojis, message length)
        has_question = bool(self.QUESTION_PATTERNS.search(message.content))
        emoji_count = len(self.EMOJI_PATTERNS.findall(message.content))
        length_engagement = min(1.0, words / 50.0)  # Longer messages = more engaged
        
        engagement = (has_question * 0.3 + min(emoji_count * 0.1, 0.3) + length_engagement * 0.4)
        
        return {
            'sentiment': sentiment,
            'excitement': excitement,
            'engagement': engagement
        }
    
    def analyze_phase(self, messages: List[Message]) -> Dict[str, any]:
        """Analyze a collection of messages to determine overall phase mood."""
        if not messages:
            return {
                'mood': 'Neutral',
                'vibe': 'Calm',
                'sentiment': 0.0,
                'excitement': 0.0,
                'engagement': 0.0,
                'emoji': '😐'
            }
        
        analyses = [self.analyze_message(msg) for msg in messages if not msg.is_system]
        
        if not analyses:
            return {
                'mood': 'Neutral',
                'vibe': 'Calm',
                'sentiment': 0.0,
                'excitement': 0.0,
                'engagement': 0.0,
                'emoji': '😐'
            }
        
        avg_sentiment = sum(a['sentiment'] for a in analyses) / len(analyses)
        avg_excitement = sum(a['excitement'] for a in analyses) / len(analyses)
        avg_engagement = sum(a['engagement'] for a in analyses) / len(analyses)
        
        # Determine mood
        if avg_sentiment > 0.3:
            if avg_excitement > 0.4:
                mood = 'Euphoric'
                vibe = 'Electric'
                emoji = '🤩'
            elif avg_excitement > 0.2:
                mood = 'Happy'
                vibe = 'Upbeat'
                emoji = '😊'
            else:
                mood = 'Content'
                vibe = 'Positive'
                emoji = '🙂'
        elif avg_sentiment < -0.3:
            if avg_excitement > 0.3:
                mood = 'Frustrated'
                vibe = 'Tense'
                emoji = '😤'
            else:
                mood = 'Down'
                vibe = 'Melancholic'
                emoji = '😔'
        else:
            if avg_excitement > 0.3:
                mood = 'Energetic'
                vibe = 'Dynamic'
                emoji = '⚡'
            elif avg_engagement > 0.5:
                mood = 'Engaged'
                vibe = 'Active'
                emoji = '💬'
            else:
                mood = 'Neutral'
                vibe = 'Calm'
                emoji = '😐'
        
        return {
            'mood': mood,
            'vibe': vibe,
            'sentiment': avg_sentiment,
            'excitement': avg_excitement,
            'engagement': avg_engagement,
            'emoji': emoji
        }
    
    def get_mood_color(self, mood_data: Dict[str, any]) -> str:
        """Get a color associated with the mood."""
        mood = mood_data['mood']
        sentiment = mood_data['sentiment']
        
        if sentiment > 0.3:
            if mood_data['excitement'] > 0.4:
                return '#FFD700'  # Gold - Euphoric
            return '#90EE90'  # Light green - Happy/Positive
        elif sentiment < -0.3:
            return '#FFB6C1'  # Light pink/salmon - Down/Frustrated
        elif mood_data['excitement'] > 0.3:
            return '#FFA500'  # Orange - Energetic
        elif mood_data['engagement'] > 0.5:
            return '#87CEEB'  # Sky blue - Engaged
        else:
            return '#D3D3D3'  # Light gray - Neutral

