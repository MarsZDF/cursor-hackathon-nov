"""
Phase Detection Module
Identifies distinct phases in a conversation based on various criteria.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from whatsapp_parser import Message
import numpy as np
from collections import Counter
from sentiment_analyzer import SentimentAnalyzer
import re


@dataclass
class Phase:
    """Represents a phase in the conversation."""
    start_time: datetime
    end_time: datetime
    message_indices: List[int]  # Indices of messages in this phase
    phase_type: str  # Description of the phase
    dominant_sender: str  # Most active sender in this phase
    message_count: int
    duration_hours: float
    mood: str = "Neutral"  # Overall mood of the phase
    vibe: str = "Calm"  # Vibe description
    mood_emoji: str = "😐"  # Emoji representing the mood
    sentiment: float = 0.0  # Average sentiment score
    top_keywords: List[str] = field(default_factory=list)  # Top keywords/topics in this phase
    avg_message_length: float = 0.0  # Average message length in characters
    summary_sentence: str = ""  # One sentence summary (10-20 words) of the phase content


class PhaseDetector:
    """Detects phases in a conversation."""
    
    def __init__(self, messages: List[Message]):
        self.messages = messages
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def detect_phases(
        self,
        min_gap_hours: float = 24.0,
        min_messages_per_phase: int = 5,
        method: str = "time_gap"
    ) -> List[Phase]:
        """
        Detect phases in the conversation.
        
        Args:
            min_gap_hours: Minimum time gap (in hours) to consider as a phase break
            min_messages_per_phase: Minimum number of messages to form a phase
            method: Detection method ('time_gap', 'activity', 'hybrid')
            
        Returns:
            List of Phase objects
        """
        if method == "time_gap":
            return self._detect_by_time_gap(min_gap_hours, min_messages_per_phase)
        elif method == "activity":
            return self._detect_by_activity(min_messages_per_phase)
        elif method == "hybrid":
            return self._detect_hybrid(min_gap_hours, min_messages_per_phase)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _detect_by_time_gap(
        self,
        min_gap_hours: float,
        min_messages_per_phase: int
    ) -> List[Phase]:
        """Detect phases based on time gaps between messages."""
        if not self.messages:
            return []
        
        phases = []
        current_phase_start = 0
        
        for i in range(1, len(self.messages)):
            time_gap = (self.messages[i].timestamp - self.messages[i-1].timestamp).total_seconds() / 3600
            
            # If gap is large enough, end current phase and start new one
            if time_gap >= min_gap_hours:
                # Check if current phase has enough messages
                if i - current_phase_start >= min_messages_per_phase:
                    phase = self._create_phase(current_phase_start, i - 1)
                    phases.append(phase)
                current_phase_start = i
        
        # Add final phase
        if len(self.messages) - current_phase_start >= min_messages_per_phase:
            phase = self._create_phase(current_phase_start, len(self.messages) - 1)
            phases.append(phase)
        elif len(phases) == 0:
            # If we don't have any phases yet, create one for all messages
            phase = self._create_phase(0, len(self.messages) - 1)
            phases.append(phase)
        
        return phases
    
    def _detect_by_activity(self, min_messages_per_phase: int) -> List[Phase]:
        """Detect phases based on message activity patterns."""
        if not self.messages:
            return []
        
        # Calculate message frequency over time windows
        window_hours = 24
        if len(self.messages) < min_messages_per_phase * 2:
            # If very few messages, just create one phase
            return [self._create_phase(0, len(self.messages) - 1)]
        
        # Find natural breaks in activity
        message_rates = []
        for i in range(len(self.messages)):
            window_start = self.messages[i].timestamp - timedelta(hours=window_hours)
            window_end = self.messages[i].timestamp
            count = sum(1 for msg in self.messages 
                       if window_start <= msg.timestamp <= window_end)
            message_rates.append(count)
        
        # Find significant drops in activity (potential phase breaks)
        phases = []
        current_phase_start = 0
        
        if len(message_rates) < 2:
            return [self._create_phase(0, len(self.messages) - 1)]
        
        # Normalize rates and find significant drops
        rates_array = np.array(message_rates)
        if rates_array.std() > 0:
            normalized_rates = (rates_array - rates_array.mean()) / rates_array.std()
            threshold = -0.5  # Significant drop threshold
            
            for i in range(1, len(normalized_rates)):
                if normalized_rates[i] < threshold and normalized_rates[i-1] >= threshold:
                    if i - current_phase_start >= min_messages_per_phase:
                        phase = self._create_phase(current_phase_start, i - 1)
                        phases.append(phase)
                        current_phase_start = i
        
        # Add final phase
        if len(self.messages) - current_phase_start >= min_messages_per_phase:
            phase = self._create_phase(current_phase_start, len(self.messages) - 1)
            phases.append(phase)
        elif len(phases) == 0:
            phase = self._create_phase(0, len(self.messages) - 1)
            phases.append(phase)
        
        return phases
    
    def _detect_hybrid(
        self,
        min_gap_hours: float,
        min_messages_per_phase: int
    ) -> List[Phase]:
        """Detect phases using both time gaps and activity patterns."""
        time_gap_phases = self._detect_by_time_gap(min_gap_hours, min_messages_per_phase)
        activity_phases = self._detect_by_activity(min_messages_per_phase)
        
        # Combine and merge overlapping phases
        all_breakpoints = set()
        for phase in time_gap_phases:
            all_breakpoints.add(phase.start_time)
            all_breakpoints.add(phase.end_time)
        for phase in activity_phases:
            all_breakpoints.add(phase.start_time)
            all_breakpoints.add(phase.end_time)
        
        breakpoints = sorted(all_breakpoints)
        if not breakpoints:
            return [self._create_phase(0, len(self.messages) - 1)]
        
        phases = []
        current_start_idx = 0
        
        for breakpoint in breakpoints[1:]:
            # Find index of last message before breakpoint
            end_idx = next(
                (i for i, msg in enumerate(self.messages) if msg.timestamp >= breakpoint),
                len(self.messages)
            ) - 1
            
            if end_idx >= current_start_idx and end_idx - current_start_idx >= min_messages_per_phase - 1:
                phase = self._create_phase(current_start_idx, end_idx)
                phases.append(phase)
                current_start_idx = end_idx + 1
        
        # Add final phase
        if current_start_idx < len(self.messages):
            if len(self.messages) - current_start_idx >= min_messages_per_phase:
                phase = self._create_phase(current_start_idx, len(self.messages) - 1)
                phases.append(phase)
            elif len(phases) == 0:
                phase = self._create_phase(0, len(self.messages) - 1)
                phases.append(phase)
        
        return phases if phases else [self._create_phase(0, len(self.messages) - 1)]
    
    def _create_phase(self, start_idx: int, end_idx: int) -> Phase:
        """Create a Phase object from message indices."""
        if start_idx > end_idx or start_idx >= len(self.messages) or end_idx >= len(self.messages):
            raise ValueError(f"Invalid indices: {start_idx}, {end_idx}")
        
        phase_messages = [self.messages[i] for i in range(start_idx, end_idx + 1)]
        start_time = phase_messages[0].timestamp
        end_time = phase_messages[-1].timestamp
        duration = (end_time - start_time).total_seconds() / 3600
        
        # Find dominant sender
        sender_counts = Counter(msg.sender for msg in phase_messages if not msg.is_system)
        dominant_sender = sender_counts.most_common(1)[0][0] if sender_counts else "Unknown"
        
        # Analyze mood and vibe first (used by classification)
        mood_data = self.sentiment_analyzer.analyze_phase(phase_messages)
        
        # Determine phase type with mood (pass mood_data to avoid re-analyzing)
        phase_type = self._classify_phase(phase_messages, duration, mood_data)
        
        # Extract topics/keywords
        top_keywords = self._extract_keywords(phase_messages)
        
        # Calculate average message length
        avg_length = sum(len(msg.content) for msg in phase_messages) / max(len(phase_messages), 1)
        
        # Calculate message rate for summary generation
        msg_rate = len(phase_messages) / max(duration, 0.1)
        
        # Generate sentence summary
        summary_sentence = self._generate_summary(phase_messages, top_keywords, mood_data, duration, msg_rate)
        
        return Phase(
            start_time=start_time,
            end_time=end_time,
            message_indices=list(range(start_idx, end_idx + 1)),
            phase_type=phase_type,
            dominant_sender=dominant_sender,
            message_count=len(phase_messages),
            duration_hours=duration,
            mood=mood_data['mood'],
            vibe=mood_data['vibe'],
            mood_emoji=mood_data['emoji'],
            sentiment=mood_data['sentiment'],
            top_keywords=top_keywords,
            avg_message_length=avg_length,
            summary_sentence=summary_sentence
        )
    
    def _classify_phase(self, messages: List[Message], duration_hours: float, mood_data: dict = None) -> str:
        """Classify the type of phase based on its characteristics with mood-aware descriptions."""
        if not messages:
            return "Silence"
        
        msg_count = len(messages)
        msg_rate = msg_count / max(duration_hours, 0.1)  # Messages per hour
        
        # Use provided mood_data or analyze if not provided
        if mood_data is None:
            mood_data = self.sentiment_analyzer.analyze_phase(messages)
        mood_prefix = mood_data['mood']
        
        # Very high activity
        if msg_rate > 20:
            return f"🔥 Intense {mood_prefix} Vibes"
        # High activity
        elif msg_rate > 10:
            return f"💬 Active {mood_prefix} Chat"
        # Medium activity
        elif msg_rate > 3:
            return f"✨ {mood_prefix} Conversation"
        # Low activity but messages present
        elif msg_rate > 0.5:
            return f"💭 Casual {mood_prefix} Check-in"
        # Very low activity
        else:
            return f"🌙 Quiet {mood_prefix} Period"
    
    def _extract_keywords(self, messages: List[Message], top_n: int = 5) -> List[str]:
        """Extract top keywords from phase messages."""
        if not messages:
            return []
        
        # Get ALL participant names from the entire conversation (not just this phase)
        # This ensures we filter out names even if they weren't in this specific phase
        participant_names = set()
        for msg in self.messages:
            if not msg.is_system:
                # Split names (handles "Sam Gill" -> ["sam", "gill"])
                name_parts = msg.sender.lower().split()
                participant_names.update(name_parts)
                # Also add full name as single string (for cases like "@samgill" or "hey samgill")
                full_name = msg.sender.lower().replace(' ', '')
                if full_name:
                    participant_names.add(full_name)
        
        # Common stopwords and WhatsApp artifacts to filter out
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'then',
            'i\'m', 'you\'re', 'he\'s', 'she\'s', 'it\'s', 'we\'re', 'they\'re', 'i\'ve', 'you\'ve',
            'we\'ve', 'they\'ve', 'i\'ll', 'you\'ll', 'he\'ll', 'she\'ll', 'we\'ll', 'they\'ll',
            'don\'t', 'doesn\'t', 'didn\'t', 'won\'t', 'wouldn\'t', 'couldn\'t', 'shouldn\'t',
            'can\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'haven\'t', 'hasn\'t', 'hadn\'t',
            'image', 'omitted', 'video', 'sticker', 'omitted', 'audio', 'document', 'link', 'preview',
            'changed', 'phone', 'number', 'end-to-end', 'encrypted', 'messages', 'calls',
            'think', 'like', 'know', 'get', 'got', 'go', 'going', 'come', 'see', 'said', 'say',
            'want', 'need', 'make', 'made', 'take', 'took', 'give', 'gave', 'tell', 'told',
            'yes', 'yeah', 'yep', 'no', 'nope', 'ok', 'okay', 'sure', 'right', 'yeah',
            'lol', 'haha', 'hahaha', 'omg', 'wow', 'ugh', 'ah', 'oh', 'hey', 'hi', 'hello',
            'really', 'actually', 'probably', 'maybe', 'might', 'quite', 'pretty', 'really',
            'well', 'good', 'great', 'nice', 'cool', 'awesome', 'bad', 'sorry', 'thanks', 'thank'
        }
        
        # Collect all words from messages
        word_counter = Counter()
        for msg in messages:
            if msg.is_system:
                continue
            # Convert to lowercase and split into words
            text = msg.content.lower()
            # Remove URLs, emojis (basic), and special characters, keep only words
            text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
            text = re.sub(r'[^\w\s\']', ' ', text)  # Keep only alphanumeric and apostrophes
            words = text.split()
            
            # Filter out stopwords, participant names, and very short words
            for word in words:
                word = word.strip('\'".,!?;:()[]{}')
                # Check if word matches any participant name (case-insensitive)
                word_lower = word.lower()
                # Check exact match and also check if it's contained in any participant name or vice versa
                is_participant_name = (
                    word_lower in participant_names or
                    any(word_lower in name or name in word_lower for name in participant_names if len(name) > 2)
                )
                if (len(word) > 2 and word not in stopwords and 
                    not is_participant_name and not word.isdigit()):
                    word_counter[word] += 1
        
        # Get top N keywords
        top_keywords = [word for word, count in word_counter.most_common(top_n)]
        
        return top_keywords
    
    def _generate_summary(self, messages: List[Message], keywords: List[str], 
                         mood_data: dict, duration_hours: float, msg_rate: float) -> str:
        """Generate a one-sentence summary (10-20 words) of the phase content."""
        if not messages:
            return "No messages in this phase."
        
        # Get key phrases from messages - look for meaningful sentences
        # Get ALL participant names from entire conversation for better filtering
        participant_names = set()
        for msg in self.messages:
            if not msg.is_system:
                name_parts = msg.sender.lower().split()
                participant_names.update(name_parts)
                # Also add full name without spaces
                full_name = msg.sender.lower().replace(' ', '')
                if full_name:
                    participant_names.add(full_name)
        
        # Extract sentences with keywords
        sentences_with_keywords = []
        for msg in messages:
            if msg.is_system or len(msg.content.strip()) < 10:
                continue
            content = msg.content
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+', content)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 15 or len(sentence) > 150:
                    continue
                # Check if sentence contains any keywords
                sentence_lower = sentence.lower()
                if any(kw in sentence_lower for kw in keywords[:3] if keywords):
                    sentences_with_keywords.append(sentence)
        
        # Try to create a summary
        # Add mood/activity context
        if msg_rate > 15:
            activity = "intense"
        elif msg_rate > 8:
            activity = "active"
        elif msg_rate > 3:
            activity = "regular"
        else:
            activity = "casual"
        
        sentiment_val = mood_data.get('sentiment', 0.0)
        if sentiment_val > 0.2:
            tone = "positive"
        elif sentiment_val < -0.2:
            tone = "concerned"
        else:
            tone = "neutral"
        
        # Build summary using keywords
        if keywords:
            # Use top 2-3 keywords to form a sentence
            main_keywords = keywords[:3]
            
            # Simple template-based generation (ensure 10-20 words)
            if len(main_keywords) >= 2:
                templates = [
                    f"This was an {activity} {tone} conversation about {main_keywords[0]} and {main_keywords[1]}.",
                    f"Discussion centered on {main_keywords[0]} with {tone} {activity} messaging.",
                    f"Main topics included {main_keywords[0]} and {main_keywords[1]} in {tone} {activity} exchange.",
                    f"The conversation focused on {main_keywords[0]} during this {activity} {tone} period.",
                    f"This phase featured {main_keywords[0]} and {main_keywords[1]} in {activity} {tone} discussion."
                ]
            else:
                templates = [
                    f"This was an {activity} {tone} conversation about {main_keywords[0] if main_keywords else 'various topics'}.",
                    f"Discussion centered on {main_keywords[0] if main_keywords else 'topics'} with {tone} {activity} messaging.",
                    f"Main topic was {main_keywords[0] if main_keywords else 'conversation'} in {tone} {activity} exchange.",
                    f"The conversation focused on {main_keywords[0] if main_keywords else 'topics'} during {activity} {tone} period."
                ]
            
            # Try to find a sentence from actual messages that contains keywords
            if sentences_with_keywords:
                # Pick a representative sentence (prefer medium length)
                candidate = None
                for s in sentences_with_keywords[:10]:
                    word_count = len(s.split())
                    if 8 <= word_count <= 25:
                        candidate = s
                        break
                
                if candidate:
                    # Clean up and use it, but ensure it's within word limit
                    candidate = candidate.strip()
                    candidate_words = candidate.split()
                    if 10 <= len(candidate_words) <= 20:
                        return candidate
                    elif len(candidate_words) > 20:
                        # Truncate to 20 words
                        truncated = ' '.join(candidate_words[:20])
                        if not truncated[-1] in '.!?':
                            truncated += '.'
                        return truncated
                    else:
                        # Too short, use template
                        summary = templates[0] if len(templates) > 0 else f"Discussion about {', '.join(main_keywords[:2])}."
                else:
                    summary = templates[0] if len(templates) > 0 else f"Discussion about {', '.join(main_keywords[:2])}."
            else:
                summary = templates[0] if len(templates) > 0 else f"Discussion about {', '.join(main_keywords[:2])}."
        else:
            summary = f"This was an {activity} {tone} conversation period."
        
        # Ensure word count is between 10-20 words
        words = summary.split()
        if len(words) < 10:
            # Add more context
            if keywords and len(keywords) >= 2:
                summary += f" Key topics included {keywords[0]} and {keywords[1]}."
            elif keywords:
                summary += f" Main topic was {keywords[0]}."
            else:
                summary += " The conversation covered various topics and updates."
            words = summary.split()
            # If still too short, add activity context
            if len(words) < 10:
                summary = f"During this {activity} {tone} phase, " + summary.lower()
            words = summary.split()
        
        if len(words) > 20:
            # Truncate to 20 words, try to end at sentence boundary
            truncated_words = words[:20]
            summary = ' '.join(truncated_words)
            # Remove trailing incomplete words if they don't end properly
            if summary[-1] not in '.!?,;:':
                summary += '.'
        
        return summary

