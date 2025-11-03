"""
Phase Detection Module
Identifies distinct phases in a conversation based on various criteria.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass
from whatsapp_parser import Message
import numpy as np
from collections import Counter
from sentiment_analyzer import SentimentAnalyzer


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
            sentiment=mood_data['sentiment']
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

