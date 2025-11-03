"""
WhatsApp Message Parser
Parses WhatsApp exported chat files and extracts structured message data.
"""

import re
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Message:
    """Represents a single WhatsApp message."""
    timestamp: datetime
    sender: str
    content: str
    is_system: bool = False  # For system messages like "You left", etc.


class WhatsAppParser:
    """Parser for WhatsApp exported chat files."""
    
    # Pattern for standard WhatsApp format: [DD/MM/YYYY, HH:MM:SS AM/PM] Sender: Message
    # Also handles 24-hour format: [DD/MM/YYYY, HH:MM:SS] Sender: Message
    MESSAGE_PATTERN = re.compile(
        r'\[(\d{1,2}/\d{1,2}/\d{4}),\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*(AM|PM)?\]\s*(.+?):\s*(.+)$',
        re.MULTILINE
    )
    
    # Pattern for continuation lines (messages that span multiple lines)
    CONTINUATION_PATTERN = re.compile(r'^\d{1,2}/\d{1,2}/\d{4}')
    
    def __init__(self):
        self.messages: List[Message] = []
    
    def parse(self, file_path: str) -> List[Message]:
        """
        Parse a WhatsApp exported chat file.
        
        Args:
            file_path: Path to the exported WhatsApp chat file
            
        Returns:
            List of Message objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        messages = []
        lines = content.split('\n')
        current_message = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match a new message
            match = self.MESSAGE_PATTERN.match(line)
            if match:
                # Save previous message if exists
                if current_message:
                    messages.append(current_message)
                
                # Parse new message
                date_str, time_str, am_pm, sender, content = match.groups()
                timestamp = self._parse_timestamp(date_str, time_str, am_pm)
                
                current_message = Message(
                    timestamp=timestamp,
                    sender=sender.strip(),
                    content=content.strip(),
                    is_system=sender.strip().lower() in ['system', 'you left', 'you joined']
                )
            elif current_message:
                # Continuation of previous message
                current_message.content += '\n' + line
        
        # Add last message
        if current_message:
            messages.append(current_message)
        
        self.messages = messages
        return messages
    
    def _parse_timestamp(self, date_str: str, time_str: str, am_pm: Optional[str]) -> datetime:
        """Parse timestamp string into datetime object."""
        # Handle 24-hour format (no AM/PM)
        if am_pm is None:
            # Check if seconds are included
            if len(time_str.split(':')) == 3:
                format_str = '%d/%m/%Y, %H:%M:%S'
            else:
                format_str = '%d/%m/%Y, %H:%M'
            return datetime.strptime(f'{date_str}, {time_str}', format_str)
        
        # Handle 12-hour format (with AM/PM)
        time_with_am_pm = f'{time_str} {am_pm}'
        if len(time_str.split(':')) == 3:
            format_str = '%d/%m/%Y, %I:%M:%S %p'
        else:
            format_str = '%d/%m/%Y, %I:%M %p'
        
        return datetime.strptime(f'{date_str}, {time_with_am_pm}', format_str)
    
    def get_messages(self) -> List[Message]:
        """Get all parsed messages."""
        return self.messages
    
    def get_messages_by_sender(self, sender: str) -> List[Message]:
        """Get all messages from a specific sender."""
        return [msg for msg in self.messages if msg.sender == sender]
    
    def get_message_count(self) -> int:
        """Get total number of messages."""
        return len(self.messages)
    
    def get_participants(self) -> List[str]:
        """Get list of unique participants."""
        return sorted(set(msg.sender for msg in self.messages if not msg.is_system))
    
    def get_time_span(self) -> tuple:
        """Get the time span of the conversation."""
        if not self.messages:
            return None, None
        timestamps = [msg.timestamp for msg in self.messages]
        return min(timestamps), max(timestamps)

