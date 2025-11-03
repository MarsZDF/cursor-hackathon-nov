# WhatsApp Conversation Phase Analyzer

A Python tool to parse WhatsApp exported conversations, identify distinct phases in the conversation, and visualize them.

## Features

- **WhatsApp Message Parsing**: Parses standard WhatsApp export format (both 12-hour and 24-hour time formats)
- **Phase Detection**: Identifies conversation phases using multiple methods:
  - **Time Gap**: Detects phases based on time gaps between messages
  - **Activity**: Detects phases based on message activity patterns
  - **Hybrid**: Combines both time gaps and activity patterns
- **Visualizations**: Creates comprehensive visualizations including:
  - Phase timeline
  - Message activity over time
  - Phase statistics (message counts, durations, rates)
  - Summary dashboard

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py chat.txt
```

### Advanced Options

```bash
# Use activity-based phase detection
python main.py chat.txt --method activity

# Adjust minimum time gap (in hours) for phase splitting
python main.py chat.txt --method time_gap --min-gap 12

# Set minimum messages per phase
python main.py chat.txt --min-messages 10

# Save visualizations to a specific directory
python main.py chat.txt --output results/

# Generate specific visualization types
python main.py chat.txt --visualize timeline
python main.py chat.txt --visualize stats
python main.py chat.txt --visualize summary  # default
python main.py chat.txt --visualize all

# Save without displaying (useful for scripts)
python main.py chat.txt --no-display --output results/
```

### Command Line Arguments

- `input_file`: Path to WhatsApp exported chat file (required)
- `--method`: Phase detection method - `time_gap`, `activity`, or `hybrid` (default: `time_gap`)
- `--min-gap`: Minimum time gap in hours to split phases (default: 24.0)
- `--min-messages`: Minimum number of messages per phase (default: 5)
- `--output`: Output directory for visualizations (default: current directory)
- `--visualize`: Type of visualization - `timeline`, `stats`, `summary`, or `all` (default: `summary`)
- `--no-display`: Save visualizations without displaying them

## WhatsApp Export Format

The parser expects WhatsApp exported chat files in the standard format:

```
[DD/MM/YYYY, HH:MM:SS AM/PM] Sender Name: Message text
```

Example:
```
[12/01/2024, 10:30:45 AM] Alice: Hey, how are you?
[12/01/2024, 10:31:12 AM] Bob: I'm doing great! Thanks for asking.
```

The parser supports both 12-hour format (with AM/PM) and 24-hour format.

## How Phase Detection Works

### Time Gap Method
Identifies phase breaks when there's a significant time gap (default: 24 hours) between consecutive messages. This is useful for detecting periods when the conversation was inactive.

### Activity Method
Analyzes message activity patterns over time windows and identifies significant drops in activity. This can detect more subtle phase transitions based on conversation intensity.

### Hybrid Method
Combines both time gaps and activity patterns to create a more robust phase detection that captures both inactivity periods and intensity changes.

## Phase Classification

Phases are automatically classified based on their message rate:
- **High Activity**: > 20 messages/hour
- **Active Discussion**: 10-20 messages/hour
- **Regular Chat**: 3-10 messages/hour
- **Casual Conversation**: 0.5-3 messages/hour
- **Slow Period**: < 0.5 messages/hour

## Output

The tool generates:
1. Console output with phase summaries
2. Visualization files (PNG format) if `--output` is specified or `--no-display` is used
3. Interactive plots if running in a display-enabled environment

## Example Output

```
Parsing WhatsApp messages...
✓ Parsed 1234 messages
✓ Participants: Alice, Bob, Charlie
✓ Time span: 2024-01-01 08:00 to 2024-01-15 22:30

Detecting phases using method: time_gap...
✓ Detected 5 phases

Phase Summary:
--------------------------------------------------------------------------------

Phase 1: Regular Chat
  Period: 2024-01-01 08:00 to 2024-01-03 14:30
  Duration: 54.50 hours
  Messages: 234
  Dominant sender: Alice
  Rate: 4.29 msg/hour
...
```

## License

See LICENSE file for details.

## Troubleshooting

**Issue**: "No messages found in file"
- Make sure your WhatsApp export file is in the correct format
- Check that the file is not empty
- Verify the date/time format matches the expected pattern

**Issue**: "No phases detected"
- Try reducing `--min-messages` (e.g., `--min-messages 1`)
- Try reducing `--min-gap` if using time_gap method
- Try using `--method activity` or `--method hybrid`

**Issue**: Visualization errors
- Ensure matplotlib is properly installed: `pip install matplotlib`
- If running on a server without display, use `--no-display` flag

