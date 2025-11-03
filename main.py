"""
Main script for WhatsApp conversation phase analysis.
Usage: python main.py <whatsapp_export_file.txt> [options]
"""

import argparse
import sys
import os
from pathlib import Path
from whatsapp_parser import WhatsAppParser
from phase_detector import PhaseDetector
from visualizer import ConversationVisualizer


def main():
    parser = argparse.ArgumentParser(
        description='Analyze WhatsApp conversations and identify phases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py chat.txt
  python main.py chat.txt --method time_gap --min-gap 12
  python main.py chat.txt --output results/
  python main.py chat.txt --method hybrid --visualize all
        """
    )
    
    parser.add_argument('input_file', type=str, help='Path to WhatsApp exported chat file')
    parser.add_argument('--method', type=str, default='time_gap',
                       choices=['time_gap', 'activity', 'hybrid'],
                       help='Phase detection method (default: time_gap)')
    parser.add_argument('--min-gap', type=float, default=24.0,
                       help='Minimum time gap in hours to split phases (default: 24.0)')
    parser.add_argument('--min-messages', type=int, default=5,
                       help='Minimum messages per phase (default: 5)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for visualizations (default: current directory)')
    parser.add_argument('--visualize', type=str, default='summary',
                       choices=['timeline', 'stats', 'summary', 'all'],
                       help='Type of visualization to generate (default: summary)')
    parser.add_argument('--no-display', action='store_true',
                       help='Save visualizations without displaying them')
    
    args = parser.parse_args()
    
    # Validate and sanitize input file path
    try:
        input_path = Path(args.input_file).resolve()
        # Prevent path traversal by ensuring the resolved path is within allowed directory
        # Check if file exists
        if not input_path.exists():
            print(f"❌ Error: File not found: {args.input_file}", file=sys.stderr)
            sys.exit(1)
        
        # Check if it's actually a file (not a directory)
        if not input_path.is_file():
            print(f"❌ Error: Path is not a file: {args.input_file}", file=sys.stderr)
            sys.exit(1)
        
        # Check file size (limit to 100MB to prevent DoS)
        max_file_size = 100 * 1024 * 1024  # 100MB
        file_size = input_path.stat().st_size
        if file_size > max_file_size:
            print(f"❌ Error: File too large ({file_size / (1024*1024):.1f}MB). Maximum size is 100MB.", file=sys.stderr)
            sys.exit(1)
            
    except (ValueError, OSError) as e:
        print(f"❌ Error: Invalid file path: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Parse WhatsApp messages
    print("\n🔍 Analyzing your conversation...")
    print("=" * 70)
    whatsapp_parser = WhatsAppParser()
    try:
        messages = whatsapp_parser.parse(str(input_path))
        print(f"✨ Found {len(messages)} messages to explore")
    except Exception as e:
        print(f"❌ Error parsing file: {e}", file=sys.stderr)
        sys.exit(1)
    
    if len(messages) == 0:
        print("❌ No messages found in file", file=sys.stderr)
        sys.exit(1)
    
    # Display conversation info
    participants = whatsapp_parser.get_participants()
    start_time, end_time = whatsapp_parser.get_time_span()
    print(f"👥 Participants: {', '.join(participants)}")
    print(f"📅 Timeline: {start_time.strftime('%b %d, %Y at %H:%M')} → {end_time.strftime('%b %d, %Y at %H:%M')}")
    
    # Detect phases
    print(f"\n🎭 Detecting conversation vibes...")
    print("=" * 70)
    phase_detector = PhaseDetector(messages)
    try:
        phases = phase_detector.detect_phases(
            min_gap_hours=args.min_gap,
            min_messages_per_phase=args.min_messages,
            method=args.method
        )
        print(f"✨ Discovered {len(phases)} distinct conversation phases!\n")
    except Exception as e:
        print(f"❌ Error detecting phases: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Display phase information with mood and emojis
    print("📊 CONVERSATION PHASES")
    print("=" * 70)
    for i, phase in enumerate(phases, 1):
        duration_days = phase.duration_hours / 24
        duration_str = f"{duration_days:.1f} days" if duration_days >= 1 else f"{phase.duration_hours:.1f} hours"
        
        print(f"\n{phase.mood_emoji} Phase {i}: {phase.phase_type}")
        print(f"   🎭 Vibe: {phase.vibe}")
        print(f"   📅 When: {phase.start_time.strftime('%b %d, %Y')} ({duration_str})")
        print(f"   💬 Messages: {phase.message_count}")
        print(f"   👤 Most active: {phase.dominant_sender}")
        print(f"   ⚡ Energy: {phase.message_count / max(phase.duration_hours, 0.1):.1f} messages/hour")
        sentiment_emoji = "😊" if phase.sentiment > 0.1 else "😐" if phase.sentiment > -0.1 else "😔"
        print(f"   {sentiment_emoji} Mood score: {phase.sentiment:+.2f}")
    
    # Create visualizations
    if phases:
        print("\n" + "=" * 70)
        print("🎨 Creating beautiful visualizations...")
        print("=" * 70)
        visualizer = ConversationVisualizer(messages, phases)
        
        # Validate and sanitize output directory
        if args.output:
            try:
                output_dir = Path(args.output).resolve()
                # Prevent path traversal - ensure output stays within current directory tree
                # or at least doesn't escape to system directories
                if '..' in str(output_dir) and not str(output_dir).startswith(str(Path.cwd().resolve())):
                    print(f"❌ Error: Output directory path not allowed", file=sys.stderr)
                    sys.exit(1)
                output_dir.mkdir(parents=True, exist_ok=True)
            except (ValueError, OSError) as e:
                print(f"❌ Error: Invalid output directory: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            output_dir = Path.cwd()
        
        # Sanitize base name to prevent path injection
        base_name = input_path.stem
        # Remove any potentially dangerous characters from filename
        base_name = "".join(c for c in base_name if c.isalnum() or c in ('-', '_')).strip()
        if not base_name:
            base_name = "chat"  # Default if sanitization removes everything
        
        try:
            if args.visualize in ['timeline', 'all']:
                save_path = str(output_dir / f"{base_name}_timeline.png") if args.no_display else None
                visualizer.plot_phases_timeline(save_path=save_path)
                if save_path:
                    print(f"💾 Saved timeline visualization → {save_path}")
            
            if args.visualize in ['stats', 'all']:
                save_path = str(output_dir / f"{base_name}_statistics.png") if args.no_display else None
                visualizer.plot_phase_statistics(save_path=save_path)
                if save_path:
                    print(f"💾 Saved statistics visualization → {save_path}")
            
            if args.visualize in ['summary', 'all']:
                save_path = str(output_dir / f"{base_name}_summary.png") if args.no_display else None
                visualizer.plot_phase_summary(save_path=save_path)
                if save_path:
                    print(f"💾 Saved summary visualization → {save_path}")
            
            print("\n" + "=" * 70)
            print("✨ Analysis complete! Your conversation vibes have been captured!")
            print("=" * 70 + "\n")
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    else:
        print("\n😶 No phases detected for visualization.")


if __name__ == '__main__':
    main()

