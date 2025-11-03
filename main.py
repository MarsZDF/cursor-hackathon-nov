"""
Main script for WhatsApp conversation phase analysis.
Usage: python main.py <whatsapp_export_file.txt> [options]
"""

import argparse
import sys
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
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Parse WhatsApp messages
    print("Parsing WhatsApp messages...")
    whatsapp_parser = WhatsAppParser()
    try:
        messages = whatsapp_parser.parse(str(input_path))
        print(f"✓ Parsed {len(messages)} messages")
    except Exception as e:
        print(f"Error parsing file: {e}", file=sys.stderr)
        sys.exit(1)
    
    if len(messages) == 0:
        print("Error: No messages found in file", file=sys.stderr)
        sys.exit(1)
    
    # Display conversation info
    participants = whatsapp_parser.get_participants()
    start_time, end_time = whatsapp_parser.get_time_span()
    print(f"✓ Participants: {', '.join(participants)}")
    print(f"✓ Time span: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Detect phases
    print(f"\nDetecting phases using method: {args.method}...")
    phase_detector = PhaseDetector(messages)
    try:
        phases = phase_detector.detect_phases(
            min_gap_hours=args.min_gap,
            min_messages_per_phase=args.min_messages,
            method=args.method
        )
        print(f"✓ Detected {len(phases)} phases")
    except Exception as e:
        print(f"Error detecting phases: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Display phase information
    print("\nPhase Summary:")
    print("-" * 80)
    for i, phase in enumerate(phases, 1):
        print(f"\nPhase {i}: {phase.phase_type}")
        print(f"  Period: {phase.start_time.strftime('%Y-%m-%d %H:%M')} to "
              f"{phase.end_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Duration: {phase.duration_hours:.2f} hours")
        print(f"  Messages: {phase.message_count}")
        print(f"  Dominant sender: {phase.dominant_sender}")
        print(f"  Rate: {phase.message_count / max(phase.duration_hours, 0.1):.2f} msg/hour")
    
    # Create visualizations
    if phases:
        print("\nGenerating visualizations...")
        visualizer = ConversationVisualizer(messages, phases)
        
        output_dir = Path(args.output) if args.output else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = input_path.stem
        
        try:
            if args.visualize in ['timeline', 'all']:
                save_path = str(output_dir / f"{base_name}_timeline.png") if args.no_display else None
                visualizer.plot_phases_timeline(save_path=save_path)
                if save_path:
                    print(f"✓ Saved timeline visualization")
            
            if args.visualize in ['stats', 'all']:
                save_path = str(output_dir / f"{base_name}_statistics.png") if args.no_display else None
                visualizer.plot_phase_statistics(save_path=save_path)
                if save_path:
                    print(f"✓ Saved statistics visualization")
            
            if args.visualize in ['summary', 'all']:
                save_path = str(output_dir / f"{base_name}_summary.png") if args.no_display else None
                visualizer.plot_phase_summary(save_path=save_path)
                if save_path:
                    print(f"✓ Saved summary visualization")
            
            print("\n✓ Analysis complete!")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    else:
        print("\nNo phases detected for visualization.")


if __name__ == '__main__':
    main()

