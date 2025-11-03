"""
Visualization Module
Creates visualizations of conversation phases.
"""

from datetime import datetime
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
from whatsapp_parser import Message
from phase_detector import Phase
from sentiment_analyzer import SentimentAnalyzer


class ConversationVisualizer:
    """Creates visualizations for conversation phases."""
    
    def __init__(self, messages: List[Message], phases: List[Phase]):
        self.messages = messages
        self.phases = phases
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Use mood-based colors instead of generic palette
        if phases:
            self.colors = [self.sentiment_analyzer.get_mood_color({
                'mood': p.mood,
                'sentiment': p.sentiment,
                'excitement': 0.0,
                'engagement': 0.0
            }) for p in phases]
        else:
            self.colors = ['#D3D3D3']
    
    def plot_phases_timeline(self, figsize: tuple = (14, 8), save_path: Optional[str] = None):
        """
        Create a timeline visualization showing phases and message activity.
        
        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 2])
        
        # Top subplot: Phase timeline
        self._plot_phase_timeline(ax1)
        
        # Bottom subplot: Message activity over time
        self._plot_message_activity(ax2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def _plot_phase_timeline(self, ax):
        """Plot phase blocks on a timeline with mood-based styling."""
        if not self.phases:
            ax.text(0.5, 0.5, 'No phases detected', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return
        
        # Set background color to match overall vibe
        ax.set_facecolor('#F5F5F5')
        
        for i, phase in enumerate(self.phases):
            start = mdates.date2num(phase.start_time)
            width = mdates.date2num(phase.end_time) - start
            height = 0.8
            
            # Use mood-based color with gradient effect
            color = self.colors[i]
            
            rect = Rectangle((start, 0.1), width, height, 
                           facecolor=color,
                           edgecolor='white', alpha=0.85, linewidth=2,
                           joinstyle='round')
            ax.add_patch(rect)
            
            # Add phase label with emoji
            center_x = start + width / 2
            label = f"{phase.mood_emoji}\n{phase.phase_type}\n\n💬 {phase.message_count} messages"
            ax.text(center_x, 0.5, label, ha='center', va='center',
                   fontsize=10, weight='bold', color='#2C3E50',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Format x-axis
        if self.phases:
            ax.set_xlim(
                mdates.date2num(self.phases[0].start_time),
                mdates.date2num(self.phases[-1].end_time)
            )
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.set_ylim(0, 1)
        ax.set_ylabel('Conversation Vibes ✨', fontsize=13, weight='bold', color='#34495E')
        ax.set_title('📊 Your Conversation Journey', fontsize=16, weight='bold', 
                    pad=20, color='#2C3E50')
        ax.grid(True, alpha=0.2, axis='x', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    
    def _plot_message_activity(self, ax):
        """Plot message activity as a scatter/line plot over time with mood shading."""
        if not self.messages:
            return
        
        ax.set_facecolor('#FAFAFA')
        
        timestamps = [msg.timestamp for msg in self.messages]
        participants = list(set(msg.sender for msg in self.messages if not msg.is_system))
        
        # Use a more vibrant color palette
        vibrant_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22']
        color_map = {p: vibrant_colors[i % len(vibrant_colors)] for i, p in enumerate(participants)}
        
        # Add phase background shading first (behind everything)
        for i, phase in enumerate(self.phases):
            ax.axvspan(phase.start_time, phase.end_time,
                      alpha=0.15, color=self.colors[i % len(self.colors)],
                      label=f'{phase.mood_emoji} {phase.mood}' if i == 0 else '')
        
        # Plot messages by participant with larger, more visible points
        for participant in participants:
            participant_messages = [msg for msg in self.messages if msg.sender == participant]
            if not participant_messages:
                continue
            participant_times = [msg.timestamp for msg in participant_messages]
            y_positions = np.full(len(participant_times), participants.index(participant))
            
            # Add some jitter to y-positions for better visibility
            y_positions = y_positions + np.random.normal(0, 0.05, len(y_positions))
            
            ax.scatter(participant_times, y_positions, 
                      c=[color_map[participant]], label=f'👤 {participant}',
                      alpha=0.7, s=50, edgecolors='white', linewidths=1.5,
                      zorder=10)
        
        # Format axes
        ax.set_xlabel('📅 Time', fontsize=12, weight='bold', color='#34495E')
        ax.set_ylabel('💬 Participants', fontsize=12, weight='bold', color='#34495E')
        ax.set_title('💭 Message Flow Over Time', fontsize=15, weight='bold', pad=15, color='#2C3E50')
        ax.set_yticks(range(len(participants)))
        ax.set_yticklabels([f'👤 {p}' for p in participants], fontsize=10)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, 
                 framealpha=0.9, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.2, axis='x', linestyle='--', zorder=0)
        ax.grid(True, alpha=0.1, axis='y', linestyle=':', zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    
    def plot_phase_statistics(self, figsize: tuple = (14, 6), save_path: Optional[str] = None):
        """Create bar charts showing statistics for each phase with mood indicators."""
        if not self.phases:
            print("No phases to visualize")
            return
        
        fig = plt.figure(figsize=figsize, facecolor='white')
        axes = [fig.add_subplot(1, 3, i+1) for i in range(3)]
        
        phase_labels = [f"{p.mood_emoji}\nPhase {i+1}\n{p.mood}" for i, p in enumerate(self.phases)]
        message_counts = [phase.message_count for phase in self.phases]
        durations = [phase.duration_hours for phase in self.phases]
        message_rates = [phase.message_count / max(phase.duration_hours, 0.1) 
                        for phase in self.phases]
        
        # Plot 1: Message counts
        bars1 = axes[0].bar(range(len(phase_labels)), message_counts, 
                           color=self.colors[:len(self.phases)], 
                           edgecolor='white', alpha=0.85, linewidth=2)
        axes[0].set_xticks(range(len(phase_labels)))
        axes[0].set_xticklabels(phase_labels, fontsize=9, ha='center')
        axes[0].set_title('💬 Message Volume', fontsize=13, weight='bold', pad=15, color='#2C3E50')
        axes[0].set_ylabel('Messages', fontsize=11, weight='bold')
        axes[0].grid(True, alpha=0.2, axis='y', linestyle='--')
        axes[0].set_facecolor('#FAFAFA')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Duration
        bars2 = axes[1].bar(range(len(phase_labels)), durations,
                           color=self.colors[:len(self.phases)],
                           edgecolor='white', alpha=0.85, linewidth=2)
        axes[1].set_xticks(range(len(phase_labels)))
        axes[1].set_xticklabels(phase_labels, fontsize=9, ha='center')
        axes[1].set_title('⏱️ Conversation Duration', fontsize=13, weight='bold', pad=15, color='#2C3E50')
        axes[1].set_ylabel('Hours', fontsize=11, weight='bold')
        axes[1].grid(True, alpha=0.2, axis='y', linestyle='--')
        axes[1].set_facecolor('#FAFAFA')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}h', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Message rate
        bars3 = axes[2].bar(range(len(phase_labels)), message_rates,
                           color=self.colors[:len(self.phases)],
                           edgecolor='white', alpha=0.85, linewidth=2)
        axes[2].set_xticks(range(len(phase_labels)))
        axes[2].set_xticklabels(phase_labels, fontsize=9, ha='center')
        axes[2].set_title('⚡ Energy Level', fontsize=13, weight='bold', pad=15, color='#2C3E50')
        axes[2].set_ylabel('Messages/Hour', fontsize=11, weight='bold')
        axes[2].grid(True, alpha=0.2, axis='y', linestyle='--')
        axes[2].set_facecolor('#FAFAFA')
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Statistics visualization saved to {save_path}")
        else:
            plt.show()
    
    def plot_phase_summary(self, figsize: tuple = (16, 10), save_path: Optional[str] = None):
        """Create a comprehensive summary visualization with mood-based styling."""
        fig = plt.figure(figsize=figsize, facecolor='white')
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :])  # Phase timeline (full width)
        ax2 = fig.add_subplot(gs[1, 0])  # Message counts
        ax3 = fig.add_subplot(gs[1, 1])  # Duration
        ax4 = fig.add_subplot(gs[2, :])  # Activity timeline (full width)
        
        # Phase timeline
        self._plot_phase_timeline(ax1)
        
        # Statistics with mood labels
        if self.phases:
            phase_labels = [f"{p.mood_emoji}\nPhase {i+1}" for i, p in enumerate(self.phases)]
            message_counts = [phase.message_count for phase in self.phases]
            durations = [phase.duration_hours for phase in self.phases]
            
            bars2 = ax2.bar(range(len(phase_labels)), message_counts, 
                           color=self.colors[:len(self.phases)],
                           edgecolor='white', alpha=0.85, linewidth=2)
            ax2.set_xticks(range(len(phase_labels)))
            ax2.set_xticklabels(phase_labels, fontsize=10, ha='center')
            ax2.set_title('💬 Message Volume', fontsize=12, weight='bold', pad=12, color='#2C3E50')
            ax2.set_ylabel('Messages', fontsize=10, weight='bold')
            ax2.grid(True, alpha=0.2, axis='y', linestyle='--')
            ax2.set_facecolor('#FAFAFA')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            bars3 = ax3.bar(range(len(phase_labels)), durations,
                           color=self.colors[:len(self.phases)],
                           edgecolor='white', alpha=0.85, linewidth=2)
            ax3.set_xticks(range(len(phase_labels)))
            ax3.set_xticklabels(phase_labels, fontsize=10, ha='center')
            ax3.set_title('⏱️ Duration', fontsize=12, weight='bold', pad=12, color='#2C3E50')
            ax3.set_ylabel('Hours', fontsize=10, weight='bold')
            ax3.grid(True, alpha=0.2, axis='y', linestyle='--')
            ax3.set_facecolor('#FAFAFA')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            for bar in bars3:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}h', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Activity timeline
        self._plot_message_activity(ax4)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Summary visualization saved to {save_path}")
        else:
            plt.show()

