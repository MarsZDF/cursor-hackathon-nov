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


class ConversationVisualizer:
    """Creates visualizations for conversation phases."""
    
    def __init__(self, messages: List[Message], phases: List[Phase]):
        self.messages = messages
        self.phases = phases
        self.colors = plt.cm.Set3(np.linspace(0, 1, len(phases) if phases else 1))
    
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
        """Plot phase blocks on a timeline."""
        if not self.phases:
            ax.text(0.5, 0.5, 'No phases detected', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return
        
        for i, phase in enumerate(self.phases):
            start = mdates.date2num(phase.start_time)
            width = mdates.date2num(phase.end_time) - start
            height = 0.8
            
            rect = Rectangle((start, 0.1), width, height, 
                           facecolor=self.colors[i % len(self.colors)],
                           edgecolor='black', alpha=0.7, linewidth=1.5)
            ax.add_patch(rect)
            
            # Add phase label
            center_x = start + width / 2
            label = f"Phase {i+1}\n{phase.phase_type}\n({phase.message_count} msgs)"
            ax.text(center_x, 0.5, label, ha='center', va='center',
                   fontsize=9, weight='bold', wrap=True)
        
        # Format x-axis
        if self.phases:
            ax.set_xlim(
                mdates.date2num(self.phases[0].start_time),
                mdates.date2num(self.phases[-1].end_time)
            )
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.set_ylim(0, 1)
        ax.set_ylabel('Phases', fontsize=12, weight='bold')
        ax.set_title('Conversation Phases Timeline', fontsize=14, weight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='x')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_message_activity(self, ax):
        """Plot message activity as a scatter/line plot over time."""
        if not self.messages:
            return
        
        timestamps = [msg.timestamp for msg in self.messages]
        participants = list(set(msg.sender for msg in self.messages if not msg.is_system))
        participant_colors = plt.cm.tab10(np.linspace(0, 1, len(participants)))
        color_map = {p: participant_colors[i] for i, p in enumerate(participants)}
        
        # Plot messages by participant
        for participant in participants:
            participant_messages = [msg for msg in self.messages if msg.sender == participant]
            if not participant_messages:
                continue
            participant_times = [msg.timestamp for msg in participant_messages]
            y_positions = np.full(len(participant_times), participants.index(participant))
            
            ax.scatter(participant_times, y_positions, 
                      c=[color_map[participant]], label=participant,
                      alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
        
        # Add phase background shading
        for i, phase in enumerate(self.phases):
            ax.axvspan(phase.start_time, phase.end_time,
                      alpha=0.1, color=self.colors[i % len(self.colors)],
                      label=f'Phase {i+1}' if i == 0 else '')
        
        # Format axes
        ax.set_xlabel('Time', fontsize=12, weight='bold')
        ax.set_ylabel('Participant', fontsize=12, weight='bold')
        ax.set_title('Message Activity Over Time', fontsize=14, weight='bold', pad=15)
        ax.set_yticks(range(len(participants)))
        ax.set_yticklabels(participants)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def plot_phase_statistics(self, figsize: tuple = (12, 6), save_path: Optional[str] = None):
        """Create bar charts showing statistics for each phase."""
        if not self.phases:
            print("No phases to visualize")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        phase_labels = [f'Phase {i+1}' for i in range(len(self.phases))]
        message_counts = [phase.message_count for phase in self.phases]
        durations = [phase.duration_hours for phase in self.phases]
        message_rates = [phase.message_count / max(phase.duration_hours, 0.1) 
                        for phase in self.phases]
        
        # Plot 1: Message counts
        axes[0].bar(phase_labels, message_counts, color=self.colors[:len(self.phases)], 
                   edgecolor='black', alpha=0.7)
        axes[0].set_title('Messages per Phase', fontsize=12, weight='bold')
        axes[0].set_ylabel('Number of Messages', fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Duration
        axes[1].bar(phase_labels, durations, color=self.colors[:len(self.phases)],
                   edgecolor='black', alpha=0.7)
        axes[1].set_title('Phase Duration', fontsize=12, weight='bold')
        axes[1].set_ylabel('Duration (hours)', fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 3: Message rate
        axes[2].bar(phase_labels, message_rates, color=self.colors[:len(self.phases)],
                   edgecolor='black', alpha=0.7)
        axes[2].set_title('Message Rate', fontsize=12, weight='bold')
        axes[2].set_ylabel('Messages per Hour', fontsize=10)
        axes[2].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Statistics visualization saved to {save_path}")
        else:
            plt.show()
    
    def plot_phase_summary(self, figsize: tuple = (16, 10), save_path: Optional[str] = None):
        """Create a comprehensive summary visualization."""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :])  # Phase timeline (full width)
        ax2 = fig.add_subplot(gs[1, 0])  # Message counts
        ax3 = fig.add_subplot(gs[1, 1])  # Duration
        ax4 = fig.add_subplot(gs[2, :])  # Activity timeline (full width)
        
        # Phase timeline
        self._plot_phase_timeline(ax1)
        
        # Statistics
        if self.phases:
            phase_labels = [f'Phase {i+1}' for i in range(len(self.phases))]
            message_counts = [phase.message_count for phase in self.phases]
            durations = [phase.duration_hours for phase in self.phases]
            
            ax2.bar(phase_labels, message_counts, color=self.colors[:len(self.phases)],
                   edgecolor='black', alpha=0.7)
            ax2.set_title('Messages per Phase', fontsize=11, weight='bold')
            ax2.set_ylabel('Count', fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            ax3.bar(phase_labels, durations, color=self.colors[:len(self.phases)],
                   edgecolor='black', alpha=0.7)
            ax3.set_title('Phase Duration', fontsize=11, weight='bold')
            ax3.set_ylabel('Hours', fontsize=10)
            ax3.grid(True, alpha=0.3, axis='y')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Activity timeline
        self._plot_message_activity(ax4)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary visualization saved to {save_path}")
        else:
            plt.show()

