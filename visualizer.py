"""
Visualization Module
Creates visualizations of conversation phases.
"""

from datetime import datetime, timedelta
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
from collections import defaultdict
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
            
            # Add phase label with emoji and summary
            center_x = start + width / 2
            # Create label with summary sentence if available
            if phase.summary_sentence:
                # Truncate summary if too long for display
                summary_display = phase.summary_sentence
                if len(summary_display) > 60:
                    summary_display = summary_display[:57] + "..."
                label = f"{phase.mood_emoji}\n{summary_display}\n\n💬 {phase.message_count} msgs"
            else:
                label = f"{phase.mood_emoji}\n{phase.phase_type}\n\n💬 {phase.message_count} messages"
            ax.text(center_x, 0.5, label, ha='center', va='center',
                   fontsize=8, weight='bold', color='#2C3E50',
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
        """Plot message activity as aggregated daily counts over time with mood shading."""
        if not self.messages:
            return
        
        ax.set_facecolor('#FAFAFA')
        
        participants = sorted(set(msg.sender for msg in self.messages if not msg.is_system))
        # Use a more vibrant color palette
        vibrant_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22']
        color_map = {p: vibrant_colors[i % len(vibrant_colors)] for i, p in enumerate(participants)}
        
        # Aggregate messages by day for each participant
        daily_counts = {p: defaultdict(int) for p in participants}
        
        for msg in self.messages:
            if msg.is_system:
                continue
            # Get date (without time)
            date_key = msg.timestamp.date()
            daily_counts[msg.sender][date_key] += 1
        
        # Add phase background shading first (behind everything)
        for i, phase in enumerate(self.phases):
            ax.axvspan(mdates.date2num(phase.start_time), mdates.date2num(phase.end_time),
                      alpha=0.15, color=self.colors[i % len(self.colors)],
                      zorder=0, label=f'{phase.mood_emoji} {phase.mood}' if i == 0 else '')
        
        # Plot line for each participant
        for participant in participants:
            dates = sorted(daily_counts[participant].keys())
            counts = [daily_counts[participant][d] for d in dates]
            date_nums = [mdates.date2num(d) for d in dates]
            
            ax.plot(date_nums, counts, marker='o', markersize=5, linewidth=2.5,
                   label=f'👤 {participant}', color=color_map[participant], alpha=0.8, zorder=10)
            ax.fill_between(date_nums, counts, alpha=0.25, color=color_map[participant], zorder=5)
        
        # Format axes
        ax.set_xlabel('📅 Date', fontsize=12, weight='bold', color='#34495E')
        ax.set_ylabel('💬 Messages per Day', fontsize=12, weight='bold', color='#34495E')
        ax.set_title('💭 Daily Message Activity Over Time', fontsize=15, weight='bold', pad=15, color='#2C3E50')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.2, axis='both', linestyle='--', zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
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
    
    def plot_phase_summary(self, figsize: tuple = (18, 12), save_path: Optional[str] = None):
        """Create a comprehensive summary visualization with mood-based styling and topic insights."""
        fig = plt.figure(figsize=figsize, facecolor='white')
        gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3, height_ratios=[1.2, 1, 1, 1.5])
        
        ax1 = fig.add_subplot(gs[0, :])  # Phase timeline (full width)
        ax2 = fig.add_subplot(gs[1, :])  # Mood scatter plot (full width)
        ax3 = fig.add_subplot(gs[2, :])  # Summary sentences per phase
        ax4 = fig.add_subplot(gs[3, :])  # Activity timeline (full width)
        
        # Phase timeline
        self._plot_phase_timeline(ax1)
        
        # Mood scatter plot (pleasant-unpleasant vs energy)
        if self.phases:
            sentiments = [phase.sentiment for phase in self.phases]  # X-axis: -1 (unpleasant) to +1 (pleasant)
            message_rates = [phase.message_count / max(phase.duration_hours, 0.1) for phase in self.phases]  # Y-axis: energy
            
            # Normalize energy (message rates) to 0-1 scale for better visualization
            # Use log scale or percentile for better distribution
            max_rate = max(message_rates) if message_rates else 1.0
            normalized_energy = [rate / max_rate for rate in message_rates]  # 0-1 scale
            
            # Plot scatter
            scatter = ax2.scatter(sentiments, normalized_energy, 
                                 s=[300 + p.message_count / 10 for p in self.phases],  # Size by message count
                                 c=range(len(self.phases)), cmap='viridis', alpha=0.7,
                                 edgecolors='white', linewidths=2, zorder=10)
            
            # Add phase labels
            for i, (phase, x, y) in enumerate(zip(self.phases, sentiments, normalized_energy)):
                ax2.text(x, y, f"{phase.mood_emoji}\nPhase {i+1}", 
                        ha='center', va='center', fontsize=8, weight='bold',
                        color='white', zorder=20)
            
            # Add quadrant labels
            ax2.text(0.7, 0.85, 'High Energy\nPleasant', ha='center', fontsize=10, 
                    style='italic', alpha=0.6, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
            ax2.text(-0.7, 0.85, 'High Energy\nUnpleasant', ha='center', fontsize=10,
                    style='italic', alpha=0.6, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
            ax2.text(0.7, 0.15, 'Low Energy\nPleasant', ha='center', fontsize=10,
                    style='italic', alpha=0.6, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            ax2.text(-0.7, 0.15, 'Low Energy\nUnpleasant', ha='center', fontsize=10,
                    style='italic', alpha=0.6, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
            
            # Format axes
            ax2.set_xlabel('Pleasant ← → Unpleasant (Sentiment)', fontsize=12, weight='bold', color='#34495E')
            ax2.set_ylabel('Low Energy ← → High Energy (Message Rate)', fontsize=12, weight='bold', color='#34495E')
            ax2.set_title('Mood & Energy Scatter Plot', fontsize=14, weight='bold', pad=15, color='#2C3E50')
            ax2.set_xlim(-1.1, 1.1)
            ax2.set_ylim(-0.05, 1.05)
            ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, zorder=0)
            ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.3, zorder=0)
            ax2.grid(True, alpha=0.2, linestyle='--', zorder=0)
            ax2.set_facecolor('#FAFAFA')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
        
        # Summary sentences per phase
        if self.phases:
            ax3.axis('off')
            ax3.set_title('Phase Summaries', fontsize=12, weight='bold', pad=10, color='#2C3E50')
            
            # Create a text display for each phase's summary
            y_positions = np.linspace(0.95, 0.05, len(self.phases))
            for i, (phase, y_pos) in enumerate(zip(self.phases, y_positions)):
                phase_label = f"{phase.mood_emoji} Phase {i+1}:"
                summary = phase.summary_sentence if phase.summary_sentence else "No summary available."
                
                full_text = f"{phase_label} {summary}"
                
                ax3.text(0.02, y_pos, full_text, transform=ax3.transAxes,
                       fontsize=9, weight='bold' if i == 0 else 'normal',
                       bbox=dict(boxstyle='round,pad=0.8', facecolor=self.colors[i % len(self.colors)], 
                               alpha=0.25, edgecolor=self.colors[i % len(self.colors)], linewidth=1.5),
                       wrap=True)
        
        # Activity timeline
        self._plot_message_activity(ax4)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Summary visualization saved to {save_path}")
        else:
            plt.show()

