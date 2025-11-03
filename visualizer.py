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
    
    def _plot_phase_list(self, ax):
        """Plot phases as a multi-column list with one-sentence summaries."""
        if not self.phases:
            ax.text(0.5, 0.5, 'No phases detected', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return
        
        ax.set_facecolor('#F5F5F5')
        ax.axis('off')  # Remove axes for clean list display
        
        # Calculate number of columns based on number of phases
        num_phases = len(self.phases)
        if num_phases <= 6:
            num_cols = 2
        elif num_phases <= 12:
            num_cols = 3
        else:
            num_cols = 4
        
        num_rows = (num_phases + num_cols - 1) // num_cols  # Ceiling division
        
        # Calculate spacing with padding
        col_width = 0.92 / num_cols
        row_height = 0.88 / num_rows
        
        # Plot each phase
        for idx, phase in enumerate(self.phases):
            col = idx % num_cols
            row = idx // num_cols
            
            # Calculate position (from top-left, going left to right, top to bottom)
            x_pos = 0.04 + col * col_width + col_width * 0.05
            y_pos = 0.93 - (row + 1) * row_height + row_height * 0.12
            
            # Get summary sentence
            summary = phase.summary_sentence if phase.summary_sentence else "General conversation period."
            
            # Truncate if too long (keep it to one sentence)
            if len(summary) > 120:
                # Find last complete sentence
                sentences = summary.split('.')
                if len(sentences) > 1:
                    summary = '.'.join(sentences[:1]) + '.'
                else:
                    summary = summary[:117] + '...'
            
            # Format phase number and date
            duration_days = phase.duration_hours / 24
            if duration_days >= 1:
                duration_str = f"{duration_days:.1f} days"
            else:
                duration_str = f"{phase.duration_hours:.1f} hours"
            
            date_str = phase.start_time.strftime('%b %d, %Y')
            
            # Create phase text
            phase_text = f"{phase.mood_emoji} Phase {idx+1}: {date_str} ({duration_str})\n"
            phase_text += f"💬 {phase.message_count} messages\n"
            phase_text += f"📝 {summary}"
            
            # Add colored box background
            color = self.colors[idx % len(self.colors)]
            box_height = row_height * 0.75
            rect = Rectangle((x_pos - 0.015, y_pos - box_height), 
                           col_width * 0.85, box_height,
                           facecolor=color, edgecolor='white', 
                           alpha=0.25, linewidth=1.5, transform=ax.transAxes)
            ax.add_patch(rect)
            
            # Add text
            ax.text(x_pos, y_pos, phase_text, 
                   transform=ax.transAxes,
                   fontsize=8.5, weight='normal',
                   color='#2C3E50',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='white', alpha=0.95, 
                            edgecolor=color, linewidth=1.5))
        
        # Add title
        ax.text(0.5, 0.98, '📊 Conversation Phases', 
               transform=ax.transAxes,
               ha='center', va='top',
               fontsize=16, weight='bold', color='#2C3E50')
    
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
        ax3a = fig.add_subplot(gs[2, 0])  # Peak hours (left)
        ax3b = fig.add_subplot(gs[2, 1])  # Activity heatmap (right)
        ax4 = fig.add_subplot(gs[3, :])  # Activity timeline (full width)
        
        # Phase list (multi-column)
        self._plot_phase_list(ax1)
        
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
            
            # Add phase labels with better readability
            for i, (phase, x, y) in enumerate(zip(self.phases, sentiments, normalized_energy)):
                # Add background box for better text readability
                text = ax2.text(x, y, f"{phase.mood_emoji}\nPhase {i+1}", 
                        ha='center', va='center', fontsize=10, weight='bold',
                        color='#2C3E50', zorder=25,
                        bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='white', 
                                edgecolor=self.colors[i % len(self.colors)],
                                linewidth=2,
                                alpha=0.9))
            
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
        
        # Activity patterns: Peak hours and heatmap
        self._plot_peak_hours(ax3a)
        self._plot_activity_heatmap(ax3b)
        
        # Activity timeline
        self._plot_message_activity(ax4)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Summary visualization saved to {save_path}")
        else:
            plt.show()
    
    def plot_usage_insights(self, figsize: tuple = (16, 10), save_path: Optional[str] = None):
        """
        Create a user-friendly visualization showing usage patterns and insights.
        Perfect for casual users to understand their messaging habits.
        """
        if not self.messages:
            print("No messages to visualize")
            return
        
        fig = plt.figure(figsize=figsize, facecolor='white')
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        
        # 1. Peak Activity Hours (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_peak_hours(ax1)
        
        # 2. Day of Week Activity (Top Right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_day_of_week_activity(ax2)
        
        # 3. Activity Heatmap (Bottom Left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_activity_heatmap(ax3)
        
        # 4. Quick Stats (Bottom Right)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_quick_stats(ax4)
        
        fig.suptitle('📱 Your WhatsApp Usage Insights', fontsize=18, weight='bold', 
                    color='#2C3E50', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Usage insights visualization saved to {save_path}")
        else:
            plt.show()
    
    def _plot_peak_hours(self, ax):
        """Plot message activity by hour of day."""
        hour_counts = defaultdict(int)
        for msg in self.messages:
            if not msg.is_system:
                hour = msg.timestamp.hour
                hour_counts[hour] += 1
        
        # Get all hours (0-23) and their counts
        all_hours = list(range(24))
        counts = [hour_counts[h] for h in all_hours]
        
        # Create color gradient based on activity level
        max_count = max(counts) if counts else 1
        colors = []
        for count in counts:
            if count == 0:
                colors.append('#E8E8E8')  # Light gray for no activity
            elif count == max_count:
                colors.append('#FF6B6B')  # Red for peak hour
            else:
                # Gradient from yellow to orange based on relative activity
                intensity = count / max_count
                colors.append(plt.cm.YlOrRd(0.4 + 0.6 * intensity))
        
        bars = ax.bar(all_hours, counts, color=colors, alpha=0.85, 
                     edgecolor='white', linewidth=1.5)
        
        # Highlight peak hour with stronger styling
        if max_count > 0:
            max_idx = counts.index(max_count)
            bars[max_idx].set_color('#FF6B6B')
            bars[max_idx].set_edgecolor('#2C3E50')
            bars[max_idx].set_linewidth(3)
            bars[max_idx].set_alpha(1.0)
            bars[max_idx].set_zorder(10)
        
        ax.set_xlabel('Hour of Day', fontsize=12, weight='bold', color='#34495E')
        ax.set_ylabel('Messages', fontsize=12, weight='bold', color='#34495E')
        ax.set_title('⏰ Peak Activity Hours', fontsize=13, weight='bold', 
                    pad=15, color='#2C3E50')
        ax.set_xticks(range(0, 24, 3))  # Show every 3 hours instead of 2 for better readability
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 3)], 
                          rotation=0, ha='center', fontsize=10, weight='bold')
        ax.grid(True, alpha=0.2, axis='y', linestyle='--', zorder=0)
        ax.set_facecolor('#FAFAFA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Adjust bottom margin to prevent label cutoff
        ax.tick_params(axis='x', pad=8)
        
        # Add value labels on bars (only for non-zero values)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=9, color='#2C3E50')
    
    def _plot_day_of_week_activity(self, ax):
        """Plot message activity by day of week."""
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_counts = defaultdict(int)
        
        for msg in self.messages:
            if not msg.is_system:
                day = msg.timestamp.weekday()  # 0=Monday, 6=Sunday
                day_counts[day] += 1
        
        days = list(range(7))
        counts = [day_counts[d] for d in days]
        
        colors = ['#FF6B6B' if counts[i] == max(counts) else '#4ECDC4' 
                 for i in range(7)]
        
        bars = ax.bar(day_names, counts, color=colors, alpha=0.8, 
                     edgecolor='white', linewidth=2)
        
        ax.set_xlabel('Day of Week', fontsize=11, weight='bold', color='#34495E')
        ax.set_ylabel('Messages', fontsize=11, weight='bold', color='#34495E')
        ax.set_title('📅 Most Active Days', fontsize=13, weight='bold', 
                    pad=15, color='#2C3E50')
        ax.grid(True, alpha=0.2, axis='y', linestyle='--')
        ax.set_facecolor('#FAFAFA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=10)
    
    def _plot_activity_heatmap(self, ax):
        """Create a heatmap showing activity by day of week and hour of day."""
        # Initialize 7x24 grid (days x hours)
        heatmap_data = np.zeros((7, 24))
        
        for msg in self.messages:
            if not msg.is_system:
                day = msg.timestamp.weekday()  # 0=Monday, 6=Sunday
                hour = msg.timestamp.hour
                heatmap_data[day, hour] += 1
        
        # Create heatmap with enhanced styling
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', 
                      interpolation='nearest', alpha=0.9, vmin=0)
        
        # Set labels
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax.set_xticks(range(0, 24, 4))  # Show every 4 hours for better readability
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 4)], 
                          fontsize=10, weight='bold', rotation=0, ha='center')
        ax.set_yticks(range(7))
        ax.set_yticklabels(day_names, fontsize=11, weight='bold', va='center')
        
        ax.set_xlabel('Hour of Day', fontsize=12, weight='bold', color='#34495E')
        ax.set_ylabel('Day of Week', fontsize=12, weight='bold', color='#34495E')
        ax.set_title('🔥 Activity Heatmap', fontsize=13, weight='bold', 
                    pad=15, color='#2C3E50')
        
        # Add colorbar with better styling
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Messages', fontsize=11, weight='bold', color='#34495E')
        cbar.ax.tick_params(labelsize=10)
        
        # Adjust padding to prevent label cutoff
        ax.tick_params(axis='x', pad=8)
        ax.tick_params(axis='y', pad=5)
        
        # Add grid lines for better readability
        ax.set_xticks(range(24), minor=True)
        ax.set_yticks(range(7), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Find and highlight peak cell
        max_val = np.max(heatmap_data)
        if max_val > 0:
            max_day, max_hour = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
            # Add a subtle border around the peak cell
            rect = Rectangle((max_hour - 0.5, max_day - 0.5), 1, 1,
                           fill=False, edgecolor='#FF6B6B', linewidth=3, zorder=10)
            ax.add_patch(rect)
    
    def _plot_quick_stats(self, ax):
        """Display quick statistics in a clean format."""
        ax.axis('off')
        
        # Calculate stats
        total_messages = len([m for m in self.messages if not m.is_system])
        total_phases = len(self.phases)
        
        # Calculate time span
        if self.messages:
            timestamps = [m.timestamp for m in self.messages if not m.is_system]
            time_span = (max(timestamps) - min(timestamps)).days
            if time_span == 0:
                time_span_text = "Less than 1 day"
            elif time_span == 1:
                time_span_text = "1 day"
            else:
                time_span_text = f"{time_span} days"
        else:
            time_span_text = "N/A"
        
        # Calculate average messages per day
        if time_span > 0:
            avg_per_day = total_messages / time_span
        else:
            avg_per_day = total_messages
        
        # Find most active participant
        participant_counts = defaultdict(int)
        for msg in self.messages:
            if not msg.is_system:
                participant_counts[msg.sender] += 1
        
        most_active = max(participant_counts.items(), key=lambda x: x[1])[0] if participant_counts else "N/A"
        
        # Find peak hour
        hour_counts = defaultdict(int)
        for msg in self.messages:
            if not msg.is_system:
                hour_counts[msg.timestamp.hour] += 1
        
        peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else "N/A"
        peak_hour_text = f"{peak_hour:02d}:00" if isinstance(peak_hour, int) else "N/A"
        
        # Create text display
        stats_text = f"""
📊 Quick Stats

💬 Total Messages: {total_messages:,}
📅 Conversation Span: {time_span_text}
📈 Messages/Day: {avg_per_day:.1f}
👤 Most Active: {most_active}
⏰ Peak Hour: {peak_hour_text}
🎭 Conversation Phases: {total_phases}
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=13, weight='bold',
               verticalalignment='center', family='monospace',
               color='#2C3E50', bbox=dict(boxstyle='round,pad=1',
               facecolor='#F8F9FA', edgecolor='#E0E0E0', linewidth=2))
        
        ax.set_title('📈 Conversation Overview', fontsize=13, weight='bold', 
                    pad=15, color='#2C3E50')

