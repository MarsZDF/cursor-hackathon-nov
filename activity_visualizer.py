"""
Activity Visualization Module
Creates focused visualizations for activity patterns - peak hours and heatmap.
"""

from datetime import datetime
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from whatsapp_parser import Message


class ActivityVisualizer:
    """Creates focused activity pattern visualizations."""
    
    def __init__(self, messages: List[Message]):
        self.messages = messages
    
    def plot_activity_patterns(self, figsize: tuple = (16, 7), save_path: Optional[str] = None):
        """
        Create a visualization showing peak activity hours and activity heatmap.
        Perfect for understanding when conversations are most active.
        """
        if not self.messages:
            print("No messages to visualize")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='white')
        
        # Left: Peak Activity Hours
        self._plot_peak_hours(ax1)
        
        # Right: Activity Heatmap
        self._plot_activity_heatmap(ax2)
        
        fig.suptitle('📱 Activity Patterns', fontsize=20, weight='bold', 
                    color='#2C3E50', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Activity patterns visualization saved to {save_path}")
        else:
            plt.show()
    
    def _plot_peak_hours(self, ax):
        """Plot message activity by hour of day with enhanced styling."""
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
        
        ax.set_xlabel('Hour of Day', fontsize=13, weight='bold', color='#34495E')
        ax.set_ylabel('Number of Messages', fontsize=13, weight='bold', color='#34495E')
        ax.set_title('⏰ Peak Activity Hours', fontsize=16, weight='bold', 
                    pad=20, color='#2C3E50')
        
        # Set x-axis ticks every 2 hours
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], 
                          rotation=45, ha='right', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.2, axis='y', linestyle='--', zorder=0)
        ax.set_facecolor('#FAFAFA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E0E0E0')
        ax.spines['bottom'].set_color('#E0E0E0')
        
        # Add value labels on bars (only for non-zero values)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=10, color='#2C3E50')
        
        # Add peak hour annotation
        if max_count > 0:
            peak_hour = counts.index(max_count)
            ax.annotate(f'Peak: {peak_hour:02d}:00',
                       xy=(peak_hour, max_count),
                       xytext=(peak_hour, max_count + max_count * 0.1),
                       ha='center', fontsize=11, weight='bold',
                       color='#FF6B6B',
                       arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=2))
    
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
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                    'Friday', 'Saturday', 'Sunday']
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], fontsize=10)
        ax.set_yticks(range(7))
        ax.set_yticklabels(day_names, fontsize=11, weight='bold')
        
        ax.set_xlabel('Hour of Day', fontsize=13, weight='bold', color='#34495E')
        ax.set_ylabel('Day of Week', fontsize=13, weight='bold', color='#34495E')
        ax.set_title('🔥 Activity Heatmap', fontsize=16, weight='bold', 
                    pad=20, color='#2C3E50')
        
        # Add colorbar with better styling
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Messages', fontsize=12, weight='bold', color='#34495E')
        cbar.ax.tick_params(labelsize=10)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add grid lines for better readability
        ax.set_xticks(range(24), minor=True)
        ax.set_yticks(range(7), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Find and highlight peak cell
        max_val = np.max(heatmap_data)
        if max_val > 0:
            max_day, max_hour = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
            # Add a subtle border around the peak cell
            rect = plt.Rectangle((max_hour - 0.5, max_day - 0.5), 1, 1,
                               fill=False, edgecolor='#FF6B6B', linewidth=3, zorder=10)
            ax.add_patch(rect)

