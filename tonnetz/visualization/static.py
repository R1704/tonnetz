"""
Static visualization using matplotlib.

This module provides functions to create static plots and visualizations
of the Tonnetz grid and chord progressions.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Optional, Tuple
from tonnetz.core.chord import Chord
from tonnetz.core.tonnetz import ToroidalTonnetz
from tonnetz.automaton.grid import ToroidalGrid


class TonnetzPlotter:
    """
    Static plotting utilities for Tonnetz visualizations.
    """
    
    def __init__(self, figsize: Tuple[float, float] = (12, 10)) -> None:
        """
        Initialize the plotter.
        
        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Color schemes for different chord qualities
        self.chord_colors = {
            'major': '#4CAF50',      # Green
            'minor': '#2196F3',      # Blue  
            'diminished': '#FF5722', # Red-orange
            'augmented': '#FF9800',  # Orange
            'major7': '#8BC34A',     # Light green
            'minor7': '#03A9F4',     # Light blue
            'dominant7': '#FFC107',  # Amber
            'diminished7': '#F44336' # Red
        }
        
        # Default color for unknown qualities
        self.default_color = '#9E9E9E'  # Grey
    
    def plot_grid(self, grid: ToroidalGrid, title: str = "Tonnetz Grid", 
                 save_path: Optional[str] = None, show: bool = True,
                 color_by: str = 'quality') -> plt.Figure:
        """
        Plot the cellular automaton grid.
        
        Args:
            grid: ToroidalGrid to visualize
            title: Plot title
            save_path: Optional path to save the figure
            show: Whether to display the plot
            color_by: How to color cells ('quality', 'root', 'activation')
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create color map based on color_by parameter
        if color_by == 'quality':
            colors = self._get_quality_colors(grid)
        elif color_by == 'root':
            colors = self._get_root_colors(grid)
        elif color_by == 'activation':
            colors = self._get_activation_colors(grid)
        else:
            raise ValueError(f"Unknown color_by option: {color_by}")
        
        # Plot cells as rectangles
        for y in range(grid.height):
            for x in range(grid.width):
                cell = grid.get_cell(x, y)
                color = colors[y][x]
                
                # Create rectangle
                rect = patches.Rectangle((x, y), 1, 1, 
                                       linewidth=1, edgecolor='black', 
                                       facecolor=color, alpha=0.8)
                ax.add_patch(rect)
                
                # Add chord label
                chord_label = self._format_chord_label(cell.chord)
                ax.text(x + 0.5, y + 0.5, chord_label, 
                       ha='center', va='center', fontsize=8, 
                       fontweight='bold', color='white')
        
        # Set up axes
        ax.set_xlim(0, grid.width)
        ax.set_ylim(0, grid.height)
        ax.set_aspect('equal')
        ax.set_title(f"{title} (Generation {grid.generation})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add grid lines
        ax.set_xticks(range(grid.width + 1))
        ax.set_yticks(range(grid.height + 1))
        ax.grid(True, alpha=0.3)
        
        # Add color legend
        self._add_color_legend(ax, color_by)
        
        # Add statistics text
        stats = grid.get_statistics()
        stats_text = (f"Changes: {stats['chord_changes']}\n"
                      f"Active: {stats['active_cells']}\n"
                      f"Harmony: {stats['harmony_level']:.2f}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_chord_evolution(self, grid: ToroidalGrid, x: int, y: int,
                           title: Optional[str] = None, save_path: Optional[str] = None,
                           show: bool = True) -> plt.Figure:
        """
        Plot the evolution of a specific cell over time.
        
        Args:
            grid: ToroidalGrid containing the cell
            x: X coordinate of the cell
            y: Y coordinate of the cell
            title: Optional plot title
            save_path: Optional path to save the figure
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        cell = grid.get_cell(x, y)
        chord_history = cell.chord_history
        
        if len(chord_history) < 2:
            raise ValueError("Not enough history to plot evolution")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot chord root over time
        generations = list(range(len(chord_history)))
        roots = [chord.root for chord in chord_history]
        
        ax1.plot(generations, roots, 'o-', linewidth=2, markersize=6)
        ax1.set_ylabel('Chord Root')
        ax1.set_title(f"Cell ({x}, {y}) Chord Evolution" if title is None else title)
        ax1.set_yticks(range(12))
        ax1.set_yticklabels(self.note_names)
        ax1.grid(True, alpha=0.3)
        
        # Plot chord quality over time (as categorical)
        qualities = [chord.quality for chord in chord_history]
        unique_qualities = list(set(qualities))
        quality_indices = [unique_qualities.index(q) for q in qualities]
        
        ax2.plot(generations, quality_indices, 's-', linewidth=2, markersize=6)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Chord Quality')
        ax2.set_yticks(range(len(unique_qualities)))
        ax2.set_yticklabels(unique_qualities)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_tonnetz_lattice(self, tonnetz: ToroidalTonnetz, chords: Optional[List[Chord]] = None,
                           title: str = "Tonnetz Lattice", save_path: Optional[str] = None,
                           show: bool = True) -> plt.Figure:
        """
        Plot the underlying Tonnetz lattice structure.
        
        Args:
            tonnetz: ToroidalTonnetz object
            chords: Optional list of chords to highlight
            title: Plot title
            save_path: Optional path to save the figure
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create lattice visualization
        lattice = tonnetz.visualize_lattice(chords)
        
        # Plot as heatmap
        im = ax.imshow(lattice, cmap='tab20', interpolation='nearest')
        
        # Add note names
        for y in range(tonnetz.height):
            for x in range(tonnetz.width):
                pitch_class = lattice[y, x]
                if pitch_class >= 0:  # Valid pitch class
                    note_name = self.note_names[pitch_class % 12]
                    ax.text(x, y, note_name, ha='center', va='center',
                           fontsize=10, fontweight='bold', color='white')
        
        ax.set_title(title)
        ax.set_xlabel('X (Coordinate)')
        ax.set_ylabel('Y (Coordinate)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Pitch Class')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_progression(self, progression: List[Chord], title: str = "Chord Progression",
                        save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """
        Plot a chord progression.
        
        Args:
            progression: List of chords in the progression
            title: Plot title
            save_path: Optional path to save the figure
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        if not progression:
            raise ValueError("Empty progression provided")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot chord roots
        positions = list(range(len(progression)))
        roots = [chord.root for chord in progression]
        
        ax1.plot(positions, roots, 'o-', linewidth=2, markersize=8)
        ax1.set_ylabel('Chord Root')
        ax1.set_title(title)
        ax1.set_yticks(range(12))
        ax1.set_yticklabels(self.note_names)
        ax1.grid(True, alpha=0.3)
        
        # Add chord labels
        for i, chord in enumerate(progression):
            ax1.annotate(str(chord), (i, chord.root), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        # Plot voice leading distances
        if len(progression) > 1:
            distances = []
            for i in range(len(progression) - 1):
                distance = progression[i].voice_leading_distance(progression[i + 1])
                distances.append(distance)
            
            ax2.bar(range(1, len(progression)), distances, alpha=0.7)
            ax2.set_xlabel('Chord Transition')
            ax2.set_ylabel('Voice Leading Distance')
            ax2.set_title('Voice Leading Analysis')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_grid_animation_frames(self, grid: ToroidalGrid, rule_func,
                                 num_frames: int = 10, save_dir: str = "./frames/") -> List[str]:
        """
        Generate animation frames for grid evolution.
        
        Args:
            grid: ToroidalGrid to animate
            rule_func: Rule function for evolution
            num_frames: Number of frames to generate
            save_dir: Directory to save frame images
            
        Returns:
            List of frame file paths
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        frame_paths = []
        
        for frame in range(num_frames):
            # Plot current state
            frame_path = os.path.join(save_dir, f"frame_{frame:03d}.png")
            self.plot_grid(grid, title=f"Generation {grid.generation}", 
                          save_path=frame_path, show=False)
            frame_paths.append(frame_path)
            
            # Update grid for next frame
            if frame < num_frames - 1:  # Don't update after last frame
                grid.update(rule_func)
            
            plt.close()  # Close figure to save memory
        
        return frame_paths
    
    def _get_quality_colors(self, grid: ToroidalGrid) -> List[List[str]]:
        """Get color matrix based on chord qualities."""
        colors = []
        for y in range(grid.height):
            row = []
            for x in range(grid.width):
                cell = grid.get_cell(x, y)
                color = self.chord_colors.get(cell.chord.quality, self.default_color)
                row.append(color)
            colors.append(row)
        return colors
    
    def _get_root_colors(self, grid: ToroidalGrid) -> List[List[str]]:
        """Get color matrix based on chord roots."""
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Use a colormap for the 12 pitch classes
        cmap = cm.get_cmap('tab20')
        norm = mcolors.Normalize(vmin=0, vmax=11)
        
        colors = []
        for y in range(grid.height):
            row = []
            for x in range(grid.width):
                cell = grid.get_cell(x, y)
                color = cmap(norm(cell.chord.root))
                row.append(color)
            colors.append(row)
        return colors
    
    def _get_activation_colors(self, grid: ToroidalGrid) -> List[List[str]]:
        """Get color matrix based on cell activation levels."""
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        cmap = cm.get_cmap('plasma')
        norm = mcolors.Normalize(vmin=0, vmax=1)
        
        colors = []
        for y in range(grid.height):
            row = []
            for x in range(grid.width):
                cell = grid.get_cell(x, y)
                color = cmap(norm(cell.activation))
                row.append(color)
            colors.append(row)
        return colors
    
    def _format_chord_label(self, chord: Chord) -> str:
        """Format chord for display in grid cells."""
        note_name = self.note_names[chord.root]
        quality_short = {
            'major': '',
            'minor': 'm',
            'diminished': '°',
            'augmented': '+',
            'major7': 'M7',
            'minor7': 'm7',
            'dominant7': '7',
            'diminished7': '°7'
        }
        quality = quality_short.get(chord.quality, chord.quality[:3])
        return f"{note_name}{quality}"
    
    def _add_color_legend(self, ax: plt.Axes, color_by: str) -> None:
        """Add a color legend to the plot."""
        if color_by == 'quality':
            # Create legend for chord qualities
            legend_elements = []
            for quality, color in self.chord_colors.items():
                legend_elements.append(patches.Patch(color=color, label=quality))
            
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        # For other color_by options, could add different legends


# Convenience functions
def plot_grid(grid: ToroidalGrid, **kwargs) -> plt.Figure:
    """Convenience function to plot a grid."""
    plotter = TonnetzPlotter()
    return plotter.plot_grid(grid, **kwargs)


def plot_progression(progression: List[Chord], **kwargs) -> plt.Figure:
    """Convenience function to plot a progression."""
    plotter = TonnetzPlotter()
    return plotter.plot_progression(progression, **kwargs)


def plot_tonnetz(tonnetz: ToroidalTonnetz, **kwargs) -> plt.Figure:
    """Convenience function to plot a Tonnetz lattice."""
    plotter = TonnetzPlotter()
    return plotter.plot_tonnetz_lattice(tonnetz, **kwargs)
