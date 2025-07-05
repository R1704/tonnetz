# tonnetz/visualization/interactive.py
import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, ColorBar, LinearColorMapper
from bokeh.layouts import column, row
from bokeh.models.widgets import Button, Slider
from bokeh.palettes import Viridis256
import asyncio

from tonnetz.automaton.grid import ToroidalGrid
from tonnetz.core.chord import Chord
from tonnetz.progression.rule_based import RuleBasedProgression

class InteractiveTonnetzApp:
    def __init__(self):
        self.grid = ToroidalGrid(width=12, height=12)
        self.setup_grid()
        self.setup_plots()
        self.setup_controls()
        
    def setup_grid(self):
        """Initialize grid with rule-based progression."""
        progression_algo = RuleBasedProgression(key=0, mode="major")
        start_chord = Chord(0, "major")
        self.grid.populate_with_progression(progression_algo, start_chord)
        
    def setup_plots(self):
        """Create the main visualization plots."""
        # Create figure for chord grid
        self.chord_plot = figure(
            width=600, height=600,
            title="Tonnetz Chord Evolution",
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        # Initialize data source
        self.update_plot_data()
        
    def setup_controls(self):
        """Create control widgets."""
        self.play_button = Button(label="▶ Play", button_type="success")
        self.step_button = Button(label="Step", button_type="primary")
        self.reset_button = Button(label="Reset", button_type="warning")
        
        self.speed_slider = Slider(start=100, end=2000, value=500, step=100, 
                                 title="Animation Speed (ms)")
        
        # Bind callbacks
        self.play_button.on_click(self.toggle_animation)
        self.step_button.on_click(self.step_simulation)
        self.reset_button.on_click(self.reset_simulation)
        
    def update_plot_data(self):
        """Update the plot with current grid state."""
        # Convert grid to visualization data
        chord_data = []
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                cell = self.grid.get_cell(x, y)
                chord_data.append({
                    'x': x,
                    'y': y,
                    'chord_root': cell.chord.root,
                    'chord_quality': cell.chord.quality,
                    'chord_name': str(cell.chord)
                })
        
        # Create data source
        if not hasattr(self, 'source'):
            self.source = ColumnDataSource(data=chord_data)
            
            # Create color mapper
            color_mapper = LinearColorMapper(palette=Viridis256, 
                                           low=0, high=11)
            
            # Add rectangles for chords
            self.chord_plot.rect(x='x', y='y', width=0.9, height=0.9,
                               source=self.source,
                               fill_color={'field': 'chord_root', 
                                         'transform': color_mapper},
                               line_color="white")
        else:
            self.source.data = chord_data
            
    def step_simulation(self):
        """Advance simulation by one step."""
        self.grid.update()
        self.update_plot_data()
        
    def reset_simulation(self):
        """Reset the simulation."""
        self.setup_grid()
        self.update_plot_data()
        
    def toggle_animation(self):
        """Start/stop animation."""
        if self.play_button.label == "▶ Play":
            self.play_button.label = "⏸ Pause"
            curdoc().add_periodic_callback(self.step_simulation, 
                                         self.speed_slider.value)
        else:
            self.play_button.label = "▶ Play"
            curdoc().remove_periodic_callback(self.step_simulation)
            
    def create_layout(self):
        """Create the main layout."""
        controls = row(self.play_button, self.step_button, 
                      self.reset_button, self.speed_slider)
        return column(controls, self.chord_plot)

# Create and serve the app
app = InteractiveTonnetzApp()
curdoc().add_root(app.create_layout())
curdoc().title = "Tonnetz Cellular Automaton"