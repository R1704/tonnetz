# app.py - Main web application with real Tonnetz implementation
import asyncio
import json
import threading

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

# Audio functionality
try:
    import sounddevice as sd

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: sounddevice not available. Audio will be disabled.")

# Import real Tonnetz implementation
try:
    from tonnetz.automaton.grid import ToroidalGrid
    from tonnetz.core.chord import Chord
    from tonnetz.progression.rule_based import RuleBasedProgression
    from tonnetz.progression.markov import MarkovProgression
    from tonnetz.progression.search_based import SearchBasedProgression, SearchObjective

    TONNETZ_AVAILABLE = True
    print("✅ Using real Tonnetz implementation")
except ImportError as e:
    print(f"⚠️  Tonnetz modules not found: {e}")
    print("📝 Creating basic implementation...")
    TONNETZ_AVAILABLE = False

app = FastAPI(title="Tonnetz Cellular Automaton")


# Real audio implementation (same as before)
class ChordPlayer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.is_playing = False
        self.base_freq = 261.63  # C4

    def note_to_frequency(self, note_number):
        """Convert MIDI note number to frequency."""
        return self.base_freq * (2 ** ((note_number - 60) / 12))

    def chord_to_frequencies(self, chord):
        """Convert chord to frequencies."""
        frequencies = []

        # Define chord intervals (in semitones from root)
        chord_intervals = {
            "major": [0, 4, 7],
            "minor": [0, 3, 7],
            "diminished": [0, 3, 6],
            "augmented": [0, 4, 8],
            "major7": [0, 4, 7, 11],
            "minor7": [0, 3, 7, 10],
            "dominant7": [0, 4, 7, 10],
            "diminished7": [0, 3, 6, 9],
            # Legacy support
            "dom7": [0, 4, 7, 10],
            "min7": [0, 3, 7, 10],
        }

        intervals = chord_intervals.get(chord.quality, [0, 4, 7])

        # Base note (C4 = 60)
        base_note = 60 + (chord.root if hasattr(chord, "root") else 0)

        for interval in intervals:
            note = base_note + interval
            freq = self.note_to_frequency(note)
            frequencies.append(freq)

        return frequencies

    def generate_chord_audio(self, chord, duration=0.5):
        """Generate audio for a chord."""
        if not AUDIO_AVAILABLE:
            return np.array([])

        frequencies = self.chord_to_frequencies(chord)

        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros(len(t))

        for freq in frequencies:
            # Generate sine wave with envelope
            wave = np.sin(2 * np.pi * freq * t)
            # Add some harmonics for richer sound
            wave += 0.3 * np.sin(2 * np.pi * freq * 2 * t)  # Octave
            wave += 0.2 * np.sin(2 * np.pi * freq * 3 * t)  # Fifth

            # Apply envelope (attack-decay)
            envelope = np.exp(-t * 3) * (1 - np.exp(-t * 20))
            audio += wave * envelope

        # Normalize and add some reverb-like effect
        audio = audio / len(frequencies)
        if len(audio) > 0:
            audio = np.tanh(audio * 0.7)  # Soft clipping

        return audio

    def play_chord_async(self, chord, duration=0.5):
        """Play a chord asynchronously."""
        if not AUDIO_AVAILABLE:
            print(f"Audio: Would play {chord}")
            return

        def play_thread():
            try:
                audio = self.generate_chord_audio(chord, duration)
                if len(audio) > 0:
                    sd.play(audio, self.sample_rate)
                    sd.wait()  # Wait until sound finishes
            except Exception as e:
                print(f"Audio error: {e}")

        thread = threading.Thread(target=play_thread)
        thread.daemon = True
        thread.start()


# Real Tonnetz implementation or fallback
if TONNETZ_AVAILABLE:
    # Use real implementation
    class TonnetzChordEngine:
        def __init__(self, width=12, height=12, config=None):
            self.width = width
            self.height = height
            self.config = config or {}
            
            # Configuration options
            self.neighborhood = self.config.get("neighborhood", "moore")
            self.update_mode = self.config.get("update_mode", "synchronous")
            self.progression_type = self.config.get("progression_type", "rule_based")
            self.key = self.config.get("key", 0)
            self.mode = self.config.get("mode", "major")
            self.pattern = self.config.get("pattern", "I-vi-IV-V")
            self.randomness = self.config.get("randomness", 0.2)

            # Initialize with real Tonnetz components
            try:
                self.grid = ToroidalGrid(
                    width=width,
                    height=height,
                    neighborhood=self.neighborhood,
                    update_mode=self.update_mode,
                )
                print(f"✅ Real Tonnetz grid created: {self.neighborhood} neighborhood, {self.update_mode} update")

                # Setup initial progression
                self.setup_initial_state()

            except Exception as e:
                print(f"❌ Failed to create real Tonnetz grid: {e}")
                self.grid = None
                self._manual_setup()

        def setup_initial_state(self):
            """Initialize the grid with a musical progression."""
            try:
                # Create progression algorithm based on type
                if self.progression_type == "rule_based":
                    progression_algo = RuleBasedProgression(
                        key=self.key, 
                        mode=self.mode, 
                        pattern=self.pattern, 
                        randomness=self.randomness
                    )
                elif self.progression_type == "markov":
                    try:
                        progression_algo = MarkovProgression(order=1)
                        # Train with some basic progressions
                        training_data = [
                            [Chord(0, "major"), Chord(9, "minor"), Chord(5, "major"), Chord(7, "major")],
                            [Chord(2, "minor"), Chord(7, "major"), Chord(0, "major"), Chord(5, "major")],
                        ]
                        progression_algo.train(training_data)
                    except Exception as e:
                        # Fallback to rule-based
                        progression_algo = RuleBasedProgression(
                            key=self.key, mode=self.mode, pattern=self.pattern
                        )
                        print(f"⚠️ Markov not available ({e}), using rule-based")
                elif self.progression_type == "search_based":
                    try:
                        progression_algo = SearchBasedProgression(
                            search_method="beam",
                            objectives=[SearchObjective.SMOOTH_VOICE_LEADING, SearchObjective.HARMONIC_TENSION],
                            objective_weights=[0.7, 0.3]
                        )
                    except Exception as e:
                        # Fallback to rule-based
                        progression_algo = RuleBasedProgression(
                            key=self.key, mode=self.mode, pattern=self.pattern
                        )
                        print(f"⚠️ Search-based not available ({e}), using rule-based")
                else:
                    progression_algo = RuleBasedProgression(
                        key=self.key, mode=self.mode, pattern=self.pattern
                    )

                start_chord = Chord(root=self.key, quality="major" if self.mode == "major" else "minor")

                # Use the proper populate_with_progression method
                self.grid.populate_with_progression(progression_algo, start_chord)
                print(f"✅ Grid populated with {self.progression_type} progression successfully")

            except Exception as e:
                print(f"❌ Setup error: {e}")
                print(f"   Error details: {type(e).__name__}: {str(e)}")
                # Fallback to manual setup
                self._manual_setup()

        def reconfigure(self, new_config):
            """Reconfigure the engine with new settings."""
            self.config.update(new_config)
            self.neighborhood = self.config.get("neighborhood", "moore")
            self.update_mode = self.config.get("update_mode", "synchronous")
            self.progression_type = self.config.get("progression_type", "rule_based")
            self.key = self.config.get("key", 0)
            self.mode = self.config.get("mode", "major")
            self.pattern = self.config.get("pattern", "I-vi-IV-V")
            self.randomness = self.config.get("randomness", 0.2)
            
            # Recreate grid with new settings
            try:
                self.grid = ToroidalGrid(
                    width=self.width,
                    height=self.height,
                    neighborhood=self.neighborhood,
                    update_mode=self.update_mode,
                )
                self.setup_initial_state()
                print(f"✅ Reconfigured with {self.progression_type}, {self.neighborhood}, key={self.key}")
            except Exception as e:
                print(f"❌ Reconfiguration failed: {e}")
                self._manual_setup()

        def _manual_setup(self):
            """Manual setup if automatic fails."""
            print("🔧 Setting up manual chord grid...")
            # Create varied chord progressions based on current key and mode
            if self.mode == "major":
                chord_patterns = [
                    [Chord((self.key + 0) % 12, "major"), Chord((self.key + 9) % 12, "minor"), 
                     Chord((self.key + 5) % 12, "major"), Chord((self.key + 7) % 12, "major")],  # I-vi-IV-V
                    [Chord((self.key + 2) % 12, "minor"), Chord((self.key + 7) % 12, "major"), 
                     Chord((self.key + 0) % 12, "major"), Chord((self.key + 5) % 12, "major")],  # ii-V-I-IV
                    [Chord((self.key + 4) % 12, "minor"), Chord((self.key + 7) % 12, "major"), 
                     Chord((self.key + 0) % 12, "major"), Chord((self.key + 9) % 12, "minor")],  # iii-V-I-vi
                ]
            else:  # minor
                chord_patterns = [
                    [Chord((self.key + 0) % 12, "minor"), Chord((self.key + 10) % 12, "major"), 
                     Chord((self.key + 5) % 12, "minor"), Chord((self.key + 7) % 12, "major")],  # i-VII-iv-V
                    [Chord((self.key + 3) % 12, "major"), Chord((self.key + 7) % 12, "major"), 
                     Chord((self.key + 0) % 12, "minor"), Chord((self.key + 5) % 12, "minor")],  # III-V-i-iv
                ]

            # Store chords manually since real grid might not work
            self.manual_chords = {}

            for y in range(self.height):
                for x in range(self.width):
                    pattern_idx = (x // 4) % len(chord_patterns)
                    chord_idx = (x + y) % len(chord_patterns[pattern_idx])
                    chord = chord_patterns[pattern_idx][chord_idx]

                    # Add some variation based on randomness setting
                    if (x + y) % int(10 / max(self.randomness, 0.1)) == 0:
                        # Occasionally change to diminished
                        chord = Chord(chord.root, "diminished")
                    elif (x + y) % int(8 / max(self.randomness, 0.1)) == 0:
                        # Occasionally add 7th
                        chord = Chord(
                            chord.root,
                            "dominant7" if chord.quality == "major" else "minor7",
                        )

                    self.manual_chords[(x, y)] = chord

        # ... rest of the methods stay the same ...

        def get_cell_chord(self, x, y):
            """Get chord at position."""
            # First try real grid
            if self.grid is not None:
                try:
                    cell = self.grid.get_cell(x, y)
                    if hasattr(cell, "chord"):
                        return cell.chord
                except Exception as e:
                    print(f"Grid access error: {e}")

            # Use manual chords if available
            if hasattr(self, "manual_chords") and (x, y) in self.manual_chords:
                return self.manual_chords[(x, y)]

            # Final fallback - create varied chords
            root = (x * 4 + y * 7) % 12  # Major third + perfect fifth
            if (x + y) % 3 == 0:
                quality = "major"
            elif (x + y) % 3 == 1:
                quality = "minor"
            else:
                quality = "diminished" if (x + y) % 7 == 0 else "major"

            return Chord(root, quality)

        def update(self):
            """Update the grid state."""
            # Try real grid update first
            if self.grid is not None:
                try:
                    self.grid.update()
                    print(f"✅ Real grid updated to generation {self.grid.generation}")
                    return
                except Exception as e:
                    print(f"Real grid update failed: {e}")

            # Manual update using neo-Riemannian transformations
            if hasattr(self, "manual_chords"):
                self._manual_update()

        def _manual_update(self):
            """Manual update using cellular automaton rules."""
            new_chords = {}
            generation = getattr(self, "manual_generation", 0) + 1
            self.manual_generation = generation

            for y in range(self.height):
                for x in range(self.width):
                    current_chord = self.manual_chords[(x, y)]

                    # Count neighbor chord types
                    neighbor_qualities = []
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = (x + dx) % self.width, (y + dy) % self.height
                            neighbor_qualities.append(
                                self.manual_chords[(nx, ny)].quality
                            )

                    # Apply transformation based on neighborhood
                    major_count = neighbor_qualities.count("major")
                    minor_count = neighbor_qualities.count("minor")

                    new_root = current_chord.root
                    new_quality = current_chord.quality

                    # Simple evolution rules
                    if major_count > minor_count and current_chord.quality == "minor":
                        # Parallel transformation: minor -> major
                        new_quality = "major"
                    elif minor_count > major_count and current_chord.quality == "major":
                        # Parallel transformation: major -> minor
                        new_quality = "minor"
                    elif generation % 8 == 0:
                        # Occasional root movement (Leading-tone exchange)
                        new_root = (current_chord.root + 1) % 12
                    elif generation % 12 == 0:
                        # Relative transformation (major/minor to relative)
                        if current_chord.quality == "major":
                            new_root = (
                                current_chord.root + 9
                            ) % 12  # to relative minor
                            new_quality = "minor"
                        elif current_chord.quality == "minor":
                            new_root = (
                                current_chord.root + 3
                            ) % 12  # to relative major
                            new_quality = "major"

                    new_chords[(x, y)] = Chord(new_root, new_quality)

            self.manual_chords = new_chords

        def get_statistics(self):
            """Get grid statistics."""
            if self.grid is not None:
                try:
                    return self.grid.get_statistics()
                except Exception as e:
                    print(f"Stats error: {e}")

            # Manual statistics
            if hasattr(self, "manual_chords"):
                qualities = [chord.quality for chord in self.manual_chords.values()]
                major_count = qualities.count("major")
                minor_count = qualities.count("minor")
                total = len(qualities)
                harmony_level = (major_count / total) * 0.7 + (
                    minor_count / total
                ) * 0.3
                generation = getattr(self, "manual_generation", 0)
            else:
                harmony_level = 0.75
                generation = 0

            return {
                "generation": generation,
                "active_cells": self.width * self.height,
                "harmony_level": harmony_level,
            }

        def get_center_chord(self):
            """Get center chord."""
            center_x, center_y = self.width // 2, self.height // 2
            return self.get_cell_chord(center_x, center_y)
else:
    # Fallback implementation (improved mock)
    class TonnetzChordEngine:
        def __init__(self, width=12, height=12):
            self.width = width
            self.height = height
            self.generation = 0
            self.cells = {}
            self._initialize_cells()

        def _initialize_cells(self):
            """Initialize all cells with Tonnetz-inspired layout."""
            # Create a more sophisticated chord layout based on Tonnetz theory
            for y in range(self.height):
                for x in range(self.width):
                    # Use Tonnetz-like relationships
                    # Major thirds horizontally, perfect fifths vertically
                    root = (x * 4 + y * 7) % 12  # Major third + perfect fifth

                    # Determine quality based on position
                    if (x + y) % 3 == 0:
                        quality = "major"
                    elif (x + y) % 3 == 1:
                        quality = "minor"
                    else:
                        quality = "diminished" if (x + y) % 7 == 0 else "major"

                    # Create simple chord object
                    chord = type(
                        "Chord",
                        (),
                        {
                            "root": root,
                            "quality": quality,
                            "__str__": lambda self: self._chord_name(),
                        },
                    )()

                    def _chord_name(self):
                        note_names = [
                            "C",
                            "C#",
                            "D",
                            "D#",
                            "E",
                            "F",
                            "F#",
                            "G",
                            "G#",
                            "A",
                            "A#",
                            "B",
                        ]
                        quality_symbols = {
                            "major": "",
                            "minor": "m",
                            "diminished": "°",
                            "augmented": "+",
                            "dom7": "7",
                            "min7": "m7",
                        }
                        return f"{note_names[self.root]}{quality_symbols.get(self.quality, '')}"

                    chord._chord_name = _chord_name.__get__(chord, type(chord))
                    self.cells[(x, y)] = chord

        def get_cell_chord(self, x, y):
            """Get chord at position."""
            return self.cells.get((x, y))

        def update(self):
            """Update using cellular automaton rules."""
            self.generation += 1
            new_cells = {}

            for y in range(self.height):
                for x in range(self.width):
                    # Apply neo-Riemannian transformations based on neighbors
                    current_chord = self.cells[(x, y)]

                    # Count neighbor chord types
                    neighbor_qualities = []
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = (x + dx) % self.width, (y + dy) % self.height
                            neighbor_qualities.append(self.cells[(nx, ny)].quality)

                    # Apply transformation based on neighborhood
                    major_count = neighbor_qualities.count("major")
                    minor_count = neighbor_qualities.count("minor")

                    new_chord = type(
                        "Chord",
                        (),
                        {
                            "root": current_chord.root,
                            "quality": current_chord.quality,
                            "__str__": current_chord.__str__,
                        },
                    )()

                    # Simple evolution rules
                    if major_count > minor_count and current_chord.quality == "minor":
                        # Parallel transformation: minor -> major
                        new_chord.quality = "major"
                    elif minor_count > major_count and current_chord.quality == "major":
                        # Parallel transformation: major -> minor
                        new_chord.quality = "minor"
                    elif self.generation % 8 == 0:
                        # Occasional root movement (Leading-tone exchange)
                        new_chord.root = (current_chord.root + 1) % 12

                    new_cells[(x, y)] = new_chord

            self.cells = new_cells

        def get_statistics(self):
            """Get grid statistics."""
            qualities = [chord.quality for chord in self.cells.values()]
            major_count = qualities.count("major")
            minor_count = qualities.count("minor")

            # Calculate harmony level based on major/minor ratio
            total = len(qualities)
            harmony_level = (major_count / total) * 0.7 + (minor_count / total) * 0.3

            return {
                "generation": self.generation,
                "active_cells": total,
                "harmony_level": harmony_level,
            }

        def get_center_chord(self):
            """Get center chord."""
            center_x, center_y = self.width // 2, self.height // 2
            return self.get_cell_chord(center_x, center_y)


# Global state
engine = TonnetzChordEngine(width=12, height=12, config={
    "neighborhood": "moore",
    "update_mode": "synchronous", 
    "progression_type": "rule_based",
    "key": 0,
    "mode": "major",
    "pattern": "I-vi-IV-V",
    "randomness": 0.2
})
chord_player = ChordPlayer()


# ... (same HTML interface as before)
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tonnetz Cellular Automaton - Advanced Controls</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 0; background: #f8f9fa; 
            }
            .container { max-width: 1600px; margin: 0 auto; padding: 20px; }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
                text-align: center;
            }
            
            .main-content {
                display: grid;
                grid-template-columns: 320px 1fr 300px;
                gap: 20px;
            }
            
            .control-panel {
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                height: fit-content;
                position: sticky;
                top: 20px;
            }
            
            .control-section {
                margin-bottom: 25px;
                border-bottom: 1px solid #e9ecef;
                padding-bottom: 15px;
            }
            
            .control-section:last-child {
                border-bottom: none;
                margin-bottom: 0;
            }
            
            .control-section h3 {
                margin: 0 0 15px 0;
                color: #495057;
                font-size: 16px;
                font-weight: 600;
            }
            
            .form-group {
                margin-bottom: 15px;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 5px;
                font-weight: 500;
                color: #495057;
                font-size: 14px;
            }
            
            select, input[type="range"], input[type="number"] {
                width: 100%;
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 6px;
                font-size: 14px;
            }
            
            .grid-area {
                background: white;
                border-radius: 12px;
                padding: 25px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
            }
            
            .chord-grid {
                display: inline-block;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 15px;
                background: #f8f9fa;
            }
            
            .chord-cell {
                display: inline-block;
                width: 50px;
                height: 50px;
                text-align: center;
                line-height: 50px;
                border: 1px solid #dee2e6;
                margin: 1px;
                font-size: 11px;
                font-weight: bold;
                border-radius: 6px;
                transition: all 0.2s ease;
                cursor: pointer;
                background: white;
            }
            
            .chord-cell:hover {
                transform: scale(1.1);
                z-index: 10;
                position: relative;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }
            
            .chord-cell.center {
                border: 3px solid #dc3545;
                box-shadow: 0 0 0 2px rgba(220,53,69,0.2);
            }
            
            .info-panel {
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                height: fit-content;
                position: sticky;
                top: 20px;
            }
            
            .transport-controls {
                display: flex;
                gap: 8px;
                margin-bottom: 20px;
            }
            
            button {
                border: none;
                padding: 10px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.2s ease;
                min-width: 80px;
            }
            
            .btn-primary { background: #007bff; color: white; }
            .btn-primary:hover { background: #0056b3; }
            .btn-success { background: #28a745; color: white; }
            .btn-success:hover { background: #1e7e34; }
            .btn-danger { background: #dc3545; color: white; }
            .btn-danger:hover { background: #c82333; }
            .btn-warning { background: #ffc107; color: black; }
            .btn-warning:hover { background: #e0a800; }
            .btn-info { background: #17a2b8; color: white; }
            .btn-info:hover { background: #138496; }
            .btn-secondary { background: #6c757d; color: white; }
            .btn-secondary:hover { background: #545b62; }
            
            .range-group {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .range-group input {
                flex: 1;
                margin: 0;
            }
            
            .range-value {
                min-width: 60px;
                text-align: center;
                font-weight: 500;
                color: #495057;
            }
            
            .status-indicator {
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
                margin: 5px 0;
            }
            
            .status-connected { background: #d4edda; color: #155724; }
            .status-disconnected { background: #f8d7da; color: #721c24; }
            .status-running { background: #d1ecf1; color: #0c5460; }
            
            .stats-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin: 15px 0;
            }
            
            .stat-item {
                text-align: center;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 6px;
            }
            
            .stat-value {
                font-size: 20px;
                font-weight: bold;
                color: #495057;
            }
            
            .stat-label {
                font-size: 12px;
                color: #6c757d;
                margin-top: 2px;
            }
            
            .implementation-badge {
                background: #e3f2fd;
                color: #1976d2;
                padding: 8px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 500;
                display: inline-block;
                margin-bottom: 15px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎵 Tonnetz: Cellular Automaton Chord Engine</h1>
                <div class="implementation-badge">
                    Implementation: """ + ("Real Tonnetz" if TONNETZ_AVAILABLE else "Enhanced Mock") + """ | 
                    Audio: """ + ("Enabled" if AUDIO_AVAILABLE else "Disabled") + """
                </div>
            </div>
            
            <div class="main-content">
                <!-- Left Control Panel -->
                <div class="control-panel">
                    <div class="control-section">
                        <h3>🎮 Transport Controls</h3>
                        <div class="transport-controls">
                            <button onclick="startSimulation()" id="startBtn" class="btn-primary">▶</button>
                            <button onclick="stopSimulation()" class="btn-danger">⏸</button>
                            <button onclick="stepSimulation()" class="btn-info">⏭</button>
                            <button onclick="resetSimulation()" class="btn-warning">🔄</button>
                        </div>
                        <div class="form-group">
                            <label>Speed Control</label>
                            <div class="range-group">
                                <input type="range" id="speed" min="100" max="3000" value="500" step="100">
                                <span class="range-value" id="speedValue">500ms</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="control-section">
                        <h3>🎼 Progression Algorithm</h3>
                        <div class="form-group">
                            <label>Algorithm Type</label>
                            <select id="progressionType" onchange="updateProgression()">
                                <option value="rule_based">Rule-Based</option>
                                <option value="markov">Markov Chain</option>
                                <option value="search_based">Search-Based</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Pattern (Rule-Based)</label>
                            <select id="pattern" onchange="updateProgression()">
                                <option value="I-vi-IV-V">I-vi-IV-V (50s Progression)</option>
                                <option value="ii-V-I">ii-V-I (Jazz)</option>
                                <option value="vi-IV-V-I">vi-IV-V-I (Pop)</option>
                                <option value="circle_of_fifths">Circle of Fifths</option>
                                <option value="blues">12-Bar Blues</option>
                                <option value="modal_iv">Modal IV</option>
                                <option value="chromatic_mediant">Chromatic Mediant</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Randomness</label>
                            <div class="range-group">
                                <input type="range" id="randomness" min="0" max="1" value="0.2" step="0.1" onchange="updateProgression()">
                                <span class="range-value" id="randomnessValue">0.2</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="control-section">
                        <h3>🎹 Musical Settings</h3>
                        <div class="form-group">
                            <label>Key Center</label>
                            <select id="key" onchange="updateProgression()">
                                <option value="0">C</option>
                                <option value="1">C#/Db</option>
                                <option value="2">D</option>
                                <option value="3">D#/Eb</option>
                                <option value="4">E</option>
                                <option value="5">F</option>
                                <option value="6">F#/Gb</option>
                                <option value="7">G</option>
                                <option value="8">G#/Ab</option>
                                <option value="9">A</option>
                                <option value="10">A#/Bb</option>
                                <option value="11">B</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Mode</label>
                            <select id="mode" onchange="updateProgression()">
                                <option value="major">Major</option>
                                <option value="minor">Minor</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="control-section">
                        <h3>🔬 Cellular Automaton</h3>
                        <div class="form-group">
                            <label>Neighborhood</label>
                            <select id="neighborhood" onchange="updateGrid()">
                                <option value="moore">Moore (8-connected)</option>
                                <option value="von_neumann">Von Neumann (4-connected)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Update Mode</label>
                            <select id="updateMode" onchange="updateGrid()">
                                <option value="synchronous">Synchronous</option>
                                <option value="asynchronous">Asynchronous</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Grid Size</label>
                            <div class="range-group">
                                <input type="range" id="gridSize" min="8" max="20" value="12" step="1" onchange="updateGridSize()">
                                <span class="range-value" id="gridSizeValue">12x12</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="control-section">
                        <h3>🔊 Audio Controls</h3>
                        <button onclick="toggleAudio()" class="btn-secondary" id="audioBtn">🔊 Audio OFF</button>
                        <button onclick="playTestChord()" class="btn-info" style="margin-top: 10px;">🎹 Test Sound</button>
                    </div>
                </div>
                
                <!-- Center Grid Area -->
                <div class="grid-area">
                    <h3>Tonnetz Chord Evolution Grid</h3>
                    <div class="chord-grid" id="chord-grid"></div>
                    <div style="margin-top: 15px; font-size: 14px; color: #6c757d;">
                        Click any chord cell to hear it | Red border = center chord
                    </div>
                </div>
                
                <!-- Right Info Panel -->
                <div class="info-panel">
                    <h3>System Status</h3>
                    <div class="status-indicator" id="connectionStatus">Connecting...</div>
                    <div class="status-indicator" id="audioStatus">Audio: Ready</div>
                    
                    <h4 style="margin-top: 20px;">Live Statistics</h4>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value" id="generation">0</div>
                            <div class="stat-label">Generation</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="activeCells">0</div>
                            <div class="stat-label">Active Cells</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="harmonyLevel">0.00</div>
                            <div class="stat-label">Harmony Level</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="centerChord">-</div>
                            <div class="stat-label">Center Chord</div>
                        </div>
                    </div>
                    
                    <h4 style="margin-top: 20px;">Current Configuration</h4>
                    <div style="font-size: 12px; background: #f8f9fa; padding: 10px; border-radius: 6px;">
                        <div><strong>Algorithm:</strong> <span id="currentAlgorithm">rule_based</span></div>
                        <div><strong>Pattern:</strong> <span id="currentPattern">I-vi-IV-V</span></div>
                        <div><strong>Key:</strong> <span id="currentKey">C major</span></div>
                        <div><strong>Neighborhood:</strong> <span id="currentNeighborhood">moore</span></div>
                        <div><strong>Update:</strong> <span id="currentUpdateMode">synchronous</span></div>
                    </div>
                    
                    <h4 style="margin-top: 20px;">Neo-Riemannian Operations</h4>
                    <div style="font-size: 11px; color: #6c757d;">
                        <div><strong>P (Parallel):</strong> Major ↔ Minor</div>
                        <div><strong>L (Leading-tone):</strong> Root movement</div>
                        <div><strong>R (Relative):</strong> Quality changes</div>
                        <div style="margin-top: 8px; font-style: italic;">Grid evolves based on neighbor harmony relationships</div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let ws = null;
            let isRunning = false;
            let audioEnabled = false;
            let reconnectAttempts = 0;
            const maxReconnectAttempts = 5;
            
            // WebSocket connection
            function connectWebSocket() {
                try {
                    ws = new WebSocket('ws://localhost:8000/ws');
                    
                    ws.onopen = function() {
                        console.log('WebSocket connected');
                        updateConnectionStatus('Connected', true);
                        reconnectAttempts = 0;
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        updateDisplay(data);
                    };
                    
                    ws.onclose = function() {
                        console.log('WebSocket disconnected');
                        updateConnectionStatus('Disconnected', false);
                        
                        if (reconnectAttempts < maxReconnectAttempts) {
                            setTimeout(() => {
                                reconnectAttempts++;
                                connectWebSocket();
                            }, 2000);
                        }
                    };
                    
                    ws.onerror = function(error) {
                        console.error('WebSocket error:', error);
                        updateConnectionStatus('Connection Error', false);
                    };
                } catch (error) {
                    console.error('Failed to create WebSocket:', error);
                    updateConnectionStatus('Failed to Connect', false);
                }
            }
            
            function updateConnectionStatus(message, connected) {
                const statusEl = document.getElementById('connectionStatus');
                statusEl.textContent = message;
                statusEl.className = connected ? 'status-indicator status-connected' : 'status-indicator status-disconnected';
            }
            
            function updateDisplay(data) {
                const gridContainer = document.getElementById('chord-grid');
                gridContainer.innerHTML = '';
                
                const centerX = Math.floor(data.width / 2);
                const centerY = Math.floor(data.height / 2);
                
                for (let y = 0; y < data.height; y++) {
                    for (let x = 0; x < data.width; x++) {
                        const cell = data.cells[y * data.width + x];
                        const cellDiv = document.createElement('div');
                        cellDiv.className = 'chord-cell';
                        if (x === centerX && y === centerY) {
                            cellDiv.className += ' center';
                        }
                        cellDiv.textContent = cell.chord_name;
                        cellDiv.style.backgroundColor = getChordColor(cell.chord_quality);
                        cellDiv.title = `${cell.chord_name} at (${x},${y}) - Click to hear!`;
                        
                        cellDiv.onclick = function() {
                            playChord(cell);
                        };
                        
                        gridContainer.appendChild(cellDiv);
                    }
                    gridContainer.appendChild(document.createElement('br'));
                }
                
                // Update statistics
                document.getElementById('generation').textContent = data.generation;
                document.getElementById('activeCells').textContent = data.active_cells;
                document.getElementById('harmonyLevel').textContent = data.harmony_level.toFixed(2);
                
                const centerCell = data.cells[centerY * data.width + centerX];
                document.getElementById('centerChord').textContent = centerCell.chord_name;
            }
            
            function getChordColor(quality) {
                const colors = {
                    'major': '#28a745',
                    'minor': '#007bff', 
                    'diminished': '#dc3545',
                    'augmented': '#fd7e14',
                    'major7': '#6f42c1',
                    'minor7': '#6c757d',
                    'dominant7': '#e83e8c',
                    'diminished7': '#495057'
                };
                return colors[quality] || '#adb5bd';
            }
            
            function sendMessage(message) {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify(message));
                } else {
                    updateConnectionStatus('Not Connected', false);
                }
            }
            
            // Transport controls
            function startSimulation() {
                sendMessage({action: 'start'});
                isRunning = true;
                document.getElementById('startBtn').textContent = '⏸';
                document.getElementById('startBtn').className = 'btn-success';
            }
            
            function stopSimulation() {
                sendMessage({action: 'stop'});
                isRunning = false;
                document.getElementById('startBtn').textContent = '▶';
                document.getElementById('startBtn').className = 'btn-primary';
            }
            
            function stepSimulation() {
                sendMessage({action: 'step'});
            }
            
            function resetSimulation() {
                sendMessage({action: 'reset'});
                isRunning = false;
                document.getElementById('startBtn').textContent = '▶';
                document.getElementById('startBtn').className = 'btn-primary';
            }
            
            // Configuration updates
            function updateProgression() {
                const config = {
                    progression_type: document.getElementById('progressionType').value,
                    pattern: document.getElementById('pattern').value,
                    key: parseInt(document.getElementById('key').value),
                    mode: document.getElementById('mode').value,
                    randomness: parseFloat(document.getElementById('randomness').value)
                };
                
                sendMessage({action: 'reconfigure', config: config});
                updateConfigDisplay();
            }
            
            function updateGrid() {
                const config = {
                    neighborhood: document.getElementById('neighborhood').value,
                    update_mode: document.getElementById('updateMode').value
                };
                
                sendMessage({action: 'reconfigure', config: config});
                updateConfigDisplay();
            }
            
            function updateGridSize() {
                const size = parseInt(document.getElementById('gridSize').value);
                document.getElementById('gridSizeValue').textContent = `${size}x${size}`;
                sendMessage({action: 'resize_grid', width: size, height: size});
            }
            
            function updateConfigDisplay() {
                const keyNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
                document.getElementById('currentAlgorithm').textContent = document.getElementById('progressionType').value;
                document.getElementById('currentPattern').textContent = document.getElementById('pattern').value;
                document.getElementById('currentKey').textContent = keyNames[parseInt(document.getElementById('key').value)] + ' ' + document.getElementById('mode').value;
                document.getElementById('currentNeighborhood').textContent = document.getElementById('neighborhood').value;
                document.getElementById('currentUpdateMode').textContent = document.getElementById('updateMode').value;
                document.getElementById('randomnessValue').textContent = document.getElementById('randomness').value;
            }
            
            // Audio controls
            function toggleAudio() {
                audioEnabled = !audioEnabled;
                const audioBtn = document.getElementById('audioBtn');
                if (audioEnabled) {
                    audioBtn.textContent = '🔊 Audio ON';
                    audioBtn.className = 'btn-success';
                    document.getElementById('audioStatus').textContent = 'Audio: Enabled';
                    document.getElementById('audioStatus').className = 'status-indicator status-connected';
                } else {
                    audioBtn.textContent = '🔊 Audio OFF';
                    audioBtn.className = 'btn-secondary';
                    document.getElementById('audioStatus').textContent = 'Audio: Disabled';
                    document.getElementById('audioStatus').className = 'status-indicator status-disconnected';
                }
                sendMessage({action: 'toggle_audio', enabled: audioEnabled});
            }
            
            function playChord(cell) {
                sendMessage({
                    action: 'play_chord',
                    chord: {
                        root: cell.chord_root,
                        quality: cell.chord_quality,
                        name: cell.chord_name
                    }
                });
                document.getElementById('audioStatus').textContent = `Playing: ${cell.chord_name}`;
                document.getElementById('audioStatus').className = 'status-indicator status-running';
            }
            
            function playTestChord() {
                sendMessage({
                    action: 'play_chord',
                    chord: {
                        root: 0,
                        quality: 'major',
                        name: 'C'
                    }
                });
                document.getElementById('audioStatus').textContent = 'Playing test chord: C major';
                document.getElementById('audioStatus').className = 'status-indicator status-running';
            }
            
            // Speed control
            document.getElementById('speed').addEventListener('input', function() {
                document.getElementById('speedValue').textContent = this.value + 'ms';
                sendMessage({action: 'set_speed', speed: parseInt(this.value)});
            });
            
            // Initialize
            connectWebSocket();
            updateConfigDisplay();
        </script>
    </body>
    </html>
    """


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    global engine  # Move this to the top

    await websocket.accept()

    # Send initial state
    await send_grid_state(websocket)

    simulation_running = False
    audio_enabled = False
    speed = 500  # ms

    async def simulation_loop():
        while simulation_running:
            engine.update()
            await send_grid_state(websocket)

            # Play center chord if audio is enabled
            if audio_enabled:
                center_chord = engine.get_center_chord()
                chord_player.play_chord_async(center_chord, duration=speed / 1000 * 0.8)

            await asyncio.sleep(speed / 1000.0)

    simulation_task = None

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["action"] == "start":
                if not simulation_running:
                    simulation_running = True
                    simulation_task = asyncio.create_task(simulation_loop())

            elif message["action"] == "stop":
                simulation_running = False
                if simulation_task:
                    simulation_task.cancel()

            elif message["action"] == "step":
                engine.update()
                await send_grid_state(websocket)

            elif message["action"] == "reset":
                simulation_running = False
                if simulation_task:
                    simulation_task.cancel()
                # Recreate engine with current configuration
                engine = TonnetzChordEngine(width=engine.width, height=engine.height, config=engine.config)
                await send_grid_state(websocket)

            elif message["action"] == "reconfigure":
                # Reconfigure the engine with new settings
                config = message.get("config", {})
                engine.reconfigure(config)
                await send_grid_state(websocket)
                
            elif message["action"] == "resize_grid":
                # Resize the grid
                simulation_running = False
                if simulation_task:
                    simulation_task.cancel()
                width = message.get("width", 12)
                height = message.get("height", 12)
                engine = TonnetzChordEngine(width=width, height=height, config=engine.config)
                await send_grid_state(websocket)

            elif message["action"] == "toggle_audio":
                audio_enabled = message["enabled"]

            elif message["action"] == "set_speed":
                speed = message["speed"]

            elif message["action"] == "play_chord":
                # Play individual chord when clicked
                chord_data = message["chord"]
                # Create chord object compatible with player
                chord = type(
                    "Chord",
                    (),
                    {"root": chord_data["root"], "quality": chord_data["quality"]},
                )()
                chord_player.play_chord_async(chord, duration=1.0)

    except Exception as e:
        print(f"WebSocket error: {e}")


async def send_grid_state(websocket):
    """Send current grid state to client."""
    cells = []
    for y in range(engine.height):
        for x in range(engine.width):
            chord = engine.get_cell_chord(x, y)
            cells.append(
                {
                    "x": x,
                    "y": y,
                    "chord_name": str(chord),
                    "chord_quality": chord.quality,
                    "chord_root": chord.root,
                }
            )

    stats = engine.get_statistics()

    data = {
        "cells": cells,
        "width": engine.width,
        "height": engine.height,
        "generation": stats["generation"],
        "active_cells": stats["active_cells"],
        "harmony_level": stats["harmony_level"],
    }

    await websocket.send_text(json.dumps(data))


if __name__ == "__main__":
    import uvicorn

    print("Starting Tonnetz Cellular Automaton Server...")
    print("🎵 Audio functionality:", "ENABLED" if AUDIO_AVAILABLE else "DISABLED")
    print(
        "📦 Implementation:", "Real Tonnetz" if TONNETZ_AVAILABLE else "Enhanced Mock"
    )
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
