# tonnetz/audio/playback.py
import numpy as np
import sounddevice as sd
import threading
import time
from typing import List

class ChordPlayer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.is_playing = False
        
    def chord_to_frequencies(self, chord):
        """Convert chord to frequencies."""
        # Base frequency for C4
        base_freq = 261.63
        
        # Calculate frequencies for chord tones
        frequencies = []
        for pitch_class in chord.pitch_classes():
            freq = base_freq * (2 ** (pitch_class / 12))
            frequencies.append(freq)
            
        return frequencies
        
    def generate_chord_audio(self, chord, duration=1.0):
        """Generate audio for a chord."""
        frequencies = self.chord_to_frequencies(chord)
        
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros(len(t))
        
        for freq in frequencies:
            # Generate sine wave with envelope
            wave = np.sin(2 * np.pi * freq * t)
            envelope = np.exp(-t * 2)  # Decay envelope
            audio += wave * envelope
            
        return audio / len(frequencies)  # Normalize
        
    def play_chord(self, chord, duration=1.0):
        """Play a single chord."""
        audio = self.generate_chord_audio(chord, duration)
        sd.play(audio, self.sample_rate)
        
    def play_progression(self, progression: List, chord_duration=1.0):
        """Play a chord progression."""
        for chord in progression:
            if not self.is_playing:
                break
            self.play_chord(chord, chord_duration)
            time.sleep(chord_duration)
            
    def start_grid_playback(self, grid, update_interval=1.0):
        """Play chords from grid evolution."""
        self.is_playing = True
        
        def playback_thread():
            while self.is_playing:
                # Get current chord from center of grid
                center_x, center_y = grid.width // 2, grid.height // 2
                cell = grid.get_cell(center_x, center_y)
                
                self.play_chord(cell.chord, update_interval * 0.8)
                time.sleep(update_interval)
                
        thread = threading.Thread(target=playback_thread)
        thread.daemon = True
        thread.start()
        
    def stop_playback(self):
        """Stop audio playback."""
        self.is_playing = False
        sd.stop()