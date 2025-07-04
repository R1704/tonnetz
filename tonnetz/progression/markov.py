"""
Markov chain-based chord progression algorithm.

This module implements a progression algorithm that uses Markov chains
to generate chord sequences based on statistical models of chord transitions.
"""

import random
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

from tonnetz.core.chord import Chord
from tonnetz.progression.base import ProgressionAlgo


class MarkovProgression(ProgressionAlgo):
    """
    Markov chain-based chord progression generator.
    
    This algorithm learns transition probabilities between chords and generates
    new sequences based on those probabilities. It can be trained on existing
    chord progressions or use predefined transition matrices.
    """
    
    def __init__(self, order: int = 1, **kwargs):
        """
        Initialize the Markov progression generator.
        
        Args:
            order: Order of the Markov chain (1 = first-order, 2 = second-order, etc.)
            **kwargs: Additional parameters
        """
        super().__init__()
        self.order = order
        self.transition_matrix: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.chord_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self._trained = False
        
        # Default transition probabilities for common progressions
        # Only setup if no external training is provided
        if kwargs.get('auto_train', True):
            self._setup_default_transitions()
    
    def _setup_default_transitions(self):
        """Setup default transition probabilities based on common progressions."""
        # Common progressions in major keys
        progressions = [
            ['C', 'Am', 'F', 'G'],  # vi-IV-V-I
            ['C', 'F', 'G', 'C'],   # I-IV-V-I
            ['Am', 'F', 'C', 'G'],  # vi-IV-I-V
            ['C', 'G', 'Am', 'F'],  # I-V-vi-IV
            ['F', 'G', 'C', 'Am'],  # IV-V-I-vi
            ['Dm', 'G', 'C', 'Am'], # ii-V-I-vi
            ['C', 'Am', 'Dm', 'G'], # I-vi-ii-V
        ]
        
        # Train on these progressions
        for progression in progressions:
            self.train([progression])
    
    def train(self, progressions: List[List[str]], weights: Optional[List[float]] = None):
        """
        Train the Markov model on chord progressions.
        
        Args:
            progressions: List of chord progressions (each progression is a list of chord names)
            weights: Optional weights for each progression
        """
        if weights is None:
            weights = [1.0] * len(progressions)
        
        for progression, weight in zip(progressions, weights):
            if len(progression) < self.order + 1:
                continue
            
            for i in range(len(progression) - self.order):
                state = tuple(progression[i:i + self.order])
                next_chord = progression[i + self.order]
                
                self.transition_matrix[state][next_chord] += weight
                self.chord_counts[state] += weight
        
        # Normalize probabilities
        for state in self.transition_matrix:
            total = sum(self.transition_matrix[state].values())
            if total > 0:
                for chord in self.transition_matrix[state]:
                    self.transition_matrix[state][chord] /= total
        
        self._trained = True
    
    def generate(self, start_chord: Chord, length: int, **kwargs) -> List[Chord]:
        """
        Generate a chord progression using the Markov model.
        
        Args:
            start_chord: Starting chord
            length: Desired length of progression
            **kwargs: Additional generation parameters
            
        Returns:
            List of chords in the progression
        """
        return self.generate_progression(start_chord, length)
    
    def generate_progression(self, start_chord: Chord, length: int) -> List[Chord]:
        """
        Generate a chord progression using the Markov model.
        
        Args:
            start_chord: Starting chord
            length: Desired length of progression
            
        Returns:
            List of chords in the progression
        """
        if not self._trained:
            self._setup_default_transitions()
        
        progression = [start_chord]
        current_state = [str(start_chord)]
        
        for _ in range(length - 1):
            # Create state tuple for lookup
            state_key = tuple(current_state[-self.order:])
            
            # Get possible next chords
            if state_key in self.transition_matrix:
                candidates = self.transition_matrix[state_key]
            else:
                # Fallback to most common transitions from any state
                candidates = self._get_fallback_candidates(current_state[-1] if current_state else str(start_chord))
            
            if not candidates:
                # If no candidates, use common chord progression fallback
                candidates = {'C': 0.3, 'Am': 0.2, 'F': 0.3, 'G': 0.2}
            
            # Select next chord based on probabilities
            chord_names = list(candidates.keys())
            probabilities = list(candidates.values())
            
            next_chord_name = random.choices(chord_names, weights=probabilities)[0]
            
            try:
                next_chord = Chord.from_name(next_chord_name)
            except Exception:
                # Fallback if chord parsing fails
                next_chord = Chord(0, 'major')  # C major
            
            progression.append(next_chord)
            current_state.append(str(next_chord))
        
        return progression
    
    def _get_fallback_candidates(self, current_chord: str) -> Dict[str, float]:
        """Get fallback candidates when no trained transitions are available."""
        # Simple fallback based on common chord relationships
        fallbacks = {
            'C': {'Am': 0.3, 'F': 0.3, 'G': 0.3, 'Dm': 0.1},
            'Am': {'F': 0.3, 'C': 0.3, 'G': 0.2, 'Dm': 0.2},
            'F': {'G': 0.4, 'C': 0.3, 'Am': 0.2, 'Dm': 0.1},
            'G': {'C': 0.4, 'Am': 0.3, 'F': 0.2, 'Dm': 0.1},
            'Dm': {'G': 0.4, 'C': 0.3, 'Am': 0.2, 'F': 0.1},
        }
        
        return fallbacks.get(current_chord, {'C': 0.4, 'Am': 0.3, 'F': 0.2, 'G': 0.1})
    
    def set_parameters(self, **params):
        """Set algorithm parameters."""
        if 'order' in params:
            self.order = max(1, int(params['order']))
        
        if 'training_progressions' in params:
            self.train(params['training_progressions'])
        
        if 'temperature' in params:
            # Temperature parameter for controlling randomness
            self.temperature = float(params['temperature'])
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the algorithm."""
        return {
            'algorithm': 'markov',
            'order': self.order,
            'trained': self._trained,
            'num_states': len(self.transition_matrix),
            'transition_matrix': dict(self.transition_matrix) if len(self.transition_matrix) < 50 else "too_large"
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the current parameters of the algorithm."""
        return {
            'order': self.order,
            'trained': self._trained,
            'num_states': len(self.transition_matrix)
        }


def _apply_temperature(probabilities: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    """Apply temperature scaling to probabilities."""
    if temperature == 1.0:
        return probabilities
    
    import math
    
    # Apply temperature scaling
    scaled = {}
    for key, prob in probabilities.items():
        if prob > 0:
            scaled[key] = math.exp(math.log(prob) / temperature)
        else:
            scaled[key] = 0
    
    # Renormalize
    total = sum(scaled.values())
    if total > 0:
        for key in scaled:
            scaled[key] /= total
    
    return scaled
