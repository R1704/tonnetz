"""
Rule-based chord progression algorithm.

This module implements a progression algorithm based on functional harmony
rules and common chord progression patterns.
"""

import random
from typing import List, Dict, Any, Optional
from tonnetz.core.chord import Chord
from tonnetz.progression.base import ProgressionAlgo


class RuleBasedProgression(ProgressionAlgo):
    """
    Generates chord progressions using functional harmony rules.
    
    This algorithm implements common progression patterns from tonal harmony,
    including circle of fifths movements, ii-V-I progressions, and other
    standard patterns.
    """
    
    def __init__(self, key: int = 0, mode: str = 'major', 
                 pattern: Optional[str] = None, randomness: float = 0.1) -> None:
        """
        Initialize the rule-based progression generator.
        
        Args:
            key: Key center (0-11, where C=0)
            mode: Mode ('major' or 'minor')
            pattern: Named progression pattern to use
            randomness: Amount of randomness to inject (0.0-1.0)
        """
        self.key = key
        self.mode = mode
        self.pattern = pattern
        self.randomness = randomness
        
        # Define scale degrees and their typical functions
        self._setup_harmonic_functions()
        self._setup_progression_patterns()
    
    def _setup_harmonic_functions(self) -> None:
        """Set up harmonic function mappings for the current key and mode."""
        if self.mode == 'major':
            # Roman numeral degrees for major key
            self.scale_degrees = {
                0: {'numeral': 'I', 'quality': 'major', 'function': 'tonic'},
                2: {'numeral': 'ii', 'quality': 'minor', 'function': 'predominant'},
                4: {'numeral': 'iii', 'quality': 'minor', 'function': 'tonic'},
                5: {'numeral': 'IV', 'quality': 'major', 'function': 'predominant'},
                7: {'numeral': 'V', 'quality': 'major', 'function': 'dominant'},
                9: {'numeral': 'vi', 'quality': 'minor', 'function': 'tonic'},
                11: {'numeral': 'vii°', 'quality': 'diminished', 'function': 'dominant'},
            }
        else:  # minor
            self.scale_degrees = {
                0: {'numeral': 'i', 'quality': 'minor', 'function': 'tonic'},
                2: {'numeral': 'ii°', 'quality': 'diminished', 'function': 'predominant'},
                3: {'numeral': 'III', 'quality': 'major', 'function': 'tonic'},
                5: {'numeral': 'iv', 'quality': 'minor', 'function': 'predominant'},
                7: {'numeral': 'V', 'quality': 'major', 'function': 'dominant'},  # Harmonic minor
                8: {'numeral': 'VI', 'quality': 'major', 'function': 'tonic'},
                10: {'numeral': 'VII', 'quality': 'major', 'function': 'subtonic'},
            }
        
        # Function-based chord groups
        self.tonic_chords = [deg for deg, info in self.scale_degrees.items() 
                           if info['function'] == 'tonic']
        self.predominant_chords = [deg for deg, info in self.scale_degrees.items() 
                                 if info['function'] == 'predominant']
        self.dominant_chords = [deg for deg, info in self.scale_degrees.items() 
                              if info['function'] == 'dominant']
    
    def _setup_progression_patterns(self) -> None:
        """Set up named progression patterns."""
        self.patterns = {
            'ii-V-I': [2, 7, 0],  # ii - V - I
            'vi-IV-V-I': [9, 5, 7, 0],  # vi - IV - V - I
            'I-vi-IV-V': [0, 9, 5, 7],  # I - vi - IV - V (50s progression)
            'circle_of_fifths': [0, 7, 2, 9, 4, 11, 5],  # Descending fifths
            'blues': [0, 0, 0, 0, 5, 5, 0, 0, 7, 5, 0, 7],  # 12-bar blues
            'modal_iv': [0, 5, 10, 0],  # I - IV - bVII - I (modal)
            'chromatic_mediant': [0, 8, 4, 0],  # I - bVI - iii - I
        }
        
        # Convert patterns to actual chord roots in the current key
        self.resolved_patterns = {}
        for name, pattern in self.patterns.items():
            resolved = [(self.key + degree) % 12 for degree in pattern]
            self.resolved_patterns[name] = resolved
    
    def _get_chord_from_degree(self, degree: int) -> Chord:
        """
        Create a chord from a scale degree.
        
        Args:
            degree: Scale degree (0-11)
            
        Returns:
            Chord object for the scale degree
        """
        # Normalize degree to scale
        scale_degree = degree % 12
        
        if scale_degree in self.scale_degrees:
            info = self.scale_degrees[scale_degree]
            root = (self.key + scale_degree) % 12
            return Chord(root, info['quality'])
        else:
            # For non-diatonic degrees, use chord from chromatic context
            root = (self.key + scale_degree) % 12
            # Simple heuristic: major for most, minor for some specific cases
            quality = 'minor' if scale_degree in [1, 3, 6, 8, 10] else 'major'
            return Chord(root, quality)
    
    def _get_function_candidates(self, current_function: str) -> List[str]:
        """
        Get candidate functions that can follow the current function.
        
        Args:
            current_function: Current harmonic function
            
        Returns:
            List of functions that can logically follow
        """
        # Common function progressions in tonal harmony
        function_rules = {
            'tonic': ['predominant', 'dominant', 'tonic'],
            'predominant': ['dominant', 'tonic'],
            'dominant': ['tonic', 'predominant'],
            'subtonic': ['tonic'],  # For minor keys
        }
        
        return function_rules.get(current_function, ['tonic'])
    
    def _select_chord_by_function(self, target_function: str, 
                                previous_chord: Optional[Chord] = None) -> Chord:
        """
        Select a chord with the specified harmonic function.
        
        Args:
            target_function: Desired harmonic function
            previous_chord: Previous chord for voice leading considerations
            
        Returns:
            Chord with the specified function
        """
        # Get candidates for this function
        candidates = []
        for degree, info in self.scale_degrees.items():
            if info['function'] == target_function:
                candidates.append(self._get_chord_from_degree(degree))
        
        if not candidates:
            # Fallback to tonic if no candidates found
            return self._get_chord_from_degree(0)
        
        # If no previous chord, return random candidate
        if previous_chord is None:
            return random.choice(candidates)
        
        # Choose based on voice leading distance
        best_chord = candidates[0]
        best_distance = previous_chord.voice_leading_distance(best_chord)
        
        for candidate in candidates[1:]:
            distance = previous_chord.voice_leading_distance(candidate)
            if distance < best_distance:
                best_distance = distance
                best_chord = candidate
        
        # Add some randomness
        if random.random() < self.randomness:
            return random.choice(candidates)
        
        return best_chord
    
    def _apply_pattern(self, pattern_name: str, length: int, start: Chord) -> List[Chord]:
        """
        Apply a named progression pattern.
        
        Args:
            pattern_name: Name of the pattern to apply
            length: Desired progression length
            start: Starting chord
            
        Returns:
            Generated progression following the pattern
        """
        if pattern_name not in self.resolved_patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        pattern = self.resolved_patterns[pattern_name]
        progression = [start]
        
        # Cycle through pattern to reach desired length
        pattern_index = 0
        
        for i in range(1, length):
            # Get next root from pattern
            root = pattern[pattern_index % len(pattern)]
            
            # Determine quality based on scale degree
            scale_degree = (root - self.key) % 12
            if scale_degree in self.scale_degrees:
                quality = self.scale_degrees[scale_degree]['quality']
            else:
                quality = 'major'  # Default
            
            chord = Chord(root, quality)
            
            # Add some variation with randomness
            if random.random() < self.randomness:
                # Occasionally substitute with a chord of different quality
                alternate_qualities = ['major', 'minor']
                if quality in alternate_qualities:
                    alternate_qualities.remove(quality)
                if alternate_qualities:
                    chord = Chord(root, random.choice(alternate_qualities))
            
            progression.append(chord)
            pattern_index += 1
        
        return progression
    
    def _generate_functional_progression(self, start: Chord, length: int) -> List[Chord]:
        """
        Generate progression based on harmonic function rules.
        
        Args:
            start: Starting chord
            length: Number of chords to generate
            
        Returns:
            Generated progression
        """
        progression = [start]
        
        # Determine function of starting chord
        start_degree = (start.root - self.key) % 12
        if start_degree in self.scale_degrees:
            current_function = self.scale_degrees[start_degree]['function']
        else:
            current_function = 'tonic'  # Default assumption
        
        for i in range(1, length):
            # Get candidate functions
            next_functions = self._get_function_candidates(current_function)
            
            # Add some randomness to function selection
            if random.random() < self.randomness and len(next_functions) > 1:
                selected_function = random.choice(next_functions)
            else:
                # Use most common progression
                selected_function = next_functions[0]
            
            # Select chord with the chosen function
            next_chord = self._select_chord_by_function(selected_function, progression[-1])
            progression.append(next_chord)
            
            # Update current function
            next_degree = (next_chord.root - self.key) % 12
            if next_degree in self.scale_degrees:
                current_function = self.scale_degrees[next_degree]['function']
        
        return progression
    
    def generate(self, start: Chord, length: int, **kwargs: Any) -> List[Chord]:
        """
        Generate a rule-based chord progression.
        
        Args:
            start: Starting chord
            length: Number of chords to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated chord progression
        """
        if length <= 0:
            return []
        
        if length == 1:
            return [start]
        
        # Update parameters from kwargs
        pattern = kwargs.get('pattern', self.pattern)
        
        if pattern and pattern in self.patterns:
            return self._apply_pattern(pattern, length, start)
        else:
            return self._generate_functional_progression(start, length)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current algorithm parameters."""
        return {
            'key': self.key,
            'mode': self.mode,
            'pattern': self.pattern,
            'randomness': self.randomness,
        }
    
    def set_parameters(self, **kwargs: Any) -> None:
        """Set algorithm parameters."""
        if 'key' in kwargs:
            self.key = kwargs['key'] % 12
            self._setup_harmonic_functions()
            self._setup_progression_patterns()
        
        if 'mode' in kwargs:
            if kwargs['mode'] not in ['major', 'minor']:
                raise ValueError("Mode must be 'major' or 'minor'")
            self.mode = kwargs['mode']
            self._setup_harmonic_functions()
            self._setup_progression_patterns()
        
        if 'pattern' in kwargs:
            pattern = kwargs['pattern']
            if pattern is not None and pattern not in self.patterns:
                raise ValueError(f"Unknown pattern: {pattern}")
            self.pattern = pattern
        
        if 'randomness' in kwargs:
            randomness = kwargs['randomness']
            if not 0.0 <= randomness <= 1.0:
                raise ValueError("Randomness must be between 0.0 and 1.0")
            self.randomness = randomness
    
    def get_available_patterns(self) -> List[str]:
        """
        Get list of available progression patterns.
        
        Returns:
            List of pattern names
        """
        return list(self.patterns.keys())
    
    def analyze_chord_in_key(self, chord: Chord) -> Dict[str, Any]:
        """
        Analyze a chord's function within the current key.
        
        Args:
            chord: Chord to analyze
            
        Returns:
            Dictionary containing harmonic analysis
        """
        degree = (chord.root - self.key) % 12
        
        if degree in self.scale_degrees:
            info = self.scale_degrees[degree].copy()
            info['degree'] = degree
            info['diatonic'] = True
        else:
            info = {
                'degree': degree,
                'numeral': f'♭{degree}' if degree < 6 else f'#{degree}',
                'quality': chord.quality,
                'function': 'chromatic',
                'diatonic': False
            }
        
        return info


# Convenience functions for common progressions
def generate_ii_V_I(key: int = 0, mode: str = 'major') -> List[Chord]:
    """Generate a ii-V-I progression in the specified key."""
    algo = RuleBasedProgression(key=key, mode=mode, pattern='ii-V-I')
    return algo.generate(Chord(key, 'major' if mode == 'major' else 'minor'), 3)


def generate_circle_of_fifths(key: int = 0, length: int = 7) -> List[Chord]:
    """Generate a circle of fifths progression."""
    algo = RuleBasedProgression(key=key, pattern='circle_of_fifths')
    return algo.generate(Chord(key, 'major'), length)


def generate_blues_progression(key: int = 0) -> List[Chord]:
    """Generate a 12-bar blues progression."""
    algo = RuleBasedProgression(key=key, pattern='blues')
    return algo.generate(Chord(key, 'major'), 12)
