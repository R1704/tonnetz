"""
Base classes and interfaces for chord progression algorithms.

This module defines the abstract interfaces that all progression algorithms
must implement, enabling a pluggable architecture for different approaches.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from tonnetz.core.chord import Chord


class ProgressionAlgo(ABC):
    """
    Abstract base class for all chord progression algorithms.
    
    This interface ensures that all progression algorithms provide a consistent
    API for generating chord sequences from a starting chord.
    """
    
    @abstractmethod
    def generate(self, start: Chord, length: int, **kwargs: Any) -> List[Chord]:
        """
        Generate a chord progression starting from a given chord.
        
        Args:
            start: Starting chord for the progression
            length: Number of chords to generate (including start chord)
            **kwargs: Algorithm-specific parameters
            
        Returns:
            List of chords forming the progression
            
        Raises:
            ValueError: If parameters are invalid
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current parameters/configuration of the algorithm.
        
        Returns:
            Dictionary of parameter names and their current values
        """
        pass
    
    @abstractmethod
    def set_parameters(self, **kwargs: Any) -> None:
        """
        Set algorithm parameters.
        
        Args:
            **kwargs: Parameter names and values to set
            
        Raises:
            ValueError: If any parameter name or value is invalid
        """
        pass
    
    def validate_parameters(self, **kwargs: Any) -> None:
        """
        Validate algorithm parameters.
        
        Override this method to implement parameter validation specific
        to your algorithm.
        
        Args:
            **kwargs: Parameters to validate
            
        Raises:
            ValueError: If any parameter is invalid
        """
        pass
    
    def reset(self) -> None:
        """
        Reset the algorithm to its initial state.
        
        This is useful for algorithms with internal state (e.g., Markov chains)
        that need to be reset between progression generations.
        """
        pass


class StatefulProgressionAlgo(ProgressionAlgo):
    """
    Base class for progression algorithms that maintain internal state.
    
    This class provides common functionality for algorithms that need to
    track state between chord generations, such as Markov chains or
    rule-based systems with memory.
    """
    
    def __init__(self) -> None:
        """Initialize the stateful algorithm."""
        self._state_history: List[Chord] = []
        self._current_position = 0
    
    def get_state_history(self) -> List[Chord]:
        """
        Get the history of states/chords processed by the algorithm.
        
        Returns:
            List of chords in the order they were processed
        """
        return self._state_history.copy()
    
    def get_current_position(self) -> int:
        """
        Get the current position in the progression.
        
        Returns:
            Current position index
        """
        return self._current_position
    
    def reset(self) -> None:
        """Reset the algorithm state."""
        self._state_history.clear()
        self._current_position = 0
    
    def _record_state(self, chord: Chord) -> None:
        """
        Record a chord in the state history.
        
        Args:
            chord: Chord to record
        """
        self._state_history.append(chord)
        self._current_position = len(self._state_history) - 1


class CompositeProgressionAlgo(ProgressionAlgo):
    """
    Composite algorithm that combines multiple progression algorithms.
    
    This allows for complex progression generation by switching between
    different algorithms or combining their outputs.
    """
    
    def __init__(self, algorithms: List[ProgressionAlgo], strategy: str = 'sequential') -> None:
        """
        Initialize the composite algorithm.
        
        Args:
            algorithms: List of progression algorithms to combine
            strategy: Strategy for combining algorithms ('sequential', 'random', 'weighted')
        """
        if not algorithms:
            raise ValueError("At least one algorithm must be provided")
        
        self.algorithms = algorithms
        self.strategy = strategy
        self._weights: Optional[List[float]] = None
        
        if strategy not in ['sequential', 'random', 'weighted']:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def set_weights(self, weights: List[float]) -> None:
        """
        Set weights for weighted strategy.
        
        Args:
            weights: List of weights for each algorithm (must sum to 1.0)
        """
        if len(weights) != len(self.algorithms):
            raise ValueError("Number of weights must match number of algorithms")
        
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        self._weights = weights
    
    def generate(self, start: Chord, length: int, **kwargs: Any) -> List[Chord]:
        """
        Generate progression using the composite strategy.
        
        Args:
            start: Starting chord
            length: Number of chords to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated chord progression
        """
        if self.strategy == 'sequential':
            return self._generate_sequential(start, length, **kwargs)
        elif self.strategy == 'random':
            return self._generate_random(start, length, **kwargs)
        elif self.strategy == 'weighted':
            return self._generate_weighted(start, length, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _generate_sequential(self, start: Chord, length: int, **kwargs: Any) -> List[Chord]:
        """Generate using algorithms in sequence."""
        import math
        
        progression = [start]
        remaining = length - 1
        
        if remaining <= 0:
            return progression
        
        # Divide remaining chords among algorithms
        chords_per_algo = math.ceil(remaining / len(self.algorithms))
        
        current_chord = start
        for i, algo in enumerate(self.algorithms):
            if remaining <= 0:
                break
            
            # Generate segment with this algorithm
            segment_length = min(chords_per_algo, remaining) + 1  # +1 for starting chord
            segment = algo.generate(current_chord, segment_length, **kwargs)
            
            # Add all but the first chord (avoid duplication)
            progression.extend(segment[1:])
            remaining -= len(segment) - 1
            
            if segment:
                current_chord = segment[-1]
        
        return progression[:length]  # Ensure exact length
    
    def _generate_random(self, start: Chord, length: int, **kwargs: Any) -> List[Chord]:
        """Generate using randomly selected algorithms."""
        import random
        
        progression = [start]
        current_chord = start
        
        for _ in range(length - 1):
            # Randomly select an algorithm
            algo = random.choice(self.algorithms)
            
            # Generate next chord
            segment = algo.generate(current_chord, 2, **kwargs)  # Current + next
            if len(segment) > 1:
                next_chord = segment[1]
                progression.append(next_chord)
                current_chord = next_chord
            else:
                progression.append(current_chord)  # Fallback
        
        return progression
    
    def _generate_weighted(self, start: Chord, length: int, **kwargs: Any) -> List[Chord]:
        """Generate using weighted selection of algorithms."""
        import random
        
        if self._weights is None:
            raise ValueError("Weights must be set for weighted strategy")
        
        progression = [start]
        current_chord = start
        
        for _ in range(length - 1):
            # Weighted selection of algorithm
            algo = random.choices(self.algorithms, weights=self._weights)[0]
            
            # Generate next chord
            segment = algo.generate(current_chord, 2, **kwargs)
            if len(segment) > 1:
                next_chord = segment[1]
                progression.append(next_chord)
                current_chord = next_chord
            else:
                progression.append(current_chord)  # Fallback
        
        return progression
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters from all constituent algorithms."""
        params = {
            'strategy': self.strategy,
            'algorithms': [algo.get_parameters() for algo in self.algorithms]
        }
        if self._weights is not None:
            params['weights'] = self._weights
        return params
    
    def set_parameters(self, **kwargs: Any) -> None:
        """Set parameters for the composite algorithm."""
        if 'strategy' in kwargs:
            self.strategy = kwargs['strategy']
        
        if 'weights' in kwargs:
            self.set_weights(kwargs['weights'])
        
        # Set parameters for individual algorithms
        if 'algorithm_params' in kwargs:
            algo_params = kwargs['algorithm_params']
            for i, params in enumerate(algo_params):
                if i < len(self.algorithms):
                    self.algorithms[i].set_parameters(**params)
    
    def reset(self) -> None:
        """Reset all constituent algorithms."""
        for algo in self.algorithms:
            algo.reset()


class ProgressionAnalyzer:
    """
    Utility class for analyzing chord progressions.
    
    Provides methods to analyze the harmonic content, voice leading,
    and other properties of generated progressions.
    """
    
    @staticmethod
    def analyze_voice_leading(progression: List[Chord]) -> Dict[str, float]:
        """
        Analyze voice leading efficiency in a progression.
        
        Args:
            progression: List of chords to analyze
            
        Returns:
            Dictionary containing voice leading metrics
        """
        if len(progression) < 2:
            return {'total_distance': 0.0, 'average_distance': 0.0, 'max_distance': 0.0}
        
        distances = []
        for i in range(len(progression) - 1):
            distance = progression[i].voice_leading_distance(progression[i + 1])
            distances.append(distance)
        
        return {
            'total_distance': sum(distances),
            'average_distance': sum(distances) / len(distances),
            'max_distance': max(distances),
            'min_distance': min(distances),
            'distances': distances
        }
    
    @staticmethod
    def analyze_harmonic_rhythm(progression: List[Chord]) -> Dict[str, Any]:
        """
        Analyze the harmonic rhythm and chord change patterns.
        
        Args:
            progression: List of chords to analyze
            
        Returns:
            Dictionary containing harmonic rhythm analysis
        """
        if len(progression) < 2:
            return {'unique_chords': len(set(progression)), 'repetitions': []}
        
        unique_chords = len(set(progression))
        
        # Find repetition patterns
        repetitions = []
        for i in range(len(progression) - 1):
            if progression[i] == progression[i + 1]:
                repetitions.append(i)
        
        # Analyze chord quality distribution
        quality_counts = {}
        for chord in progression:
            quality_counts[chord.quality] = quality_counts.get(chord.quality, 0) + 1
        
        return {
            'length': len(progression),
            'unique_chords': unique_chords,
            'repetition_rate': len(repetitions) / (len(progression) - 1),
            'repetition_positions': repetitions,
            'quality_distribution': quality_counts
        }
    
    @staticmethod
    def analyze_neo_riemannian_moves(progression: List[Chord]) -> Dict[str, Any]:
        """
        Analyze neo-Riemannian transformations used in the progression.
        
        Args:
            progression: List of chords to analyze
            
        Returns:
            Dictionary containing neo-Riemannian analysis
        """
        from tonnetz.core.neo_riemannian import PLRGroup
        
        if len(progression) < 2:
            return {'transformations': [], 'plr_efficiency': 0.0}
        
        plr = PLRGroup()
        transformations = []
        
        for i in range(len(progression) - 1):
            current = progression[i]
            next_chord = progression[i + 1]
            
            # Find the PLR transformation sequence
            trans_sequence = plr.find_transformation_sequence(current, next_chord)
            transformations.append({
                'from': current,
                'to': next_chord,
                'transformation': trans_sequence,
                'length': len(trans_sequence)
            })
        
        # Calculate PLR efficiency (average transformation length)
        trans_lengths = [t['length'] for t in transformations if t['transformation']]
        plr_efficiency = sum(trans_lengths) / len(trans_lengths) if trans_lengths else 0.0
        
        return {
            'transformations': transformations,
            'plr_efficiency': plr_efficiency,
            'direct_plr_moves': len([t for t in transformations if len(t['transformation']) == 1])
        }
