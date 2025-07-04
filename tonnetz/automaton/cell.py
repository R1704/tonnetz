"""
Cell representation for the cellular automaton.

This module defines the Cell class that represents individual agents
in the cellular automaton grid, each containing a chord and state.
"""

from typing import Optional, Dict, Any
from tonnetz.core.chord import Chord


class Cell:
    """
    Represents a single cell/agent in the cellular automaton.
    
    Each cell contains a chord and can maintain additional state information
    for complex automaton behaviors.
    """
    
    def __init__(self, chord: Chord, x: int = 0, y: int = 0, 
                 state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a cell with a chord and position.
        
        Args:
            chord: The chord contained in this cell
            x: X coordinate in the grid
            y: Y coordinate in the grid
            state: Optional dictionary for additional state information
        """
        self.chord = chord
        self.x = x
        self.y = y
        self.state = state or {}
        
        # For double-buffering during updates
        self.next_chord: Optional[Chord] = None
        self.next_state: Optional[Dict[str, Any]] = None
        
        # Activation and energy for more complex behaviors
        self.activation = 1.0
        self.energy = 1.0
        
        # History tracking
        self.chord_history = [chord]
        self.max_history = 10  # Limit history size
    
    def update_chord(self, new_chord: Chord) -> None:
        """
        Update the cell's chord and record in history.
        
        Args:
            new_chord: The new chord for this cell
        """
        self.chord = new_chord
        self.chord_history.append(new_chord)
        
        # Limit history size
        if len(self.chord_history) > self.max_history:
            self.chord_history.pop(0)
    
    def prepare_update(self, new_chord: Chord, new_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Prepare the next update (for double-buffering).
        
        Args:
            new_chord: The chord for the next time step
            new_state: Optional new state for the next time step
        """
        self.next_chord = new_chord
        self.next_state = new_state or {}
    
    def commit_update(self) -> None:
        """
        Commit the prepared update to become the current state.
        """
        if self.next_chord is not None:
            self.update_chord(self.next_chord)
            self.next_chord = None
        
        if self.next_state is not None:
            self.state.update(self.next_state)
            self.next_state = None
    
    def get_state_value(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cell's state.
        
        Args:
            key: State key to retrieve
            default: Default value if key not found
            
        Returns:
            The state value or default
        """
        return self.state.get(key, default)
    
    def set_state_value(self, key: str, value: Any) -> None:
        """
        Set a value in the cell's state.
        
        Args:
            key: State key to set
            value: Value to set
        """
        self.state[key] = value
    
    def get_chord_stability(self) -> float:
        """
        Calculate the stability of the chord in this cell.
        
        Based on how long the current chord has been stable.
        
        Returns:
            Stability value (0.0 to 1.0)
        """
        if len(self.chord_history) < 2:
            return 1.0
        
        # Count consecutive occurrences of current chord
        consecutive_count = 1
        for i in range(len(self.chord_history) - 2, -1, -1):
            if self.chord_history[i] == self.chord:
                consecutive_count += 1
            else:
                break
        
        # Normalize by history length
        return min(consecutive_count / len(self.chord_history), 1.0)
    
    def get_chord_diversity(self) -> float:
        """
        Calculate the diversity of chords that have appeared in this cell.
        
        Returns:
            Diversity value (0.0 to 1.0)
        """
        if len(self.chord_history) <= 1:
            return 0.0
        
        unique_chords = len(set(self.chord_history))
        return unique_chords / len(self.chord_history)
    
    def has_changed_recently(self, steps: int = 1) -> bool:
        """
        Check if the chord has changed in the last N steps.
        
        Args:
            steps: Number of steps to look back
            
        Returns:
            True if chord changed in the specified timeframe
        """
        if len(self.chord_history) <= steps:
            return len(set(self.chord_history)) > 1
        
        recent_chords = self.chord_history[-steps-1:]
        return len(set(recent_chords)) > 1
    
    def get_previous_chord(self, steps_back: int = 1) -> Optional[Chord]:
        """
        Get a chord from the cell's history.
        
        Args:
            steps_back: How many steps back to look (1 = previous chord)
            
        Returns:
            Chord from history or None if not available
        """
        if len(self.chord_history) <= steps_back:
            return None
        
        return self.chord_history[-(steps_back + 1)]
    
    def copy(self) -> 'Cell':
        """
        Create a copy of this cell.
        
        Returns:
            New Cell object with copied data
        """
        new_cell = Cell(self.chord, self.x, self.y, self.state.copy())
        new_cell.activation = self.activation
        new_cell.energy = self.energy
        new_cell.chord_history = self.chord_history.copy()
        new_cell.max_history = self.max_history
        return new_cell
    
    def distance_to(self, other: 'Cell') -> float:
        """
        Calculate the harmonic distance to another cell.
        
        Args:
            other: Another cell
            
        Returns:
            Harmonic distance between the cells' chords
        """
        return self.chord.voice_leading_distance(other.chord)
    
    def influence_from(self, other: 'Cell', distance: float) -> float:
        """
        Calculate the influence this cell receives from another cell.
        
        Args:
            other: The influencing cell
            distance: Geometric distance between cells
            
        Returns:
            Influence strength (0.0 to 1.0)
        """
        # Influence decreases with distance
        if distance == 0:
            return other.activation
        
        # Use exponential decay
        base_influence = other.activation / (1.0 + distance)
        
        # Chord similarity affects influence
        harmonic_distance = self.distance_to(other)
        harmonic_factor = 1.0 / (1.0 + harmonic_distance * 0.1)
        
        return base_influence * harmonic_factor
    
    def update_activation(self, influences: float) -> None:
        """
        Update the cell's activation based on external influences.
        
        Args:
            influences: Total influence received from neighbors
        """
        # Simple activation function
        self.activation = max(0.0, min(1.0, influences))
    
    def update_energy(self, consumption: float = 0.1) -> None:
        """
        Update the cell's energy level.
        
        Args:
            consumption: Energy consumption per step
        """
        self.energy = max(0.0, self.energy - consumption)
        
        # Regenerate energy slowly
        if self.energy < 1.0:
            self.energy = min(1.0, self.energy + 0.05)
    
    def is_active(self, threshold: float = 0.5) -> bool:
        """
        Check if the cell is currently active.
        
        Args:
            threshold: Activation threshold
            
        Returns:
            True if activation is above threshold
        """
        return self.activation >= threshold
    
    def __str__(self) -> str:
        """String representation of the cell."""
        return f"Cell({self.chord}, pos=({self.x},{self.y}), act={self.activation:.2f})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the cell."""
        return (f"Cell(chord={self.chord!r}, x={self.x}, y={self.y}, "
                f"activation={self.activation}, energy={self.energy})")
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another cell."""
        if not isinstance(other, Cell):
            return NotImplemented
        return (self.chord == other.chord and 
                self.x == other.x and 
                self.y == other.y)
    
    def __hash__(self) -> int:
        """Hash function for using cells in sets/dicts."""
        return hash((self.chord, self.x, self.y))


class CellNeighborhood:
    """
    Utility class for managing cell neighborhoods and interactions.
    """
    
    @staticmethod
    def calculate_neighborhood_influence(center: Cell, neighbors: list[Cell], 
                                       distances: list[float]) -> float:
        """
        Calculate total influence on a center cell from its neighbors.
        
        Args:
            center: The central cell
            neighbors: List of neighboring cells
            distances: List of distances to each neighbor
            
        Returns:
            Total influence value
        """
        total_influence = 0.0
        
        for neighbor, distance in zip(neighbors, distances):
            influence = center.influence_from(neighbor, distance)
            total_influence += influence
        
        return total_influence
    
    @staticmethod
    def find_dominant_neighbor(center: Cell, neighbors: list[Cell], 
                             distances: list[float]) -> Optional[Cell]:
        """
        Find the neighbor with the strongest influence on the center cell.
        
        Args:
            center: The central cell
            neighbors: List of neighboring cells
            distances: List of distances to each neighbor
            
        Returns:
            The most influential neighbor or None if no neighbors
        """
        if not neighbors:
            return None
        
        max_influence = 0.0
        dominant_neighbor = None
        
        for neighbor, distance in zip(neighbors, distances):
            influence = center.influence_from(neighbor, distance)
            if influence > max_influence:
                max_influence = influence
                dominant_neighbor = neighbor
        
        return dominant_neighbor
    
    @staticmethod
    def calculate_consensus_chord(neighbors: list[Cell], weights: Optional[list[float]] = None) -> Optional[Chord]:
        """
        Calculate a consensus chord based on neighbor chords.
        
        Args:
            neighbors: List of neighboring cells
            weights: Optional weights for each neighbor
            
        Returns:
            Consensus chord or None if no consensus possible
        """
        if not neighbors:
            return None
        
        if weights is None:
            weights = [1.0] * len(neighbors)
        
        # Simple majority vote for now
        # More sophisticated versions could use harmonic distance
        chord_votes = {}
        
        for neighbor, weight in zip(neighbors, weights):
            chord = neighbor.chord
            if chord not in chord_votes:
                chord_votes[chord] = 0.0
            chord_votes[chord] += weight
        
        # Return chord with highest vote
        if chord_votes:
            return max(chord_votes.items(), key=lambda x: x[1])[0]
        
        return None
    
    @staticmethod
    def calculate_harmony_level(neighbors: list[Cell]) -> float:
        """
        Calculate the harmonic consonance level of a neighborhood.
        
        Args:
            neighbors: List of neighboring cells
            
        Returns:
            Harmony level (0.0 to 1.0, higher = more consonant)
        """
        if len(neighbors) < 2:
            return 1.0
        
        total_distance = 0.0
        pair_count = 0
        
        # Calculate average pairwise harmonic distance
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                distance = neighbors[i].distance_to(neighbors[j])
                total_distance += distance
                pair_count += 1
        
        if pair_count == 0:
            return 1.0
        
        average_distance = total_distance / pair_count
        
        # Convert distance to harmony (inverse relationship)
        # Assuming max reasonable distance is around 12 semitones
        harmony = 1.0 - min(average_distance / 12.0, 1.0)
        
        return harmony
