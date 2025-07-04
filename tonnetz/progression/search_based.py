"""
Search-based chord progression algorithm.

This module implements progression algorithms that use search techniques
to find chord progressions that optimize certain musical criteria.
"""

import random
import heapq
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from tonnetz.core.chord import Chord
from tonnetz.core.neo_riemannian import PLRGroup
from tonnetz.progression.base import ProgressionAlgo


class SearchObjective(Enum):
    """Objectives for search-based progression generation."""
    SMOOTH_VOICE_LEADING = "smooth_voice_leading"
    HARMONIC_TENSION = "harmonic_tension"
    NEO_RIEMANNIAN_EFFICIENCY = "neo_riemannian_efficiency"
    CHORD_DIVERSITY = "chord_diversity"
    TONAL_STABILITY = "tonal_stability"


@dataclass
class SearchState:
    """Represents a state in the search space."""
    progression: List[Chord]
    score: float
    heuristic: float
    
    def __lt__(self, other):
        return (self.score + self.heuristic) < (other.score + other.heuristic)


class SearchBasedProgression(ProgressionAlgo):
    """
    Search-based chord progression generator.
    
    This algorithm uses various search techniques (A*, beam search, genetic algorithms)
    to find chord progressions that optimize musical objectives.
    """
    
    def __init__(self, 
                 search_method: str = "beam",
                 objectives: List[SearchObjective] = None,
                 objective_weights: List[float] = None,
                 **kwargs):
        """
        Initialize the search-based progression generator.
        
        Args:
            search_method: Search method to use ("beam", "astar", "genetic")
            objectives: List of objectives to optimize
            objective_weights: Weights for each objective
            **kwargs: Additional parameters
        """
        super().__init__()
        self.search_method = search_method
        self.objectives = objectives or [SearchObjective.SMOOTH_VOICE_LEADING]
        self.objective_weights = objective_weights or [1.0] * len(self.objectives)
        
        self.beam_width = kwargs.get('beam_width', 10)
        self.max_iterations = kwargs.get('max_iterations', 1000)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        self.population_size = kwargs.get('population_size', 50)
        
        self.plr_group = PLRGroup()
        
        # Cache for expensive computations
        self._voice_leading_cache: Dict[Tuple[str, str], float] = {}
        
    def generate(self, start_chord: Chord, length: int, **kwargs) -> List[Chord]:
        """
        Generate a chord progression using search algorithms.
        
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
        Generate a chord progression using search algorithms.
        
        Args:
            start_chord: Starting chord
            length: Desired length of progression
            
        Returns:
            List of chords in the progression
        """
        if self.search_method == "beam":
            return self._beam_search(start_chord, length)
        elif self.search_method == "astar":
            return self._astar_search(start_chord, length)
        elif self.search_method == "genetic":
            return self._genetic_algorithm(start_chord, length)
        else:
            raise ValueError(f"Unknown search method: {self.search_method}")
    
    def _beam_search(self, start_chord: Chord, length: int) -> List[Chord]:
        """Generate progression using beam search."""
        beam = [SearchState([start_chord], 0.0, self._heuristic([start_chord], length))]
        
        for step in range(length - 1):
            new_beam = []
            
            for state in beam:
                candidates = self._generate_candidates(state.progression[-1])
                
                for candidate in candidates:
                    new_progression = state.progression + [candidate]
                    score = self._evaluate_progression(new_progression)
                    heuristic = self._heuristic(new_progression, length)
                    
                    new_state = SearchState(new_progression, score, heuristic)
                    new_beam.append(new_state)
            
            # Keep only the best beam_width states
            beam = heapq.nlargest(self.beam_width, new_beam, 
                                 key=lambda s: s.score + s.heuristic)
        
        # Return the best progression
        return max(beam, key=lambda s: s.score).progression if beam else [start_chord]
    
    def _astar_search(self, start_chord: Chord, length: int) -> List[Chord]:
        """Generate progression using A* search."""
        open_set = [SearchState([start_chord], 0.0, self._heuristic([start_chord], length))]
        closed_set: Set[str] = set()
        best_complete = None
        iterations = 0
        
        while open_set and iterations < self.max_iterations:
            iterations += 1
            current = heapq.heappop(open_set)
            
            # Check if we've reached the target length
            if len(current.progression) == length:
                if best_complete is None or current.score > best_complete.score:
                    best_complete = current
                continue
            
            # Generate successors
            candidates = self._generate_candidates(current.progression[-1])
            
            for candidate in candidates:
                new_progression = current.progression + [candidate]
                progression_key = "_".join(str(c) for c in new_progression)
                
                if progression_key in closed_set:
                    continue
                
                score = self._evaluate_progression(new_progression)
                heuristic = self._heuristic(new_progression, length)
                
                new_state = SearchState(new_progression, score, heuristic)
                heapq.heappush(open_set, new_state)
                closed_set.add(progression_key)
        
        return best_complete.progression if best_complete else [start_chord]
    
    def _genetic_algorithm(self, start_chord: Chord, length: int) -> List[Chord]:
        """Generate progression using genetic algorithm."""
        # Initialize population
        population = []
        for _ in range(self.population_size):
            progression = self._random_progression(start_chord, length)
            population.append(progression)
        
        for generation in range(self.max_iterations // 10):
            # Evaluate fitness
            fitness_scores = [self._evaluate_progression(prog) for prog in population]
            
            # Selection
            new_population = []
            for _ in range(self.population_size):
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                child = self._crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    child = self._mutate(child, start_chord)
                new_population.append(child)
            
            population = new_population
        
        # Return best individual
        final_scores = [self._evaluate_progression(prog) for prog in population]
        best_idx = max(range(len(final_scores)), key=lambda i: final_scores[i])
        return population[best_idx]
    
    def _generate_candidates(self, current_chord: Chord) -> List[Chord]:
        """Generate candidate next chords."""
        candidates = []
        
        # PLR transformations
        candidates.extend([
            self.plr_group.parallel(current_chord),
            self.plr_group.leading_tone_exchange(current_chord),
            self.plr_group.relative(current_chord)
        ])
        
        # Common chord progressions
        root = current_chord.root
        for interval in [2, 4, 5, 7, 9]:  # ii, iii, IV, V, vi relationships
            new_root = (root + interval) % 12
            for quality in ['major', 'minor']:
                candidates.append(Chord(new_root, quality))
        
        # Remove duplicates
        seen = set()
        unique_candidates = []
        for chord in candidates:
            chord_key = (chord.root, chord.quality)
            if chord_key not in seen:
                seen.add(chord_key)
                unique_candidates.append(chord)
        
        return unique_candidates[:8]  # Limit to 8 candidates for efficiency
    
    def _evaluate_progression(self, progression: List[Chord]) -> float:
        """Evaluate the quality of a chord progression."""
        total_score = 0.0
        
        for objective, weight in zip(self.objectives, self.objective_weights):
            if objective == SearchObjective.SMOOTH_VOICE_LEADING:
                score = self._evaluate_voice_leading(progression)
            elif objective == SearchObjective.HARMONIC_TENSION:
                score = self._evaluate_harmonic_tension(progression)
            elif objective == SearchObjective.NEO_RIEMANNIAN_EFFICIENCY:
                score = self._evaluate_neo_riemannian_efficiency(progression)
            elif objective == SearchObjective.CHORD_DIVERSITY:
                score = self._evaluate_chord_diversity(progression)
            elif objective == SearchObjective.TONAL_STABILITY:
                score = self._evaluate_tonal_stability(progression)
            else:
                score = 0.0
            
            total_score += weight * score
        
        return total_score
    
    def _evaluate_voice_leading(self, progression: List[Chord]) -> float:
        """Evaluate smoothness of voice leading."""
        if len(progression) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(progression) - 1):
            distance = progression[i].voice_leading_distance(progression[i + 1])
            total_distance += distance
        
        # Lower distance is better, so invert
        avg_distance = total_distance / (len(progression) - 1)
        return 1.0 / (1.0 + avg_distance)
    
    def _evaluate_harmonic_tension(self, progression: List[Chord]) -> float:
        """Evaluate harmonic tension and resolution."""
        if len(progression) < 2:
            return 0.0
        
        # Simple tension model based on chord qualities and relationships
        tension_score = 0.0
        for i in range(len(progression) - 1):
            current = progression[i]
            next_chord = progression[i + 1]
            
            # Dominant to tonic resolution
            if self._is_dominant_relationship(current, next_chord):
                tension_score += 1.0
            
            # Minor to major resolution
            if current.quality == 'minor' and next_chord.quality == 'major':
                tension_score += 0.5
        
        return tension_score / max(1, len(progression) - 1)
    
    def _evaluate_neo_riemannian_efficiency(self, progression: List[Chord]) -> float:
        """Evaluate how efficiently the progression uses PLR transformations."""
        if len(progression) < 2:
            return 0.0
        
        plr_moves = 0
        for i in range(len(progression) - 1):
            current = progression[i]
            next_chord = progression[i + 1]
            
            # Check if transition is a single PLR transformation
            if (self.plr_group.parallel(current) == next_chord or
                self.plr_group.leading_tone_exchange(current) == next_chord or
                self.plr_group.relative(current) == next_chord):
                plr_moves += 1
        
        return plr_moves / (len(progression) - 1)
    
    def _evaluate_chord_diversity(self, progression: List[Chord]) -> float:
        """Evaluate diversity of chords in the progression."""
        unique_chords = set((c.root, c.quality) for c in progression)
        return len(unique_chords) / len(progression)
    
    def _evaluate_tonal_stability(self, progression: List[Chord]) -> float:
        """Evaluate tonal stability of the progression."""
        # Simple model: presence of tonic, dominant, subdominant
        chord_functions = set()
        for chord in progression:
            root = chord.root
            if root == 0:  # C
                chord_functions.add('tonic')
            elif root == 7:  # G
                chord_functions.add('dominant')
            elif root == 5:  # F
                chord_functions.add('subdominant')
        
        return len(chord_functions) / 3.0  # Normalize to [0, 1]
    
    def _heuristic(self, progression: List[Chord], target_length: int) -> float:
        """Heuristic function for search algorithms."""
        remaining_length = target_length - len(progression)
        if remaining_length <= 0:
            return 0.0
        
        # Simple heuristic: estimate potential quality of remaining progression
        return remaining_length * 0.5
    
    def _is_dominant_relationship(self, chord1: Chord, chord2: Chord) -> bool:
        """Check if chord1 is dominant of chord2."""
        return (chord1.root + 5) % 12 == chord2.root
    
    def _random_progression(self, start_chord: Chord, length: int) -> List[Chord]:
        """Generate a random progression for genetic algorithm initialization."""
        progression = [start_chord]
        current = start_chord
        
        for _ in range(length - 1):
            candidates = self._generate_candidates(current)
            current = random.choice(candidates)
            progression.append(current)
        
        return progression
    
    def _tournament_selection(self, population: List[List[Chord]], 
                            fitness_scores: List[float], tournament_size: int = 3) -> List[Chord]:
        """Tournament selection for genetic algorithm."""
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx]
    
    def _crossover(self, parent1: List[Chord], parent2: List[Chord]) -> List[Chord]:
        """Crossover operation for genetic algorithm."""
        if len(parent1) != len(parent2):
            return parent1  # Fallback
        
        crossover_point = random.randint(1, len(parent1) - 1)
        return parent1[:crossover_point] + parent2[crossover_point:]
    
    def _mutate(self, progression: List[Chord], start_chord: Chord) -> List[Chord]:
        """Mutation operation for genetic algorithm."""
        if len(progression) <= 1:
            return progression
        
        mutation_point = random.randint(1, len(progression) - 1)
        candidates = self._generate_candidates(progression[mutation_point - 1])
        
        mutated = progression.copy()
        mutated[mutation_point] = random.choice(candidates)
        return mutated
    
    def set_parameters(self, **params):
        """Set algorithm parameters."""
        if 'search_method' in params:
            self.search_method = params['search_method']
        
        if 'beam_width' in params:
            self.beam_width = int(params['beam_width'])
        
        if 'max_iterations' in params:
            self.max_iterations = int(params['max_iterations'])
        
        if 'objectives' in params:
            obj_names = params['objectives']
            self.objectives = [SearchObjective(name) for name in obj_names]
        
        if 'objective_weights' in params:
            self.objective_weights = list(params['objective_weights'])
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the algorithm."""
        return {
            'algorithm': 'search_based',
            'search_method': self.search_method,
            'objectives': [obj.value for obj in self.objectives],
            'objective_weights': self.objective_weights,
            'beam_width': self.beam_width,
            'max_iterations': self.max_iterations
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the current parameters of the algorithm."""
        return {
            'search_method': self.search_method,
            'objectives': [obj.value for obj in self.objectives],
            'objective_weights': self.objective_weights,
            'beam_width': self.beam_width,
            'max_iterations': self.max_iterations,
            'mutation_rate': self.mutation_rate,
            'population_size': self.population_size
        }
