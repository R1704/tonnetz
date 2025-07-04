"""
Tests for new progression algorithms.
"""

import unittest
from tonnetz.core.chord import Chord
from tonnetz.progression.markov import MarkovProgression
from tonnetz.progression.search_based import SearchBasedProgression, SearchObjective


class TestMarkovProgression(unittest.TestCase):
    """Test the Markov chain progression algorithm."""
    
    def test_markov_creation(self):
        """Test creating a Markov progression algorithm."""
        algo = MarkovProgression(order=1, auto_train=False)
        assert algo.order == 1
        assert not algo._trained
    
    def test_markov_training(self):
        """Test training a Markov progression."""
        algo = MarkovProgression(order=1)
        progressions = [
            ['C', 'Am', 'F', 'G'],
            ['C', 'F', 'G', 'C'],
        ]
        algo.train(progressions)
        assert algo._trained
        assert len(algo.transition_matrix) > 0
    
    def test_markov_generation(self):
        """Test generating a progression with Markov algorithm."""
        algo = MarkovProgression(order=1)
        start_chord = Chord(0, 'major')  # C major
        progression = algo.generate_progression(start_chord, 4)
        
        assert len(progression) == 4
        assert progression[0] == start_chord
        assert all(isinstance(chord, Chord) for chord in progression)
    
    def test_markov_parameters(self):
        """Test setting Markov algorithm parameters."""
        algo = MarkovProgression()
        algo.set_parameters(order=2)
        assert algo.order == 2


class TestSearchBasedProgression(unittest.TestCase):
    """Test the search-based progression algorithm."""
    
    def test_search_creation(self):
        """Test creating a search-based progression algorithm."""
        algo = SearchBasedProgression(search_method='beam')
        assert algo.search_method == 'beam'
        assert algo.beam_width == 10
    
    def test_beam_search_generation(self):
        """Test generating a progression with beam search."""
        algo = SearchBasedProgression(
            search_method='beam',
            beam_width=5,
            objectives=[SearchObjective.SMOOTH_VOICE_LEADING]
        )
        start_chord = Chord(0, 'major')  # C major
        progression = algo.generate_progression(start_chord, 4)
        
        assert len(progression) == 4
        assert progression[0] == start_chord
        assert all(isinstance(chord, Chord) for chord in progression)
    
    def test_genetic_algorithm_generation(self):
        """Test generating a progression with genetic algorithm."""
        algo = SearchBasedProgression(
            search_method='genetic',
            population_size=10,
            max_iterations=50
        )
        start_chord = Chord(0, 'major')  # C major
        progression = algo.generate_progression(start_chord, 4)
        
        assert len(progression) == 4
        assert progression[0] == start_chord
        assert all(isinstance(chord, Chord) for chord in progression)
    
    def test_search_objectives(self):
        """Test different search objectives."""
        for objective in SearchObjective:
            algo = SearchBasedProgression(
                search_method='beam',
                beam_width=3,
                objectives=[objective],
                objective_weights=[1.0]
            )
            start_chord = Chord(0, 'major')
            progression = algo.generate_progression(start_chord, 3)
            
            assert len(progression) == 3
            assert progression[0] == start_chord
    
    def test_search_parameters(self):
        """Test setting search algorithm parameters."""
        algo = SearchBasedProgression()
        algo.set_parameters(
            search_method='astar',
            beam_width=15,
            max_iterations=500
        )
        assert algo.search_method == 'astar'
        assert algo.beam_width == 15
        assert algo.max_iterations == 500
    
    def test_voice_leading_evaluation(self):
        """Test voice leading evaluation."""
        algo = SearchBasedProgression()
        progression = [
            Chord(0, 'major'),   # C
            Chord(9, 'minor'),   # Am
            Chord(5, 'major'),   # F
            Chord(7, 'major')    # G
        ]
        score = algo._evaluate_voice_leading(progression)
        assert 0.0 <= score <= 1.0
    
    def test_candidate_generation(self):
        """Test candidate chord generation."""
        algo = SearchBasedProgression()
        start_chord = Chord(0, 'major')  # C major
        candidates = algo._generate_candidates(start_chord)
        
        assert len(candidates) > 0
        assert all(isinstance(chord, Chord) for chord in candidates)
        # Should include PLR transformations
        assert any(chord.quality == 'minor' for chord in candidates)


class TestProgressionIntegration(unittest.TestCase):
    """Integration tests for progression algorithms."""
    
    def test_algorithm_comparison(self):
        """Test generating progressions with different algorithms."""
        start_chord = Chord(0, 'major')  # C major
        length = 4
        
        # Rule-based
        from tonnetz.progression.rule_based import RuleBasedProgression
        rule_algo = RuleBasedProgression(pattern='I-vi-IV-V')
        rule_progression = rule_algo.generate(start_chord, length)
        
        # Markov
        markov_algo = MarkovProgression()
        markov_progression = markov_algo.generate(start_chord, length)
        
        # Search-based
        search_algo = SearchBasedProgression(search_method='beam', beam_width=3)
        search_progression = search_algo.generate(start_chord, length)
        
        # All should produce valid progressions
        for progression in [rule_progression, markov_progression, search_progression]:
            assert len(progression) == length
            assert progression[0] == start_chord
            assert all(isinstance(chord, Chord) for chord in progression)
    
    def test_algorithm_state_reporting(self):
        """Test that algorithms report their state correctly."""
        markov_algo = MarkovProgression(order=2)
        markov_state = markov_algo.get_state()
        assert markov_state['algorithm'] == 'markov'
        assert markov_state['order'] == 2
        
        search_algo = SearchBasedProgression(search_method='genetic')
        search_state = search_algo.get_state()
        assert search_state['algorithm'] == 'search_based'
        assert search_state['search_method'] == 'genetic'


if __name__ == '__main__':
    unittest.main()
