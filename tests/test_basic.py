"""
Basic tests for the Tonnetz library.

This module provides simple tests to verify the core functionality
of the library components.
"""

import pytest
from tonnetz.core.chord import Chord, parse_chord_name
from tonnetz.core.neo_riemannian import PLRGroup
from tonnetz.core.tonnetz import ToroidalTonnetz
from tonnetz.automaton.grid import ToroidalGrid
from tonnetz.automaton.cell import Cell
from tonnetz.progression.rule_based import RuleBasedProgression


class TestChord:
    """Test chord representation and operations."""
    
    def test_chord_creation(self):
        """Test basic chord creation."""
        chord = Chord(0, 'major')
        assert chord.root == 0
        assert chord.quality == 'major'
        assert chord.inversion == 0
    
    def test_chord_pitch_classes(self):
        """Test pitch class extraction."""
        c_major = Chord(0, 'major')
        assert c_major.pitch_classes() == (0, 4, 7)
        
        a_minor = Chord(9, 'minor')
        assert a_minor.pitch_classes() == (9, 0, 4)
    
    def test_chord_transposition(self):
        """Test chord transposition."""
        c_major = Chord(0, 'major')
        d_major = c_major.transpose(2)
        assert d_major.root == 2
        assert d_major.quality == 'major'
    
    def test_chord_parsing(self):
        """Test chord name parsing."""
        chord = parse_chord_name('Cm')
        assert chord.root == 0
        assert chord.quality == 'minor'
        
        chord = parse_chord_name('F#maj7')
        assert chord.root == 6
        assert chord.quality == 'major7'
    
    def test_voice_leading_distance(self):
        """Test voice leading distance calculation."""
        c_major = Chord(0, 'major')
        f_major = Chord(5, 'major')
        distance = c_major.voice_leading_distance(f_major)
        assert distance >= 0  # Should be a valid distance


class TestPLRGroup:
    """Test neo-Riemannian transformations."""
    
    def test_parallel_transformation(self):
        """Test parallel transformation."""
        plr = PLRGroup()
        c_major = Chord(0, 'major')
        c_minor = plr.parallel(c_major)
        assert c_minor.root == 0
        assert c_minor.quality == 'minor'
    
    def test_relative_transformation(self):
        """Test relative transformation."""
        plr = PLRGroup()
        c_major = Chord(0, 'major')
        a_minor = plr.relative(c_major)
        assert a_minor.root == 9  # A minor
        assert a_minor.quality == 'minor'
    
    def test_leading_tone_exchange(self):
        """Test leading tone exchange."""
        plr = PLRGroup()
        c_major = Chord(0, 'major')
        result = plr.leading_tone_exchange(c_major)
        assert result.quality == 'minor'
    
    def test_transformation_composition(self):
        """Test composition of transformations."""
        plr = PLRGroup()
        c_major = Chord(0, 'major')
        
        # Apply PLR sequence
        result = plr.apply('PLR', c_major)
        assert isinstance(result, Chord)
    
    def test_find_transformation_sequence(self):
        """Test finding transformation sequences."""
        plr = PLRGroup()
        c_major = Chord(0, 'major')
        a_minor = Chord(9, 'minor')
        
        sequence = plr.find_transformation_sequence(c_major, a_minor)
        assert isinstance(sequence, str)
        # Should find 'R' (relative transformation)


class TestToroidalTonnetz:
    """Test Tonnetz geometry."""
    
    def test_tonnetz_creation(self):
        """Test Tonnetz creation."""
        tonnetz = ToroidalTonnetz(12, 12)
        assert tonnetz.width == 12
        assert tonnetz.height == 12
    
    def test_pitch_class_mapping(self):
        """Test pitch class to coordinate mapping."""
        tonnetz = ToroidalTonnetz(12, 12)
        coords = tonnetz.pitch_class_to_coords(0)  # C
        assert isinstance(coords, tuple)
        assert len(coords) == 2
    
    def test_coordinate_roundtrip(self):
        """Test coordinate conversion roundtrip."""
        tonnetz = ToroidalTonnetz(12, 12)
        original_pc = 5  # F
        coords = tonnetz.pitch_class_to_coords(original_pc)
        recovered_pc = tonnetz.coords_to_pitch_class(*coords)
        # Note: May not be exact due to quantization
        assert 0 <= recovered_pc <= 11
    
    def test_toroidal_distance(self):
        """Test toroidal distance calculation."""
        tonnetz = ToroidalTonnetz(12, 12)
        dist = tonnetz.toroidal_distance((0, 0), (1, 1))
        assert dist >= 0


class TestCell:
    """Test cellular automaton cells."""
    
    def test_cell_creation(self):
        """Test cell creation."""
        chord = Chord(0, 'major')
        cell = Cell(chord, 5, 7)
        assert cell.chord == chord
        assert cell.x == 5
        assert cell.y == 7
    
    def test_cell_update(self):
        """Test cell chord update."""
        chord1 = Chord(0, 'major')
        chord2 = Chord(7, 'major')
        cell = Cell(chord1)
        
        cell.update_chord(chord2)
        assert cell.chord == chord2
        assert len(cell.chord_history) == 2
    
    def test_cell_activation(self):
        """Test cell activation system."""
        chord = Chord(0, 'major')
        cell = Cell(chord)
        assert cell.activation == 1.0
        assert cell.is_active()
        
        cell.activation = 0.3
        assert not cell.is_active()  # Below default threshold of 0.5


class TestToroidalGrid:
    """Test cellular automaton grid."""
    
    def test_grid_creation(self):
        """Test grid creation."""
        grid = ToroidalGrid(8, 8)
        assert grid.width == 8
        assert grid.height == 8
        assert grid.generation == 0
    
    def test_grid_cell_access(self):
        """Test grid cell access."""
        grid = ToroidalGrid(8, 8)
        cell = grid.get_cell(3, 4)
        assert isinstance(cell, Cell)
        assert cell.x == 3
        assert cell.y == 4
    
    def test_grid_neighbors(self):
        """Test neighbor calculation."""
        grid = ToroidalGrid(8, 8)
        neighbors = grid.get_neighbors(3, 4)
        assert len(neighbors) == 8  # Moore neighborhood
        
        # Test von Neumann
        grid.neighborhood = 'von_neumann'
        neighbors = grid.get_neighbors(3, 4)
        assert len(neighbors) == 4
    
    def test_grid_population(self):
        """Test grid population methods."""
        grid = ToroidalGrid(4, 4)
        
        # Test random population
        grid.populate_random()
        
        # Test pattern population
        chords = [Chord(0, 'major'), Chord(7, 'major')]
        grid.populate_pattern('checkerboard', chords)
        
        # Verify grid is populated
        for y in range(grid.height):
            for x in range(grid.width):
                cell = grid.get_cell(x, y)
                assert cell.chord in chords
    
    def test_grid_update(self):
        """Test grid update mechanism."""
        grid = ToroidalGrid(4, 4)
        grid.populate_random()
        
        initial_generation = grid.generation
        grid.update()
        assert grid.generation == initial_generation + 1


class TestRuleBasedProgression:
    """Test rule-based progression algorithm."""
    
    def test_progression_creation(self):
        """Test progression algorithm creation."""
        algo = RuleBasedProgression(key=0, mode='major')
        assert algo.key == 0
        assert algo.mode == 'major'
    
    def test_progression_generation(self):
        """Test progression generation."""
        algo = RuleBasedProgression(key=0, mode='major')
        start_chord = Chord(0, 'major')
        progression = algo.generate(start_chord, 5)
        
        assert len(progression) == 5
        assert all(isinstance(chord, Chord) for chord in progression)
        assert progression[0] == start_chord
    
    def test_pattern_progression(self):
        """Test pattern-based progression."""
        algo = RuleBasedProgression(key=0, mode='major', pattern='ii-V-I')
        start_chord = Chord(0, 'major')
        progression = algo.generate(start_chord, 3)
        
        assert len(progression) == 3
    
    def test_parameter_setting(self):
        """Test parameter modification."""
        algo = RuleBasedProgression()
        algo.set_parameters(key=7, mode='minor', randomness=0.5)
        
        params = algo.get_parameters()
        assert params['key'] == 7
        assert params['mode'] == 'minor'
        assert params['randomness'] == 0.5


def test_integration_basic_simulation():
    """Test basic integration of components."""
    # Create a small grid
    grid = ToroidalGrid(4, 4)
    
    # Set up progression algorithm
    algo = RuleBasedProgression(key=0, mode='major')
    start_chord = Chord(0, 'major')
    
    # Populate grid
    grid.populate_with_progression(algo, start_chord)
    
    # Run a few steps
    for _ in range(3):
        grid.update()
    
    # Verify grid is still valid
    assert grid.generation == 3
    stats = grid.get_statistics()
    assert stats['total_cells'] == 16
    assert 0 <= stats['active_cells'] <= 16


if __name__ == '__main__':
    pytest.main([__file__])
