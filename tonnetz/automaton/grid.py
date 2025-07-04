"""
Toroidal grid for the cellular automaton.

This module implements the 2D toroidal grid that serves as the spatial
foundation for the cellular automaton chord evolution system.
"""

import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from tonnetz.automaton.cell import Cell
from tonnetz.core.chord import Chord
from tonnetz.core.tonnetz import ToroidalTonnetz
from tonnetz.progression.base import ProgressionAlgo


class ToroidalGrid:
    """
    2D toroidal grid for cellular automaton chord evolution.

    This grid manages a collection of cells arranged on a torus, providing
    neighbor lookup, update scheduling, and spatial operations.
    """

    def __init__(
        self,
        width: int,
        height: int,
        neighborhood: str = "moore",
        update_mode: str = "synchronous",
    ) -> None:
        """
        Initialize the toroidal grid.

        Args:
            width: Grid width
            height: Grid height
            neighborhood: Neighborhood type ('moore' or 'von_neumann')
            update_mode: Update scheduling ('synchronous' or 'asynchronous')
        """
        self.width = width
        self.height = height
        self.neighborhood = neighborhood
        self.update_mode = update_mode

        # Initialize grid with empty cells
        self.grid: List[List[Cell]] = []
        self._initialize_grid()

        # Tonnetz for coordinate mapping
        self.tonnetz = ToroidalTonnetz(width, height)

        # Evolution tracking
        self.generation = 0
        self.history: List[List[List[Cell]]] = []
        self.max_history = 20

        # Statistics
        self.stats = {
            "chord_changes": 0,
            "active_cells": 0,
            "average_activation": 0.0,
            "harmony_level": 0.0,
        }

    def _initialize_grid(self) -> None:
        """Initialize the grid with empty cells."""
        self.grid = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                # Create cell with default C major chord
                cell = Cell(Chord(0, "major"), x, y)
                row.append(cell)
            self.grid.append(row)

    def get_cell(self, x: int, y: int) -> Cell:
        """
        Get cell at specified coordinates.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Cell at the specified position
        """
        # Handle toroidal wrap-around
        x = x % self.width
        y = y % self.height
        return self.grid[y][x]

    def set_cell(self, x: int, y: int, cell: Cell) -> None:
        """
        Set cell at specified coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            cell: Cell to place at the position
        """
        x = x % self.width
        y = y % self.height
        cell.x = x
        cell.y = y
        self.grid[y][x] = cell

    def get_neighbors(self, x: int, y: int) -> List[Cell]:
        """
        Get neighboring cells for a given position.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            List of neighboring cells
        """
        neighbors = []

        if self.neighborhood == "moore":
            # 8-connected neighborhood
            offsets = [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
        elif self.neighborhood == "von_neumann":
            # 4-connected neighborhood
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            raise ValueError(f"Unknown neighborhood type: {self.neighborhood}")

        for dx, dy in offsets:
            neighbor_x = (x + dx) % self.width
            neighbor_y = (y + dy) % self.height
            neighbors.append(self.grid[neighbor_y][neighbor_x])

        return neighbors

    def get_neighbor_distances(self, x: int, y: int) -> List[float]:
        """
        Get distances to neighboring cells.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            List of distances to each neighbor
        """
        distances = []

        if self.neighborhood == "moore":
            # Moore neighborhood distances
            distances = [
                1.414,
                1.0,
                1.414,
                1.0,
                1.0,
                1.414,
                1.0,
                1.414,
            ]  # sqrt(2) for diagonals
        elif self.neighborhood == "von_neumann":
            # Von Neumann neighborhood distances
            distances = [1.0, 1.0, 1.0, 1.0]

        return distances

    def populate_random(
        self, chord_generator: Optional[Callable[[], Chord]] = None
    ) -> None:
        """
        Populate the grid with random chords.

        Args:
            chord_generator: Optional function to generate chords
        """
        if chord_generator is None:
            # Default random chord generator
            def default_generator() -> Chord:
                root = random.randint(0, 11)
                quality = random.choice(["major", "minor"])
                return Chord(root, quality)

            chord_generator = default_generator

        for y in range(self.height):
            for x in range(self.width):
                chord = chord_generator()
                self.grid[y][x] = Cell(chord, x, y)

    def populate_with_progression(
        self, progression_algo: ProgressionAlgo, start_chord: Optional[Chord] = None
    ) -> None:
        """
        Populate the grid using a progression algorithm.

        Args:
            progression_algo: Algorithm to generate chord progressions
            start_chord: Optional starting chord
        """
        if start_chord is None:
            start_chord = Chord(0, "major")

        # Generate a progression for the entire grid
        total_cells = self.width * self.height
        progression = progression_algo.generate(start_chord, total_cells)

        # Fill grid with progression
        chord_index = 0
        for y in range(self.height):
            for x in range(self.width):
                if chord_index < len(progression):
                    chord = progression[chord_index]
                else:
                    chord = start_chord  # Fallback

                self.grid[y][x] = Cell(chord, x, y)
                chord_index += 1

    def populate_pattern(self, pattern: str, chords: List[Chord]) -> None:
        """
        Populate the grid with a specific pattern.

        Args:
            pattern: Pattern type ('checkerboard', 'stripes', 'center', 'random')
            chords: List of chords to use in the pattern
        """
        if not chords:
            chords = [Chord(0, "major")]

        for y in range(self.height):
            for x in range(self.width):
                if pattern == "checkerboard":
                    chord_index = (x + y) % len(chords)
                elif pattern == "stripes":
                    chord_index = x % len(chords)
                elif pattern == "center":
                    # Distance from center
                    center_x, center_y = self.width // 2, self.height // 2
                    distance = abs(x - center_x) + abs(y - center_y)
                    chord_index = distance % len(chords)
                elif pattern == "random":
                    chord_index = random.randint(0, len(chords) - 1)
                else:
                    chord_index = 0

                chord = chords[chord_index]
                self.grid[y][x] = Cell(chord, x, y)

    def apply_rule(self, rule_func: Callable[[Cell, List[Cell]], Chord]) -> None:
        """
        Apply a transformation rule to all cells.

        Args:
            rule_func: Function that takes (cell, neighbors) and returns new chord
        """
        if self.update_mode == "synchronous":
            self._apply_rule_synchronous(rule_func)
        elif self.update_mode == "asynchronous":
            self._apply_rule_asynchronous(rule_func)
        else:
            raise ValueError(f"Unknown update mode: {self.update_mode}")

        self.generation += 1
        self._update_statistics()

    def _apply_rule_synchronous(
        self, rule_func: Callable[[Cell, List[Cell]], Chord]
    ) -> None:
        """Apply rule synchronously (all cells update simultaneously)."""
        # Prepare all updates first
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                neighbors = self.get_neighbors(x, y)
                new_chord = rule_func(cell, neighbors)
                cell.prepare_update(new_chord)

        # Commit all updates
        changes = 0
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                old_chord = cell.chord
                cell.commit_update()
                if cell.chord != old_chord:
                    changes += 1

        self.stats["chord_changes"] = changes

    def _apply_rule_asynchronous(
        self, rule_func: Callable[[Cell, List[Cell]], Chord]
    ) -> None:
        """Apply rule asynchronously (cells update in random order)."""
        # Create random update order
        positions = [(x, y) for x in range(self.width) for y in range(self.height)]
        random.shuffle(positions)

        changes = 0
        for x, y in positions:
            cell = self.grid[y][x]
            neighbors = self.get_neighbors(x, y)
            new_chord = rule_func(cell, neighbors)

            if new_chord != cell.chord:
                cell.update_chord(new_chord)
                changes += 1

        self.stats["chord_changes"] = changes

    def update(
        self, rule_func: Optional[Callable[[Cell, List[Cell]], Chord]] = None
    ) -> None:
        """
        Perform one update step of the cellular automaton.

        Args:
            rule_func: Optional rule function to apply
        """
        if rule_func is None:
            # Default rule: majority vote
            def majority_rule(cell: Cell, neighbors: List[Cell]) -> Chord:
                if not neighbors:
                    return cell.chord

                # Count chord occurrences
                chord_counts = {}
                for neighbor in neighbors:
                    chord = neighbor.chord
                    chord_counts[chord] = chord_counts.get(chord, 0) + 1

                # Return most common chord, or current if tie
                max_count = max(chord_counts.values())
                most_common = [
                    chord for chord, count in chord_counts.items() if count == max_count
                ]

                if cell.chord in most_common:
                    return cell.chord
                else:
                    return most_common[0]

            rule_func = majority_rule

        # Save current state to history
        self._save_state_to_history()

        # Apply the rule
        self.apply_rule(rule_func)

    def _save_state_to_history(self) -> None:
        """Save current grid state to history."""
        # Create deep copy of current state
        state_copy = []
        for row in self.grid:
            row_copy = [cell.copy() for cell in row]
            state_copy.append(row_copy)

        self.history.append(state_copy)

        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def _update_statistics(self) -> None:
        """Update grid statistics."""
        active_count = 0
        total_activation = 0.0

        for row in self.grid:
            for cell in row:
                if cell.is_active():
                    active_count += 1
                total_activation += cell.activation

        total_cells = self.width * self.height
        self.stats["active_cells"] = active_count
        self.stats["average_activation"] = total_activation / total_cells

        # Calculate overall harmony level
        harmony_sum = 0.0
        cell_count = 0

        for y in range(self.height):
            for x in range(self.width):
                neighbors = self.get_neighbors(x, y)
                from tonnetz.automaton.cell import CellNeighborhood

                harmony = CellNeighborhood.calculate_harmony_level(neighbors)
                harmony_sum += harmony
                cell_count += 1

        self.stats["harmony_level"] = (
            harmony_sum / cell_count if cell_count > 0 else 0.0
        )

    def get_state_matrix(self) -> np.ndarray:
        """
        Get the current state as a matrix of chord roots.

        Returns:
            2D numpy array with chord roots
        """
        matrix = np.zeros((self.height, self.width), dtype=int)
        for y in range(self.height):
            for x in range(self.width):
                matrix[y, x] = self.grid[y][x].chord.root
        return matrix

    def get_quality_matrix(self) -> np.ndarray:
        """
        Get matrix representing chord qualities.

        Returns:
            2D numpy array with quality indices (0=major, 1=minor, etc.)
        """
        quality_map = {"major": 0, "minor": 1, "diminished": 2, "augmented": 3}
        matrix = np.zeros((self.height, self.width), dtype=int)

        for y in range(self.height):
            for x in range(self.width):
                quality = self.grid[y][x].chord.quality
                matrix[y, x] = quality_map.get(quality, 0)

        return matrix

    def get_activation_matrix(self) -> np.ndarray:
        """
        Get matrix of cell activation levels.

        Returns:
            2D numpy array with activation values
        """
        matrix = np.zeros((self.height, self.width), dtype=float)
        for y in range(self.height):
            for x in range(self.width):
                matrix[y, x] = self.grid[y][x].activation
        return matrix

    def find_patterns(self, pattern_size: int = 3) -> List[Dict[str, Any]]:
        """
        Find recurring patterns in the grid.

        Args:
            pattern_size: Size of patterns to look for

        Returns:
            List of found patterns with their locations
        """
        patterns = []
        state_matrix = self.get_state_matrix()

        # Simple pattern detection for now
        for y in range(self.height - pattern_size + 1):
            for x in range(self.width - pattern_size + 1):
                # Extract pattern
                pattern = state_matrix[y : y + pattern_size, x : x + pattern_size]

                # Look for identical patterns elsewhere
                for y2 in range(y + 1, self.height - pattern_size + 1):
                    for x2 in range(self.width - pattern_size + 1):
                        if y2 == y and x2 <= x:
                            continue

                        other_pattern = state_matrix[
                            y2 : y2 + pattern_size, x2 : x2 + pattern_size
                        ]
                        if np.array_equal(pattern, other_pattern):
                            patterns.append(
                                {
                                    "pattern": pattern.tolist(),
                                    "locations": [(x, y), (x2, y2)],
                                    "size": pattern_size,
                                }
                            )

        return patterns

    def reset(self) -> None:
        """Reset the grid to initial state."""
        self._initialize_grid()
        self.generation = 0
        self.history.clear()
        self.stats = {
            "chord_changes": 0,
            "active_cells": 0,
            "average_activation": 0.0,
            "harmony_level": 0.0,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current grid statistics.

        Returns:
            Dictionary of statistics
        """
        stats = self.stats.copy()
        stats["generation"] = self.generation
        stats["total_cells"] = self.width * self.height
        stats["grid_size"] = f"{self.width}x{self.height}"
        return stats

    def export_state(self) -> Dict[str, Any]:
        """
        Export current grid state to dictionary.

        Returns:
            Dictionary representation of the grid state
        """
        state = {
            "width": self.width,
            "height": self.height,
            "generation": self.generation,
            "neighborhood": self.neighborhood,
            "update_mode": self.update_mode,
            "cells": [],
        }

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                cell_data = {
                    "x": x,
                    "y": y,
                    "chord": cell.chord.to_dict(),
                    "activation": cell.activation,
                    "energy": cell.energy,
                    "state": cell.state,
                }
                state["cells"].append(cell_data)

        return state

    def import_state(self, state: Dict[str, Any]) -> None:
        """
        Import grid state from dictionary.

        Args:
            state: Dictionary containing grid state
        """
        self.width = state["width"]
        self.height = state["height"]
        self.generation = state["generation"]
        self.neighborhood = state.get("neighborhood", "moore")
        self.update_mode = state.get("update_mode", "synchronous")

        # Reinitialize grid
        self._initialize_grid()

        # Load cells
        for cell_data in state["cells"]:
            x, y = cell_data["x"], cell_data["y"]
            chord = Chord.from_dict(cell_data["chord"])
            cell = Cell(chord, x, y, cell_data.get("state", {}))
            cell.activation = cell_data.get("activation", 1.0)
            cell.energy = cell_data.get("energy", 1.0)
            self.set_cell(x, y, cell)

        self._update_statistics()
