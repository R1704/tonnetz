"""
Toroidal Tonnetz geometry and lattice operations.

This module implements the geometric structure of the Tonnetz as a toroidal lattice,
providing coordinate mapping, distance calculations, and neighbor operations.
"""

from typing import List, Optional, Tuple

import numpy as np

from tonnetz.core.chord import Chord


class ToroidalTonnetz:
    """
    Represents the Tonnetz as a toroidal lattice with customizable dimensions.

    The Tonnetz maps pitch classes and chords onto a 2D torus where:
    - One axis typically represents perfect fifths
    - The other axis represents major thirds
    - The torus allows for continuous harmonic space without boundaries
    """

    def __init__(
        self,
        width: int = 12,
        height: int = 12,
        fifth_axis: str = "x",
        third_axis: str = "y",
    ) -> None:
        """
        Initialize the toroidal Tonnetz.

        Args:
            width: Width of the torus (typically 12 for pitch class symmetry)
            height: Height of the torus (typically 12 for pitch class symmetry)
            fifth_axis: Which axis represents perfect fifths ('x' or 'y')
            third_axis: Which axis represents major thirds ('x' or 'y')
        """
        self.width = width
        self.height = height
        self.fifth_axis = fifth_axis
        self.third_axis = third_axis

        # Validate axis configuration
        if fifth_axis == third_axis:
            raise ValueError("Fifth and third axes must be different")
        if fifth_axis not in ["x", "y"] or third_axis not in ["x", "y"]:
            raise ValueError("Axes must be 'x' or 'y'")

        # Basis vectors for the lattice
        # Perfect fifth = 7 semitones, major third = 4 semitones
        if fifth_axis == "x":
            self.fifth_vector = np.array([1, 0])
            self.third_vector = np.array([0, 1])
        else:
            self.fifth_vector = np.array([0, 1])
            self.third_vector = np.array([1, 0])

    def pitch_class_to_coords(self, pitch_class: int) -> Tuple[int, int]:
        """
        Convert a pitch class to Tonnetz coordinates.

        Maps pitch classes to lattice coordinates based on the circle of fifths
        and major third relationships.

        Args:
            pitch_class: Pitch class (0-11, where C=0)

        Returns:
            Tuple of (x, y) coordinates on the torus
        """
        # Map pitch class to position in circle of fifths
        # C=0 is at origin, each step of 7 semitones moves +1 in fifth direction
        fifth_steps = (pitch_class * 7) % 12

        # Map to major third cycle
        # Each step of 4 semitones moves +1 in third direction
        third_steps = (pitch_class * 4) % 12

        # Convert to grid coordinates with modular arithmetic
        if self.fifth_axis == "x":
            x = (fifth_steps * self.width // 12) % self.width
            y = (third_steps * self.height // 12) % self.height
        else:
            x = (third_steps * self.width // 12) % self.width
            y = (fifth_steps * self.height // 12) % self.height

        return (x, y)

    def coords_to_pitch_class(self, x: int, y: int) -> int:
        """
        Convert Tonnetz coordinates back to a pitch class.

        Args:
            x: X coordinate on the torus
            y: Y coordinate on the torus

        Returns:
            Pitch class (0-11) corresponding to the coordinates
        """
        # Normalize coordinates to torus
        x = x % self.width
        y = y % self.height

        # Convert back to pitch class
        if self.fifth_axis == "x":
            fifth_component = (x * 12 // self.width) * 7
            third_component = (y * 12 // self.height) * 4
        else:
            fifth_component = (y * 12 // self.height) * 7
            third_component = (x * 12 // self.width) * 4

        return (fifth_component + third_component) % 12

    def chord_to_coords(self, chord: Chord) -> Tuple[int, int]:
        """
        Convert a chord to its representative coordinates on the Tonnetz.

        Uses the chord's root as the primary coordinate. More sophisticated
        mappings could consider the entire chord structure.

        Args:
            chord: Chord to map to coordinates

        Returns:
            Tuple of (x, y) coordinates representing the chord
        """
        return self.pitch_class_to_coords(chord.root)

    def coords_to_chord(self, x: int, y: int, quality: str = "major") -> Chord:
        """
        Create a chord from Tonnetz coordinates.

        Args:
            x: X coordinate on the torus
            y: Y coordinate on the torus
            quality: Chord quality to create

        Returns:
            Chord object with root at the specified coordinates
        """
        root = self.coords_to_pitch_class(x, y)
        return Chord(root, quality)

    def toroidal_distance(
        self,
        coord1: Tuple[int, int],
        coord2: Tuple[int, int],
        metric: str = "manhattan",
    ) -> float:
        """
        Calculate distance between two points on the torus.

        Args:
            coord1: First coordinate pair (x1, y1)
            coord2: Second coordinate pair (x2, y2)
            metric: Distance metric ('manhattan', 'euclidean', 'chebyshev')

        Returns:
            Distance between the points accounting for toroidal wrap-around
        """
        x1, y1 = coord1
        x2, y2 = coord2

        # Calculate minimal distance accounting for wrap-around
        dx = min(abs(x2 - x1), self.width - abs(x2 - x1))
        dy = min(abs(y2 - y1), self.height - abs(y2 - y1))

        if metric == "manhattan":
            return dx + dy
        elif metric == "euclidean":
            return np.sqrt(dx**2 + dy**2)
        elif metric == "chebyshev":
            return max(dx, dy)
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

    def chord_distance(
        self, chord1: Chord, chord2: Chord, metric: str = "manhattan"
    ) -> float:
        """
        Calculate geometric distance between two chords on the Tonnetz.

        Args:
            chord1: First chord
            chord2: Second chord
            metric: Distance metric to use

        Returns:
            Geometric distance between the chords
        """
        coords1 = self.chord_to_coords(chord1)
        coords2 = self.chord_to_coords(chord2)
        return self.toroidal_distance(coords1, coords2, metric)

    def get_neighbors(
        self, x: int, y: int, neighborhood: str = "moore"
    ) -> List[Tuple[int, int]]:
        """
        Get neighboring coordinates on the torus.

        Args:
            x: X coordinate
            y: Y coordinate
            neighborhood: Type of neighborhood ('moore' for 8-connected, 'von_neumann' for 4-connected)

        Returns:
            List of neighboring coordinate pairs
        """
        neighbors = []

        if neighborhood == "moore":
            # 8-connected neighborhood (Moore)
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
        elif neighborhood == "von_neumann":
            # 4-connected neighborhood (von Neumann)
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            raise ValueError(f"Unknown neighborhood type: {neighborhood}")

        for dx, dy in offsets:
            new_x = (x + dx) % self.width
            new_y = (y + dy) % self.height
            neighbors.append((new_x, new_y))

        return neighbors

    def get_chord_neighbors(
        self, chord: Chord, neighborhood: str = "moore"
    ) -> List[Chord]:
        """
        Get neighboring chords on the Tonnetz.

        Args:
            chord: Central chord
            neighborhood: Type of neighborhood

        Returns:
            List of neighboring chords with the same quality
        """
        x, y = self.chord_to_coords(chord)
        neighbor_coords = self.get_neighbors(x, y, neighborhood)

        neighbors = []
        for nx, ny in neighbor_coords:
            neighbor_chord = self.coords_to_chord(nx, ny, chord.quality)
            neighbors.append(neighbor_chord)

        return neighbors

    def visualize_lattice(self, chords: Optional[List[Chord]] = None) -> np.ndarray:
        """
        Create a visual representation of the lattice.

        Args:
            chords: Optional list of chords to highlight on the lattice

        Returns:
            2D numpy array representing the lattice visualization
        """
        lattice = np.zeros((self.height, self.width), dtype=int)

        # Fill with pitch classes
        for y in range(self.height):
            for x in range(self.width):
                pitch_class = self.coords_to_pitch_class(x, y)
                lattice[y, x] = pitch_class

        # If specific chords provided, mark them
        if chords:
            marked_lattice = np.full((self.height, self.width), -1, dtype=int)
            for i, chord in enumerate(chords):
                x, y = self.chord_to_coords(chord)
                marked_lattice[y, x] = i
            return marked_lattice

        return lattice

    def get_harmonic_regions(self) -> dict[str, List[Tuple[int, int]]]:
        """
        Identify harmonic regions on the Tonnetz.

        Returns:
            Dictionary mapping region names to lists of coordinates
        """
        regions = {
            "major_triads": [],
            "minor_triads": [],
            "diminished_triads": [],
            "augmented_triads": [],
        }

        # This is a simplified implementation
        # More sophisticated analysis could identify actual triadic regions
        for y in range(self.height):
            for x in range(self.width):
                pitch_class = self.coords_to_pitch_class(x, y)

                # Simple heuristic: assign regions based on pitch class
                if pitch_class in [0, 2, 4, 5, 7, 9, 11]:  # White keys
                    regions["major_triads"].append((x, y))
                else:  # Black keys
                    regions["minor_triads"].append((x, y))

        return regions

    def shortest_path_coords(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Find shortest path between two coordinates on the torus.

        Args:
            start: Starting coordinates
            end: Ending coordinates

        Returns:
            List of coordinates forming the shortest path
        """
        x1, y1 = start
        x2, y2 = end

        # Calculate shortest direction for each axis considering wrap-around
        def shortest_direction(pos1: int, pos2: int, size: int) -> int:
            forward = (pos2 - pos1) % size
            backward = (pos1 - pos2) % size
            return forward if forward <= backward else -backward

        dx_total = shortest_direction(x1, x2, self.width)
        dy_total = shortest_direction(y1, y2, self.height)

        # Generate path using simple linear interpolation
        steps = max(abs(dx_total), abs(dy_total))
        if steps == 0:
            return [start]

        path = []
        for i in range(steps + 1):
            progress = i / steps
            x = (x1 + int(dx_total * progress)) % self.width
            y = (y1 + int(dy_total * progress)) % self.height
            path.append((x, y))

        return path

    def get_fundamental_domain(self) -> Tuple[int, int, int, int]:
        """
        Get the fundamental domain of the torus.

        Returns:
            Tuple of (min_x, max_x, min_y, max_y) defining the fundamental domain
        """
        return (0, self.width - 1, 0, self.height - 1)
