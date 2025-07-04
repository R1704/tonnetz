"""
Neo-Riemannian transformations and PLR group operations.

This module implements the PLR (Parallel, Leittonwechsel, Relative) group
of transformations that form the foundation of neo-Riemannian theory.
"""

from typing import Callable, Dict

from tonnetz.core.chord import Chord


class PLRGroup:
    """
    Implementation of the PLR group of neo-Riemannian transformations.

    The PLR group consists of three fundamental transformations:
    - P (Parallel): Changes mode while keeping root and fifth
    - L (Leittonwechsel): Leading-tone exchange
    - R (Relative): Relative major/minor transformation

    These transformations can be composed to generate the full group.
    """

    def __init__(self) -> None:
        """Initialize the PLR group with transformation mappings."""
        # Cache for composed transformations
        self._transformation_cache: Dict[str, Callable[[Chord], Chord]] = {}

        # Define basic transformations
        self._transformations = {
            "P": self.parallel,
            "L": self.leading_tone_exchange,
            "R": self.relative,
            "I": self.identity,  # Identity transformation
        }

    def parallel(self, chord: Chord) -> Chord:
        """
        Parallel transformation: Changes major to minor or vice versa.

        The parallel transformation changes the mode of a chord while keeping
        the root and fifth the same. For triads:
        - Major chord: lowers the third by a semitone
        - Minor chord: raises the third by a semitone

        Args:
            chord: Input chord to transform

        Returns:
            Transformed chord with opposite mode

        Raises:
            ValueError: If transformation is not defined for this chord quality
        """
        if chord.quality == "major":
            return Chord(chord.root, "minor", chord.inversion)
        elif chord.quality == "minor":
            return Chord(chord.root, "major", chord.inversion)
        elif chord.quality == "augmented":
            # Augmented to minor: lower the fifth
            return Chord(chord.root, "minor", chord.inversion)
        elif chord.quality == "diminished":
            # Diminished to major: raise the fifth
            return Chord(chord.root, "major", chord.inversion)
        else:
            raise ValueError(
                f"Parallel transformation not defined for {chord.quality} chords"
            )

    def leading_tone_exchange(self, chord: Chord) -> Chord:
        """
        Leittonwechsel (L) transformation: Leading-tone exchange.

        The L transformation exchanges a chord with its leading-tone related chord:
        - Major chord: exchanges with minor chord a minor third above
        - Minor chord: exchanges with major chord a minor third below

        Args:
            chord: Input chord to transform

        Returns:
            Transformed chord with leading-tone exchange

        Raises:
            ValueError: If transformation is not defined for this chord quality
        """
        if chord.quality == "major":
            # Major to minor a minor third up
            new_root = (chord.root + 3) % 12
            return Chord(new_root, "minor", chord.inversion)
        elif chord.quality == "minor":
            # Minor to major a minor third down
            new_root = (chord.root - 3) % 12
            return Chord(new_root, "major", chord.inversion)
        elif chord.quality == "augmented":
            # Augmented special case
            new_root = (chord.root + 4) % 12
            return Chord(new_root, "minor", chord.inversion)
        elif chord.quality == "diminished":
            # Diminished special case
            new_root = (chord.root - 4) % 12
            return Chord(new_root, "major", chord.inversion)
        else:
            raise ValueError(
                f"Leading-tone exchange not defined for {chord.quality} chords"
            )

    def relative(self, chord: Chord) -> Chord:
        """
        Relative (R) transformation: Relative major/minor.

        The R transformation relates a chord to its relative major or minor:
        - Major chord: to relative minor (minor third down)
        - Minor chord: to relative major (minor third up)

        Args:
            chord: Input chord to transform

        Returns:
            Transformed chord (relative major/minor)

        Raises:
            ValueError: If transformation is not defined for this chord quality
        """
        if chord.quality == "major":
            # Major to relative minor (minor third down)
            new_root = (chord.root - 3) % 12
            return Chord(new_root, "minor", chord.inversion)
        elif chord.quality == "minor":
            # Minor to relative major (minor third up)
            new_root = (chord.root + 3) % 12
            return Chord(new_root, "major", chord.inversion)
        elif chord.quality == "augmented":
            # Augmented to diminished with root change
            new_root = (chord.root - 1) % 12
            return Chord(new_root, "diminished", chord.inversion)
        elif chord.quality == "diminished":
            # Diminished to augmented with root change
            new_root = (chord.root + 1) % 12
            return Chord(new_root, "augmented", chord.inversion)
        else:
            raise ValueError(
                f"Relative transformation not defined for {chord.quality} chords"
            )

    def identity(self, chord: Chord) -> Chord:
        """
        Identity transformation: Returns the chord unchanged.

        Args:
            chord: Input chord

        Returns:
            The same chord unchanged
        """
        return chord

    def compose(self, transformations: str) -> Callable[[Chord], Chord]:
        """
        Compose a sequence of transformations.

        Args:
            transformations: String of transformation letters (e.g., "PLR", "LPLR")

        Returns:
            Function that applies the composed transformation

        Raises:
            ValueError: If any transformation letter is not recognized
        """
        # Check cache first
        if transformations in self._transformation_cache:
            return self._transformation_cache[transformations]

        # Validate all transformations
        for trans in transformations:
            if trans not in self._transformations:
                raise ValueError(f"Unknown transformation: {trans}")

        def composed_transformation(chord: Chord) -> Chord:
            """Apply the composed transformation sequence."""
            result = chord
            for trans in transformations:
                result = self._transformations[trans](result)
            return result

        # Cache the composed transformation
        self._transformation_cache[transformations] = composed_transformation
        return composed_transformation

    def apply(self, transformation: str, chord: Chord) -> Chord:
        """
        Apply a transformation or sequence of transformations to a chord.

        Args:
            transformation: Single transformation letter or sequence
            chord: Chord to transform

        Returns:
            Transformed chord
        """
        if len(transformation) == 1:
            return self._transformations[transformation](chord)
        else:
            composed_func = self.compose(transformation)
            return composed_func(chord)

    def get_transformation_matrix(self, chord_list: list[Chord]) -> list[list[Chord]]:
        """
        Generate a matrix showing all PLR transformations of given chords.

        Args:
            chord_list: List of chords to transform

        Returns:
            Matrix where entry [i][j] is transformation j applied to chord i
        """
        basic_transformations = ["I", "P", "L", "R"]
        matrix = []

        for chord in chord_list:
            row = []
            for trans in basic_transformations:
                try:
                    transformed = self.apply(trans, chord)
                    row.append(transformed)
                except ValueError:
                    # If transformation not defined, use identity
                    row.append(chord)
            matrix.append(row)

        return matrix

    def find_transformation_sequence(
        self, start: Chord, end: Chord, max_length: int = 4
    ) -> str:
        """
        Find a sequence of PLR transformations that connects two chords.

        Uses breadth-first search to find the shortest transformation sequence.

        Args:
            start: Starting chord
            end: Target chord
            max_length: Maximum sequence length to search

        Returns:
            String of transformation letters, or empty string if no path found
        """
        from collections import deque

        # BFS to find shortest path
        queue = deque([(start, "")])
        visited = {start}

        while queue:
            current_chord, path = queue.popleft()

            if len(path) >= max_length:
                continue

            if current_chord == end:
                return path

            # Try each basic transformation
            for trans in ["P", "L", "R"]:
                try:
                    next_chord = self.apply(trans, current_chord)
                    if next_chord not in visited:
                        visited.add(next_chord)
                        queue.append((next_chord, path + trans))
                except ValueError:
                    # Skip if transformation not applicable
                    continue

        return ""  # No path found

    def get_orbit(self, chord: Chord, max_steps: int = 10) -> set[Chord]:
        """
        Get the PLR orbit of a chord (all chords reachable by PLR transformations).

        Args:
            chord: Starting chord
            max_steps: Maximum number of transformation steps

        Returns:
            Set of all chords in the PLR orbit
        """
        orbit = {chord}
        current_generation = {chord}

        for _ in range(max_steps):
            next_generation = set()

            for current_chord in current_generation:
                for trans in ["P", "L", "R"]:
                    try:
                        next_chord = self.apply(trans, current_chord)
                        if next_chord not in orbit:
                            orbit.add(next_chord)
                            next_generation.add(next_chord)
                    except ValueError:
                        continue

            if not next_generation:
                break

            current_generation = next_generation

        return orbit

    def distance(self, chord1: Chord, chord2: Chord) -> int:
        """
        Calculate the PLR distance between two chords.

        Args:
            chord1: First chord
            chord2: Second chord

        Returns:
            Minimum number of PLR transformations needed to get from chord1 to chord2,
            or -1 if no path exists within reasonable search depth
        """
        path = self.find_transformation_sequence(chord1, chord2, max_length=8)
        return len(path) if path else -1


# Convenience functions for common transformations
def parallel(chord: Chord) -> Chord:
    """Apply parallel transformation to a chord."""
    return PLRGroup().parallel(chord)


def leading_tone_exchange(chord: Chord) -> Chord:
    """Apply leading-tone exchange transformation to a chord."""
    return PLRGroup().leading_tone_exchange(chord)


def relative(chord: Chord) -> Chord:
    """Apply relative transformation to a chord."""
    return PLRGroup().relative(chord)


# Alias for shorter names
L = leading_tone_exchange
P = parallel
R = relative
