"""
Chord representation and basic operations.

This module provides the core Chord class that represents musical chords
with root, quality, inversion, and optional voicing information.
"""

import json
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Chord:
    """
    Represents a musical chord with root, quality, and inversion.

    Attributes:
        root: Root pitch class (0-11, where C=0)
        quality: Chord quality ('major', 'minor', 'diminished', 'augmented')
        inversion: Inversion number (0=root position, 1=first inversion, etc.)
        voicing: Optional list of specific pitch classes for advanced voicings
    """

    root: int
    quality: str
    inversion: int = 0
    voicing: Optional[Tuple[int, ...]] = None

    # Quality to interval mapping (semitones from root) - now a class variable
    QUALITY_INTERVALS: ClassVar[Dict[str, Tuple[int, ...]]] = {
        "major": (0, 4, 7),
        "minor": (0, 3, 7),
        "diminished": (0, 3, 6),
        "augmented": (0, 4, 8),
        "major7": (0, 4, 7, 11),
        "minor7": (0, 3, 7, 10),
        "dominant7": (0, 4, 7, 10),
        "diminished7": (0, 3, 6, 9),
    }

    def __post_init__(self) -> None:
        """Validate chord parameters."""
        if not 0 <= self.root <= 11:
            raise ValueError(f"Root must be 0-11, got {self.root}")

        if self.quality not in self.QUALITY_INTERVALS:
            raise ValueError(f"Unknown quality: {self.quality}")

        if self.inversion < 0:
            raise ValueError(f"Inversion must be non-negative, got {self.inversion}")

    def pitch_classes(self) -> Tuple[int, ...]:
        """
        Get the pitch classes of this chord.

        Returns:
            Tuple of pitch classes (0-11) in the chord
        """
        if self.voicing is not None:
            return tuple(pc % 12 for pc in self.voicing)

        base_intervals = self.QUALITY_INTERVALS[self.quality]
        pitch_classes = tuple(
            (self.root + interval) % 12 for interval in base_intervals
        )

        # Apply inversion by rotating the pitch classes
        if self.inversion > 0:
            inv = min(self.inversion, len(pitch_classes) - 1)
            pitch_classes = pitch_classes[inv:] + pitch_classes[:inv]

        return pitch_classes

    def to_vector(self) -> np.ndarray:
        """
        Convert chord to a 12-dimensional binary vector (chromatic vector).

        Returns:
            NumPy array where each index represents a pitch class (C=0, C#=1, etc.)
            and the value is 1 if that pitch class is in the chord, 0 otherwise.
        """
        vector = np.zeros(12, dtype=int)
        for pc in self.pitch_classes():
            vector[pc] = 1
        return vector

    def transpose(self, semitones: int) -> "Chord":
        """
        Transpose the chord by a given number of semitones.

        Args:
            semitones: Number of semitones to transpose (can be negative)

        Returns:
            New Chord object transposed by the specified amount
        """
        new_root = (self.root + semitones) % 12

        if self.voicing is not None:
            new_voicing = tuple((pc + semitones) % 12 for pc in self.voicing)
            return Chord(new_root, self.quality, self.inversion, new_voicing)

        return Chord(new_root, self.quality, self.inversion)

    def invert(self, inversion: int) -> "Chord":
        """
        Create a new chord with the specified inversion.

        Args:
            inversion: Target inversion number

        Returns:
            New Chord object with the specified inversion
        """
        return Chord(self.root, self.quality, inversion, self.voicing)

    def contains_pitch_class(self, pitch_class: int) -> bool:
        """
        Check if the chord contains a specific pitch class.

        Args:
            pitch_class: Pitch class to check (0-11)

        Returns:
            True if the chord contains the pitch class, False otherwise
        """
        return pitch_class % 12 in self.pitch_classes()

    def common_tones(self, other: "Chord") -> int:
        """
        Count the number of common tones with another chord.

        Args:
            other: Another Chord object

        Returns:
            Number of pitch classes in common between the two chords
        """
        self_pcs = set(self.pitch_classes())
        other_pcs = set(other.pitch_classes())
        return len(self_pcs & other_pcs)

    def voice_leading_distance(self, other: "Chord") -> float:
        """
        Calculate voice leading distance to another chord.
        Uses minimum total semitone movement between voices.

        Args:
            other: Target chord

        Returns:
            Total semitone distance for minimal voice leading
        """
        self_pcs = list(self.pitch_classes())
        other_pcs = list(other.pitch_classes())

        # Pad shorter chord with repeated notes if needed
        max_len = max(len(self_pcs), len(other_pcs))
        while len(self_pcs) < max_len:
            self_pcs.append(self_pcs[0])
        while len(other_pcs) < max_len:
            other_pcs.append(other_pcs[0])

        # Find minimum distance assignment using simple greedy approach
        # For more accuracy, could use Hungarian algorithm
        total_distance = 0
        used_targets = set()

        for source_pc in self_pcs:
            min_dist = float("inf")
            best_target = None

            for target_pc in other_pcs:
                if target_pc in used_targets:
                    continue

                # Calculate minimal distance on circle of fifths
                dist = min(abs(source_pc - target_pc), 12 - abs(source_pc - target_pc))

                if dist < min_dist:
                    min_dist = dist
                    best_target = target_pc

            if best_target is not None:
                total_distance += min_dist
                used_targets.add(best_target)

        return total_distance

    def to_dict(self) -> Dict:
        """
        Serialize chord to dictionary.

        Returns:
            Dictionary representation of the chord
        """
        result = {
            "root": self.root,
            "quality": self.quality,
            "inversion": self.inversion,
        }
        if self.voicing is not None:
            result["voicing"] = list(self.voicing)
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "Chord":
        """
        Deserialize chord from dictionary.

        Args:
            data: Dictionary containing chord data

        Returns:
            Chord object reconstructed from dictionary
        """
        voicing = None
        if "voicing" in data:
            voicing = tuple(data["voicing"])

        return cls(
            root=data["root"],
            quality=data["quality"],
            inversion=data.get("inversion", 0),
            voicing=voicing,
        )

    def to_json(self) -> str:
        """
        Serialize chord to JSON string.

        Returns:
            JSON string representation of the chord
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "Chord":
        """
        Deserialize chord from JSON string.

        Args:
            json_str: JSON string containing chord data

        Returns:
            Chord object reconstructed from JSON
        """
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_name(cls, name: str) -> "Chord":
        """
        Create a Chord from a name string.

        Args:
            name: Chord name (e.g., "C", "Am", "F#maj7")

        Returns:
            Chord object
        """
        return parse_chord_name(name)

    def __str__(self) -> str:
        """String representation of the chord."""
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        root_name = note_names[self.root]

        quality_symbols = {
            "major": "",
            "minor": "m",
            "diminished": "dim",
            "augmented": "aug",
            "major7": "maj7",
            "minor7": "m7",
            "dominant7": "7",
            "diminished7": "dim7",
        }

        quality_str = quality_symbols.get(self.quality, self.quality)

        result = f"{root_name}{quality_str}"

        if self.inversion > 0:
            result += f"/{self.inversion}"

        return result

    def __repr__(self) -> str:
        """Detailed string representation of the chord."""
        return f"Chord(root={self.root}, quality='{self.quality}', inversion={self.inversion})"


# Predefined common chords for convenience
class CommonChords:
    """Collection of commonly used chord objects."""

    # Major chords
    C_MAJOR = Chord(0, "major")
    D_MAJOR = Chord(2, "major")
    E_MAJOR = Chord(4, "major")
    F_MAJOR = Chord(5, "major")
    G_MAJOR = Chord(7, "major")
    A_MAJOR = Chord(9, "major")
    B_MAJOR = Chord(11, "major")

    # Minor chords
    C_MINOR = Chord(0, "minor")
    D_MINOR = Chord(2, "minor")
    E_MINOR = Chord(4, "minor")
    F_MINOR = Chord(5, "minor")
    G_MINOR = Chord(7, "minor")
    A_MINOR = Chord(9, "minor")
    B_MINOR = Chord(11, "minor")

    # Diminished chords
    C_DIM = Chord(0, "diminished")
    D_DIM = Chord(2, "diminished")
    E_DIM = Chord(4, "diminished")
    F_DIM = Chord(5, "diminished")
    G_DIM = Chord(7, "diminished")
    A_DIM = Chord(9, "diminished")
    B_DIM = Chord(11, "diminished")


def parse_chord_name(name: str) -> Chord:
    """
    Parse a chord name string into a Chord object.

    Args:
        name: Chord name string (e.g., "Cm", "F#maj7", "Bb/3")

    Returns:
        Chord object representing the parsed chord

    Raises:
        ValueError: If the chord name cannot be parsed
    """
    # This is a simplified parser - could be expanded significantly
    name = name.strip()

    # Note name to pitch class mapping
    note_to_pc = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
    }

    # Extract root note (prioritize longer matches)
    root_match = None
    sorted_notes = sorted(note_to_pc.items(), key=lambda x: -len(x[0]))
    for note_name, pc in sorted_notes:
        if name.startswith(note_name):
            root_match = (note_name, pc)
            break

    if root_match is None:
        raise ValueError(f"Could not parse root note from '{name}'")

    root_name, root = root_match
    remaining = name[len(root_name) :]

    # Extract inversion if present
    inversion = 0
    if "/" in remaining:
        parts = remaining.split("/")
        remaining = parts[0]
        try:
            inversion = int(parts[1])
        except ValueError:
            raise ValueError(f"Could not parse inversion from '{name}'")

    # Determine quality from remaining string
    quality_map = {
        "": "major",
        "m": "minor",
        "min": "minor",
        "maj": "major",
        "dim": "diminished",
        "aug": "augmented",
        "+": "augmented",
        "7": "dominant7",
        "maj7": "major7",
        "m7": "minor7",
        "min7": "minor7",
        "dim7": "diminished7",
    }

    quality = quality_map.get(remaining, None)
    if quality is None:
        raise ValueError(f"Unknown chord quality: '{remaining}'")

    return Chord(root, quality, inversion)
