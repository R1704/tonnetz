"""
Tonnetz: A Cellular Automaton Chord Engine

A Python library for exploring chord progressions through neo-Riemannian 
transformations on a toroidal Tonnetz, combined with cellular automaton dynamics.
"""

__version__ = "0.1.0"
__author__ = "Ron"
__email__ = "ron@example.com"

from tonnetz.core.chord import Chord
from tonnetz.core.neo_riemannian import PLRGroup
from tonnetz.core.tonnetz import ToroidalTonnetz
from tonnetz.automaton.grid import ToroidalGrid

__all__ = [
    "Chord",
    "PLRGroup", 
    "ToroidalTonnetz",
    "ToroidalGrid",
]
