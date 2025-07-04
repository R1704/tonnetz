# Tonnetz: Cellular Automaton Chord Engine

A Python library for exploring chord progressions through neo-Riemannian transformations on a toroidal Tonnetz, combined with cellular automaton dynamics.

## Features

- **Tonnetz Representation**: Map chords onto a 2D toroidal lattice with modular wrap-around
- **Neo-Riemannian Transformations**: Implement P (Parallel), L (Leittonwechsel), R (Relative) operations
- **Pluggable Progression Algorithms**: Rule-based, Markov chain, and search-driven chord progressions
  - **Rule-based**: Pattern-driven progressions (I-vi-IV-V, ii-V-I, etc.)
  - **Markov Chain**: Statistical progression generation based on training data
  - **Search-based**: AI-driven optimization for voice leading, harmonic tension, and musical objectives
- **Cellular Automaton Engine**: 2D grid of agents applying local transformation rules
- **Visualization**: Static and interactive plots of chord evolution on the torus
- **MIDI Support**: Export progressions as MIDI files
- **Configuration-driven**: YAML/JSON configuration for all simulation parameters

## Installation

```bash
pip install tonnetz
```

For development:

```bash
git clone https://github.com/ronuser/tonnetz.git
cd tonnetz
pip install -e ".[dev]"
```

## Quick Start

```python
from tonnetz.core.chord import Chord
from tonnetz.core.neo_riemannian import PLRGroup
from tonnetz.automaton.grid import ToroidalGrid
from tonnetz.progression.rule_based import RuleBasedProgression

# Create a chord and apply transformations
chord = Chord(root=0, quality='major')  # C major
plr = PLRGroup()
chord_p = plr.parallel(chord)  # C minor

# Set up a cellular automaton
grid = ToroidalGrid(width=12, height=12)
progression = RuleBasedProgression()
grid.populate_random(progression)

# Run simulation
for step in range(10):
    grid.update()
    
# Visualize
from tonnetz.visualization.static import plot_grid
plot_grid(grid, save_path='tonnetz_evolution.png')
```

## CLI Usage

```bash
# Run a simulation with configuration file
tonnetz-simulate --config examples/basic_config.yaml --steps 20 --output simulation.midi

# Start interactive visualization server
tonnetz-serve --port 8000

# Generate static visualization
tonnetz-visualize --grid-size 12x12 --algorithm markov --output tonnetz.png
```

## Project Structure

```
tonnetz/
├── core/                 # Core music theory and geometry
│   ├── chord.py         # Chord representation and operations
│   ├── neo_riemannian.py # PLR transformations
│   └── tonnetz.py       # Toroidal lattice geometry
├── progression/         # Chord progression algorithms
│   ├── base.py         # Abstract base classes
│   ├── rule_based.py   # Functional harmony rules
│   ├── markov.py       # Markov chain progressions
│   └── search.py       # A* search-based progressions
├── automaton/          # Cellular automaton engine
│   ├── cell.py         # Cell/Agent representation
│   ├── grid.py         # 2D toroidal grid management
│   ├── rules.py        # Local transformation rules
│   └── scheduler.py    # Update scheduling
├── visualization/      # Plotting and UI
│   ├── static.py       # Matplotlib plotting
│   ├── interactive.py  # Bokeh interactive plots
│   └── web/           # FastAPI web interface
├── io/                # Configuration and I/O
│   ├── config.py      # YAML/JSON configuration
│   ├── midi.py        # MIDI export/import
│   └── serialization.py # Chord serialization
└── cli/               # Command-line interface
    └── commands.py    # CLI command implementations
```

## Documentation

Full documentation is available at [tonnetz.readthedocs.io](https://tonnetz.readthedocs.io).

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on neo-Riemannian theory and the Tonnetz as described in music theory literature
- Inspired by cellular automaton research in computational creativity
- Built with the excellent [music21](https://web.mit.edu/music21/) library

### Progression Algorithms

```python
# Rule-based progression
from tonnetz.progression.rule_based import RuleBasedProgression
rule_algo = RuleBasedProgression(pattern='I-vi-IV-V', key=0, mode='major')
progression = rule_algo.generate(start_chord, 8)

# Markov chain progression
from tonnetz.progression.markov import MarkovProgression
markov_algo = MarkovProgression(order=1)
# Train on custom progressions or use built-in training
markov_algo.train([['C', 'Am', 'F', 'G'], ['C', 'F', 'G', 'C']])
progression = markov_algo.generate(start_chord, 8)

# Search-based progression with multiple objectives
from tonnetz.progression.search_based import SearchBasedProgression, SearchObjective
search_algo = SearchBasedProgression(
    search_method='beam',
    objectives=[SearchObjective.SMOOTH_VOICE_LEADING, SearchObjective.HARMONIC_TENSION],
    objective_weights=[0.7, 0.3]
)
progression = search_algo.generate(start_chord, 8)
```
