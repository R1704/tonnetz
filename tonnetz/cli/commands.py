"""
Command-line interface for the Tonnetz library.

This module provides CLI commands for running simulations, generating
visualizations, and serving the web interface.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import click
import yaml

from tonnetz.automaton.grid import ToroidalGrid
from tonnetz.core.chord import Chord, parse_chord_name
from tonnetz.progression.markov import MarkovProgression
from tonnetz.progression.rule_based import RuleBasedProgression
from tonnetz.progression.search_based import SearchBasedProgression
from tonnetz.visualization.static import TonnetzPlotter


@click.group()
@click.version_option()
def cli():
    """Tonnetz: Cellular Automaton Chord Engine."""
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Configuration file (YAML or JSON)",
)
@click.option("--width", "-w", default=12, help="Grid width")
@click.option("--height", "-h", default=12, help="Grid height")
@click.option("--steps", "-s", default=10, help="Number of simulation steps")
@click.option(
    "--start-chord", default="C", help='Starting chord (e.g., "C", "Am", "F#maj7")'
)
@click.option(
    "--algorithm",
    default="rule_based",
    type=click.Choice(["rule_based", "markov", "search_based", "random"]),
    help="Progression algorithm to use",
)
@click.option("--pattern", help="Pattern for rule-based algorithm")
@click.option("--key", default=0, help="Key center (0-11, where C=0)")
@click.option(
    "--mode",
    default="major",
    type=click.Choice(["major", "minor"]),
    help="Mode for rule-based algorithm",
)
@click.option("--output", "-o", help="Output file for results")
@click.option("--save-animation", help="Directory to save animation frames")
@click.option(
    "--neighborhood",
    default="moore",
    type=click.Choice(["moore", "von_neumann"]),
    help="Neighborhood type for cellular automaton",
)
@click.option(
    "--update-mode",
    default="synchronous",
    type=click.Choice(["synchronous", "asynchronous"]),
    help="Update mode for cellular automaton",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def simulate(
    config: Optional[str],
    width: int,
    height: int,
    steps: int,
    start_chord: str,
    algorithm: str,
    pattern: Optional[str],
    key: int,
    mode: str,
    output: Optional[str],
    save_animation: Optional[str],
    neighborhood: str,
    update_mode: str,
    verbose: bool,
):
    """Run a cellular automaton simulation."""

    # Load configuration if provided
    params = {}
    if config:
        params = load_config(config)

    # Override with command line arguments
    grid_params = {
        "width": params.get("grid", {}).get("width", width),
        "height": params.get("grid", {}).get("height", height),
        "neighborhood": params.get("automaton", {}).get("neighborhood", neighborhood),
        "update_mode": params.get("automaton", {}).get("update_mode", update_mode),
    }

    algo_params = {
        "algorithm": params.get("progression", {}).get("type", algorithm),
        "pattern": params.get("progression", {})
        .get("params", {})
        .get("pattern", pattern),
        "key": params.get("progression", {}).get("params", {}).get("key", key),
        "mode": params.get("progression", {}).get("params", {}).get("mode", mode),
    }

    sim_params = {
        "steps": params.get("simulation", {}).get("steps", steps),
        "start_chord": params.get("simulation", {}).get("start_chord", start_chord),
    }

    if verbose:
        click.echo(f"Grid: {grid_params['width']}x{grid_params['height']}")
        click.echo(f"Algorithm: {algo_params['algorithm']}")
        click.echo(f"Steps: {sim_params['steps']}")
        click.echo(f"Start chord: {sim_params['start_chord']}")

    # Create grid
    grid = ToroidalGrid(
        width=grid_params["width"],
        height=grid_params["height"],
        neighborhood=grid_params["neighborhood"],
        update_mode=grid_params["update_mode"],
    )

    # Set up progression algorithm
    if algo_params["algorithm"] == "rule_based":
        progression_algo = RuleBasedProgression(
            key=algo_params["key"],
            mode=algo_params["mode"],
            pattern=algo_params["pattern"],
        )
    elif algo_params["algorithm"] == "markov":
        progression_algo = MarkovProgression(order=algo_params.get("order", 1))
    elif algo_params["algorithm"] == "search_based":
        progression_algo = SearchBasedProgression(
            search_method=algo_params.get("search_method", "beam"),
            beam_width=algo_params.get("beam_width", 10),
        )
    else:
        # Random fallback
        def random_progression_algo():
            import random

            root = random.randint(0, 11)
            quality = random.choice(["major", "minor"])
            return Chord(root, quality)

        progression_algo = random_progression_algo

    # Parse starting chord
    try:
        start = parse_chord_name(sim_params["start_chord"])
    except ValueError:
        click.echo(f"Invalid chord name: {sim_params['start_chord']}", err=True)
        return

    # Populate grid
    if hasattr(progression_algo, "generate"):
        grid.populate_with_progression(progression_algo, start)
    else:
        grid.populate_random(progression_algo)

    if verbose:
        click.echo(f"Initial grid populated with {algo_params['algorithm']} algorithm")

    # Run simulation
    plotter = TonnetzPlotter()
    frame_paths = []

    if save_animation:
        # Generate animation frames
        def simple_rule(cell, neighbors):
            """Simple majority rule for simulation."""
            if not neighbors:
                return cell.chord

            # Count chord occurrences
            chord_counts = {}
            for neighbor in neighbors:
                chord = neighbor.chord
                chord_counts[chord] = chord_counts.get(chord, 0) + 1

            # Return most common chord
            most_common = max(chord_counts.items(), key=lambda x: x[1])[0]
            return most_common

        frame_paths = plotter.plot_grid_animation_frames(
            grid, simple_rule, sim_params["steps"], save_animation
        )

        if verbose:
            click.echo(f"Animation frames saved to {save_animation}")
            click.echo(f"Generated {len(frame_paths)} frames")
    else:
        # Run simulation without saving frames
        for step in range(sim_params["steps"]):
            grid.update()
            if verbose:
                stats = grid.get_statistics()
                click.echo(
                    f"Step {step + 1}: {stats['chord_changes']} changes, "
                    f"{stats['active_cells']} active cells"
                )

    # Save results
    if output:
        save_results(grid, output, verbose)

    # Display final statistics
    final_stats = grid.get_statistics()
    click.echo("\nFinal Statistics:")
    click.echo(f"Generation: {final_stats['generation']}")
    click.echo(f"Total cells: {final_stats['total_cells']}")
    click.echo(f"Active cells: {final_stats['active_cells']}")
    click.echo(f"Average activation: {final_stats['average_activation']:.3f}")
    click.echo(f"Harmony level: {final_stats['harmony_level']:.3f}")


@cli.command()
@click.option("--grid-size", default="12x12", help='Grid size (e.g., "12x12")')
@click.option(
    "--algorithm",
    default="rule_based",
    type=click.Choice(["rule_based", "random"]),
    help="Algorithm for grid population",
)
@click.option("--pattern", help="Pattern for grid population")
@click.option(
    "--color-by",
    default="quality",
    type=click.Choice(["quality", "root", "activation"]),
    help="How to color the grid cells",
)
@click.option("--output", "-o", required=True, help="Output image file")
@click.option("--title", help="Plot title")
@click.option(
    "--format",
    "img_format",
    default="png",
    type=click.Choice(["png", "pdf", "svg"]),
    help="Output image format",
)
@click.option("--dpi", default=300, help="Image resolution (DPI)")
@click.option("--figsize", default="12x10", help='Figure size (e.g., "12x10")')
def visualize(
    grid_size: str,
    algorithm: str,
    pattern: Optional[str],
    color_by: str,
    output: str,
    title: Optional[str],
    img_format: str,
    dpi: int,
    figsize: str,
):
    """Generate static visualization of a Tonnetz grid."""

    # Parse grid size
    try:
        width, height = map(int, grid_size.split("x"))
    except ValueError:
        click.echo(f"Invalid grid size format: {grid_size}", err=True)
        return

    # Parse figure size
    try:
        fig_width, fig_height = map(float, figsize.split("x"))
    except ValueError:
        click.echo(f"Invalid figure size format: {figsize}", err=True)
        return

    # Create grid
    grid = ToroidalGrid(width=width, height=height)

    # Populate grid
    if algorithm == "rule_based":
        progression_algo = RuleBasedProgression(pattern=pattern)
        start_chord = Chord(0, "major")  # C major
        grid.populate_with_progression(progression_algo, start_chord)
    else:
        grid.populate_random()

    # Create visualization
    plotter = TonnetzPlotter(figsize=(fig_width, fig_height))

    plot_title = title or f"Tonnetz Grid ({algorithm})"

    # Ensure output has correct extension
    if not output.endswith(f".{img_format}"):
        output = f"{output}.{img_format}"

    plotter.plot_grid(
        grid, title=plot_title, color_by=color_by, save_path=output, show=False
    )

    click.echo(f"Visualization saved to {output}")


@cli.command()
@click.option("--host", default="localhost", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool):
    """Start the web interface server."""
    try:
        import uvicorn

        from tonnetz.web.app import app

        click.echo(f"Starting server on http://{host}:{port}")
        click.echo("Press Ctrl+C to stop")

        uvicorn.run(app, host=host, port=port, reload=reload)

    except ImportError:
        click.echo("Web interface dependencies not installed.", err=True)
        click.echo("Install with: pip install tonnetz[web]", err=True)
    except ModuleNotFoundError:
        click.echo("Web interface not yet implemented.", err=True)


@cli.command()
@click.argument("chord1")
@click.argument("chord2")
@click.option("--max-length", default=4, help="Maximum transformation sequence length")
def transform(chord1: str, chord2: str, max_length: int):
    """Find PLR transformation sequence between two chords."""
    from tonnetz.core.neo_riemannian import PLRGroup

    try:
        start_chord = parse_chord_name(chord1)
        end_chord = parse_chord_name(chord2)
    except ValueError as e:
        click.echo(f"Error parsing chords: {e}", err=True)
        return

    plr = PLRGroup()
    sequence = plr.find_transformation_sequence(start_chord, end_chord, max_length)

    if sequence:
        click.echo(f"{chord1} → {chord2}")
        click.echo(f"Transformation sequence: {' → '.join(sequence)}")
        click.echo(f"Length: {len(sequence)}")

        # Show step-by-step transformation
        current_chord = start_chord
        click.echo("\nStep-by-step:")
        click.echo(f"0: {current_chord}")

        for i, transformation in enumerate(sequence):
            current_chord = plr.apply(transformation, current_chord)
            click.echo(f"{i + 1}: {current_chord} (via {transformation})")
    else:
        click.echo(f"No transformation sequence found within {max_length} steps")


@cli.command()
@click.argument("config_file", type=click.Path())
@click.option(
    "--template",
    default="basic",
    type=click.Choice(["basic", "advanced", "minimal"]),
    help="Configuration template to generate",
)
def init_config(config_file: str, template: str):
    """Generate a configuration file template."""

    templates = {
        "basic": {
            "grid": {"width": 12, "height": 12},
            "initial_pattern": "checkerboard",
            "progression": {
                "type": "rule_based",
                "params": {
                    "key": 0,
                    "mode": "major",
                    "pattern": "ii-V-I",
                    "randomness": 0.1,
                },
            },
            "automaton": {"neighborhood": "moore", "update_mode": "synchronous"},
            "simulation": {"steps": 10, "start_chord": "C"},
            "visualization": {
                "type": "static",
                "color_by": "quality",
                "figsize": [12, 10],
                "output": "tonnetz.png",
            },
        },
        "minimal": {"grid": {"width": 8, "height": 8}, "simulation": {"steps": 5}},
        "advanced": {
            "grid": {"width": 16, "height": 16},
            "initial_pattern": "center",
            "progression": {
                "type": "rule_based",
                "params": {
                    "key": 0,
                    "mode": "major",
                    "pattern": "circle_of_fifths",
                    "randomness": 0.2,
                },
            },
            "automaton": {"neighborhood": "moore", "update_mode": "asynchronous"},
            "simulation": {"steps": 20, "start_chord": "Cmaj7"},
            "visualization": {
                "type": "animation",
                "color_by": "root",
                "figsize": [14, 12],
                "output_dir": "./animation_frames/",
                "fps": 2,
            },
            "analysis": {
                "track_harmony": True,
                "pattern_detection": True,
                "export_midi": True,
            },
        },
    }

    config = templates.get(template, templates["basic"])

    # Write configuration file
    file_path = Path(config_file)
    if file_path.suffix.lower() == ".json":
        with open(file_path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        # Default to YAML
        if not file_path.suffix:
            file_path = file_path.with_suffix(".yaml")
        with open(file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

    click.echo(f"Configuration template '{template}' written to {file_path}")


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    file_path = Path(config_file)

    try:
        with open(file_path, "r") as f:
            if file_path.suffix.lower() == ".json":
                return json.load(f)
            else:
                return yaml.safe_load(f) or {}
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        return {}


def save_results(grid: ToroidalGrid, output: str, verbose: bool = False) -> None:
    """Save simulation results to file."""
    file_path = Path(output)

    # Export grid state
    state = grid.export_state()

    # Add additional analysis
    stats = grid.get_statistics()
    patterns = grid.find_patterns()

    results = {
        "grid_state": state,
        "statistics": stats,
        "patterns": patterns,
        "metadata": {
            "version": "0.1.0",
            "export_time": str(Path().cwd()),  # placeholder
        },
    }

    try:
        if file_path.suffix.lower() == ".json":
            with open(file_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
        else:
            # Default to YAML
            if not file_path.suffix:
                file_path = file_path.with_suffix(".yaml")
            with open(file_path, "w") as f:
                yaml.dump(results, f, default_flow_style=False, indent=2)

        if verbose:
            click.echo(f"Results saved to {file_path}")

    except Exception as e:
        click.echo(f"Error saving results: {e}", err=True)


# Entry points for setuptools
def simulate_command():
    """Entry point for tonnetz-simulate command."""
    simulate()


def visualize_command():
    """Entry point for tonnetz-visualize command."""
    visualize()


def serve_command():
    """Entry point for tonnetz-serve command."""
    serve()


if __name__ == "__main__":
    cli()
