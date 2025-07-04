"""
Example: Comparing Different Progression Algorithms

This example demonstrates the different chord progression algorithms
available in the Tonnetz library and compares their outputs.
"""

from tonnetz.core.chord import Chord
from tonnetz.progression.rule_based import RuleBasedProgression
from tonnetz.progression.markov import MarkovProgression
from tonnetz.progression.search_based import SearchBasedProgression, SearchObjective


def main():
    """Run example comparing different progression algorithms."""
    print("🎵 Tonnetz Progression Algorithm Comparison")
    print("=" * 50)
    
    # Starting chord
    start_chord = Chord(0, 'major')  # C major
    length = 6
    
    print(f"Starting chord: {start_chord}")
    print(f"Progression length: {length} chords")
    print()
    
    # 1. Rule-based progression
    print("1. Rule-based Progression (I-vi-IV-V pattern)")
    rule_algo = RuleBasedProgression(pattern='I-vi-IV-V')
    rule_progression = rule_algo.generate(start_chord, length)
    print("Progression:", " → ".join(str(chord) for chord in rule_progression))
    print()
    
    # 2. Markov chain progression
    print("2. Markov Chain Progression (trained on common progressions)")
    markov_algo = MarkovProgression(order=1)
    markov_progression = markov_algo.generate(start_chord, length)
    print("Progression:", " → ".join(str(chord) for chord in markov_progression))
    print()
    
    # 3. Search-based progression with smooth voice leading
    print("3. Search-based Progression (optimizing for smooth voice leading)")
    search_algo = SearchBasedProgression(
        search_method='beam',
        objectives=[SearchObjective.SMOOTH_VOICE_LEADING],
        beam_width=10
    )
    search_progression = search_algo.generate(start_chord, length)
    print("Progression:", " → ".join(str(chord) for chord in search_progression))
    print()
    
    # 4. Search-based progression with multiple objectives
    print("4. Search-based Progression (multiple objectives)")
    multi_algo = SearchBasedProgression(
        search_method='beam',
        objectives=[SearchObjective.SMOOTH_VOICE_LEADING, SearchObjective.HARMONIC_TENSION],
        objective_weights=[0.7, 0.3],
        beam_width=5
    )
    multi_progression = multi_algo.generate(start_chord, length)
    print("Progression:", " → ".join(str(chord) for chord in multi_progression))
    print()
    
    # Analyze voice leading distances
    print("Voice Leading Analysis:")
    print("-" * 30)
    
    for name, progression in [
        ("Rule-based", rule_progression),
        ("Markov", markov_progression),
        ("Search (smooth)", search_progression),
        ("Search (multi)", multi_progression)
    ]:
        total_distance = 0
        for i in range(len(progression) - 1):
            distance = progression[i].voice_leading_distance(progression[i + 1])
            total_distance += distance
        
        avg_distance = total_distance / (len(progression) - 1) if len(progression) > 1 else 0
        print(f"{name:15}: Average voice leading distance = {avg_distance:.2f}")
    
    print()
    print("✅ Example completed!")


if __name__ == "__main__":
    main()
