# Recognizing musical patterns and computing a measure related to engagement

Research code for studying musical structure via Markov chains and
information theory, defining engagement produced by atonal music.

These files were used in the study *"Markov Chains and Information Theory
in musical analysis of engagement produced by atonal music"* (URL) to
perform and visualize the analyses. They may have been modified over time
and might not function correctly in their current state.

If needed, feel free to contact me at: raphaelluca.sangiorgi@gmail.com

## Contents

| File | Description |
|---|---|
| `convertermidicsv.py` | Converts MIDI files into a structured CSV representation |
| `analysis.py` | Builds the Markov models, extracts patterns, computes φ, S_t, S_v and C |

## convertermidicsv.py

Converts MIDI files into a more convenient text-based CSV format. The
program extracts musical events and reorganizes them into a temporally
ordered sequence. Each note event is identified by pairing the
corresponding `note_on` and `note_off` messages (or a `note_on` with
velocity zero), in order to determine precisely the beginning and end of
each sound.

The actual duration of each note is reconstructed, first measured in ticks
and then normalized with respect to the MIDI file's ticks-per-beat value.
Each row of the resulting CSV represents a musical event and contains
`event` (the MIDI note number), `duration`, and `velocity`. The variable
`transition` represents the time interval until the onset of the following
note, which makes it possible to represent overlapping notes correctly.
Durations, transitions, and velocities are approximated into 12 discrete
rhythmic values to make the sequence more stable for the statistical
analysis.

## analysis.py

Organized around the main class `MusicalMarkovEngine`, which manages data
loading, the construction of probabilistic models, and the extraction of
structural patterns.

- `load_and_clean()` — reads the CSV, extracts `pitch`, `duration`,
  `transition`, `velocity`; removes pauses; determines the alphabet
  (N = 12·o, with o the number of octaves covered) and converts the note
  sequence into intervals (Δp_i = p_{i+1} − p_i), making the model
  invariant under transposition.
- `find_optimal_k()` — estimates the memory length k through the
  Past–Future Mutual Information I(k) = H(Past) + H(Future) −
  H(Past, Future), selected at the plateau of the curve.
- `build_transition_tensor()` — builds a sparse representation of the
  transitions, with observed frequencies, context totals, and the local
  convergence index c (number of different histories leading to the same
  transition).
- `compute_stationary_entropy()` — stationary entropy of the chain (not
  used for the purposes of the article).
- `extract_patterns()` — scans the interval sequence, generates candidate
  subsequences within a minimum and maximum length; `_is_alternating()`
  removes cyclic/mechanical artifacts. For each candidate a structural
  coherence index ψ is computed, and the overall relevance is φ = F·ψ,
  where F is the frequency of occurrence. The most significant patterns
  are selected through an adaptive threshold proportional to max(φ).
- `export_patterns_to_midi()` — exports the detected patterns as MIDI,
  allowing them to be compared by listening with the original piece.
- `analyze_temporal_surprise()` — analyzes the temporal distribution of
  pattern occurrences and the deviation of return times from the
  characteristic return distance of each pattern.
