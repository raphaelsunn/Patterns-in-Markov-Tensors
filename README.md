# Markov-Chain-analysis-of-musical-engagement
Research code for studying musical structure via Markov chains and information theory defining engagement produced by atonal music.

These files were used in the study "Markov Chains and Information Theory in musical analysis of engagement produced by atonal music" ()URL to perform and visualize the analyses.
They may have been modified over time and might not function correctly in their current state.
If needed, feel free to contact me at: raphaelluca.sangiorgi@gmail.com



\subsection{convertermidicsv.py}
A Python program was written to convert MIDI files into a more convenient
text-based CSV format. The program extracts musical events and reorganizes
them into a temporally ordered sequence. During this conversion, each note
event is identified by pairing the corresponding \texttt{note\_on} and
\texttt{note\_off} messages, or equivalently a \texttt{note\_on} message
with velocity equal to zero, in order to determine precisely the beginning
and end of each sound.

In this way, the actual duration of each note can be reconstructed, first
measured in ticks and then normalized with respect to the MIDI file's
ticks-per-beat value, that is, according to the temporal unit of the
composition. The final result of the conversion is a structured CSV file in
which each row represents a musical event and contains \texttt{event} (the
MIDI note number), \texttt{duration} (the actual duration of the note), and
\texttt{velocity} (the intensity of the performance).

The variable \texttt{transition} is also considered, representing the time
interval until the onset of the following note. This distinction makes it
possible to represent overlapping notes correctly, namely cases in which one
note continues to sound while a new note begins. Durations, transitions, and
velocities are also approximated into 12 discrete rhythmic values, in order
to make the sequence more stable for the subsequent statistical analysis.
This textual representation constitutes the basis for the following analyses.

<<<<< Il codice è riportato in ...>>>>

\subsection{analysis.py}
The analysis program that generated Figs.~10, 13, 14, and 15 is reported in 

<<<<<<github/.....>>>>

. It follows the logic discussed above and is structured as
follows.

The code is organized around the main class
\texttt{MusicalMarkovEngine}, which acts as the engine of the entire analysis
process. This class manages data loading, the construction of probabilistic
models, and the extraction of structural patterns from the musical sequence.
The first step of the analysis is performed by the method
\texttt{load\_and\_clean()}, which reads the CSV file generated during the
MIDI conversion stage and extracts the relevant musical information. At this
stage, the fundamental properties of the musical events are retrieved:

\begin{itemize}
    \item \texttt{pitch} (MIDI note number);
    \item \texttt{duration} (duration of the note);
    \item \texttt{transition} (time interval until the following note);
    \item \texttt{velocity} (intensity of the performance).
\end{itemize}

Pauses are removed from the melodic sequence, since the Markov analysis is
performed exclusively on transitions between successive pitches. During this
stage, the program also determines the effective musical alphabet of the
piece, calculated from the pitch range covered by the composition. The number
of symbols in the alphabet is therefore defined as

\[
N = 12\cdot o,
\]

where $o$ represents the number of octaves covered by the sequence. Once the
notes have been extracted, the sequence is transformed into a sequence of
intervals by calculating the difference between consecutive pitches. This
operation is performed internally by the method
\texttt{load\_and\_clean()}, generating the sequence

\[
\Delta p_i = p_{i+1} - p_i.
\]

This representation makes the model invariant under transposition, since the
musical structure is described in terms of relationships between notes rather
than absolute pitch values.

The memory length of the Markov model is automatically estimated by the
method \texttt{find\_optimal\_k()}. The parameter $k$ represents the number
of past events considered in predicting the following event in the sequence.
To determine the optimal value of $k$, the Past--Future Mutual Information is
calculated as

\[
I(k) =
H(\text{Past}) +
H(\text{Future}) -
H(\text{Past, Future}).
\]

The order of the chain is identified at the point where the additional
information provided by increasing the memory becomes marginal, according to
the plateau criterion.

Once the order $k$ has been determined, the method
\texttt{build\_transition\_tensor()} constructs a sparse representation of
the transitions between Markov states. Each state of the chain is defined by
a context of $k$ consecutive intervals, while the transition represents the
passage to the following interval in the sequence. At this stage, the program
constructs the observed transition frequencies and the context totals,
together with a local convergence structure measuring how many different
preceding configurations lead to the same transition.

This local convergence index is denoted by $c_{i\rightarrow j}$ and
represents the number of different histories leading to the same melodic
transition. The method \texttt{compute\_stationary\_entropy()} is not used
for the purposes of the present article. It calculates the stationary entropy
of the Markov chain by solving for the stationary distribution $\pi$ of the
transition matrix. The chain entropy is defined as

\[
H =
-\sum_i \pi_i
\sum_j p_{ij}\log_2 p_{ij},
\]

and represents the average level of informational uncertainty generated by
the musical sequence.

The identification of recurrent structures is performed by the method
\texttt{extract\_patterns()}. The procedure scans the interval sequence and
generates all candidate subsequences within a predefined minimum and maximum
length. Some patterns are automatically excluded through the method
\texttt{\_is\_alternating()}, which removes artifacts consisting of cyclic
or mechanical sequences with low internal variability.

For each candidate pattern, a structural coherence index $\psi$ is calculated
from the combination of transition probabilities and the local convergence
index. The overall relevance of a pattern is then defined through

\[
\phi = F\cdot\psi,
\]

where $F$ is the frequency of occurrence of the pattern within the sequence
and $\psi$ represents its structural coherence. The most significant patterns
are selected through an adaptive threshold proportional to the maximum
observed value of $\phi$.

Once the main patterns have been identified, the method
\texttt{export\_patterns\_to\_midi()} exports them in MIDI format, allowing
them to be compared by listening with the structures recognized within the
original piece.

The method \texttt{analyze\_temporal\_surprise()} analyzes the temporal
distribution of their occurrences throughout the composition. Temporal
surprise is calculated as the deviation between the observed distance
separating two consecutive occurrences and the mean return distance of the
pattern,

\[
S = |t_i-\bar{t}|,
\]

where $t_i$ represents the temporal distance between two successive
occurrences.

Finally, the method \texttt{plot\_markovian\_surprise()} combines temporal
surprise with the structural variability of the patterns in order to
construct the dynamic function $S(t)$.
