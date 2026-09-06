import os
import pandas as pd
import numpy as np
import math
from collections import Counter
from scipy.stats import mode
from scipy.ndimage import gaussian_filter1d


class MusicalMarkovEngine:
    """
    Statistical analysis engine for musical sequences.
    Handles the conversion into intervals and the computation of the
    information-theoretic metrics.
    """

    def __init__(self, path):
        self.path = path
        self.notes = []
        self.intervals = []
        self.best_k = 0
        self.raw_notes_data = []
        self.octaves = 0
        self.alphabet_size = 0
        self.base_midi = 0
        self.tensor = None
        self.ci_vector = None

    def load_and_clean(self):
        """
        Loads the CSV and computes the alphabet parameters.
        Supports both CSV formats used in the study. Also extracts the
        'transition' value to handle rhythmic overlaps (polyphony).
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Path not found: {self.path}")

        df = pd.read_csv(self.path)

        col_pitch = 'event' if 'event' in df.columns else 'pitch'
        col_vel = 'velocity'
        col_dur = 'duration'
        col_trans = 'transition'
        col_abs_tick = 'absolute_tick'

        cleaned_data = []
        # Maps the index of each sounding note to its original CSV row
        self.note_to_csv_map = []

        for idx, row in df.iterrows():
            raw_val = str(row[col_pitch]).strip().lower()

            # Pauses are skipped for the melodic Markov chain
            if raw_val == 'pause':
                continue

            p_clean = self._clean_val(row[col_pitch])
            if p_clean is None:
                continue

            temp_vel = self._clean_val(row[col_vel]) if col_vel in df.columns else 90
            temp_dur = self._clean_val(row[col_dur]) if col_dur in df.columns else 1.0
            temp_trans = self._clean_val(row[col_trans]) if col_trans in df.columns else 1.0

            note_info = {
                'pitch': int(p_clean),
                'velocity': int(temp_vel) if temp_vel is not None else 90,
                'duration': float(temp_dur) if temp_dur is not None else 1.0,
                'transition': float(temp_trans) if temp_trans is not None else 1.0,
                'csv_row_index': idx
            }

            if col_abs_tick in df.columns:
                val_abs = self._clean_val(row[col_abs_tick])
                if val_abs is not None:
                    note_info['absolute_tick'] = float(val_abs)

            cleaned_data.append(note_info)
            self.note_to_csv_map.append(idx)

        if len(cleaned_data) < 2:
            raise ValueError("Insufficient data in the CSV file for the analysis (at least 2 notes required).")

        self.raw_notes_data = cleaned_data

        # --- ALPHABET PARAMETERS ---
        pitches = np.array([n['pitch'] for n in cleaned_data])
        p_min, p_max = pitches.min(), pitches.max()

        self.octaves = math.ceil((p_max - p_min) / 12)
        if self.octaves == 0: self.octaves = 1

        self.alphabet_size = 12 * self.octaves
        self.base_midi = (p_min // 12) * 12

        # Pitch-class sequence
        self.notes = [n['pitch'] % 12 for n in cleaned_data]

        # Interval sequence (transposition-invariant representation)
        self.intervals = [
            cleaned_data[i+1]['pitch'] - cleaned_data[i]['pitch']
            for i in range(len(cleaned_data)-1)
        ]

        print(f"[INFO] Loading completed: {len(cleaned_data)} notes processed.")
        print(f"[INFO] Alphabet set: {self.alphabet_size} notes ({self.octaves} octaves).")

    def _clean_val(self, val):
        """
        Extracts the plain number from composite strings.
        Handles formats such as:
        - '64 (p)' -> 64.0
        - '1.497 (~9/8)' -> 1.497
        - '0.5 (1/2)' -> 0.5
        - 127 -> 127.0
        """
        if pd.isna(val):
            return None

        s_val = str(val).strip()

        if '(' in s_val:
            s_val = s_val.split('(')[0]

        s_val = s_val.replace('~', '').strip()

        try:
            return float(s_val)
        except ValueError:
            return None

    def build_transition_counts(self, sequence, k):
        """Builds the transition counts for a given order k."""
        counts = {}
        for i in range(len(sequence) - k):
            ctx = tuple(sequence[i:i+k])
            tar = sequence[i+k]
            if ctx not in counts:
                counts[ctx] = {}
            counts[ctx][tar] = counts[ctx].get(tar, 0) + 1
        return counts

    def find_optimal_k(self, max_k=10):
        """
        Finds the optimal Markov order through the Past-Future Mutual
        Information (PFMI), selected at the plateau of the curve.
        """
        seq = self.intervals
        n = len(seq)

        def calculate_entropy(data_list):
            if not data_list: return 0
            counts = Counter(data_list)
            total = sum(counts.values())
            probs = [c/total for c in counts.values()]
            return -sum(p * math.log2(p) for p in probs if p > 0)

        # H(next): entropy of the single symbol (unigram)
        h_next = calculate_entropy(seq)

        k_results = {}
        print(f"[PROCESS] Computing Mutual Information for k=1..{max_k}...")

        for k in range(1, max_k + 1):
            histories = [tuple(seq[i:i+k]) for i in range(n - k)]
            joints = [tuple(seq[i:i+k+1]) for i in range(n - k)]

            h_hist = calculate_entropy(histories)
            h_joint = calculate_entropy(joints)

            # I(k) = H(Past) + H(Future) - H(Past, Future)
            mi = h_hist + h_next - h_joint
            k_results[k] = mi

        # Plateau (elbow) criterion
        self.best_k = 1
        for k in range(2, max_k + 1):
            incremento = k_results[k] - k_results[k-1]
            if incremento > 0.03:
                self.best_k = k
            else:
                break

        print(f"[INFO] Optimal order (MI plateau): k={self.best_k} (MI: {k_results[self.best_k]:.4f} bits)")
        return k_results

    def _is_alternating(self, pattern):
        """
        Discards patterns with low internal variability
        (e.g. ABAB, AAABAA, [1, -1, -1, 1]).
        """
        if len(pattern) < 3:
            return False

        unique_intervals = set(pattern)

        if len(unique_intervals) <= 2:
            if len(pattern) >= 5:
                return True

        # Pure alternation (ABAB...)
        evens = pattern[0::2]
        odds = pattern[1::2]
        if len(set(evens)) == 1 and len(set(odds)) == 1:
            return True

        # Micro-cyclicity (identical halves)
        half = len(pattern) // 2
        if len(pattern) >= 4 and pattern[:half] == pattern[half:]:
            return True

        return False

    def plot_markovian_surprise(self, output_name="markovian_surprise.png"):
        """
        Generates the plot of the Markovian surprise S(t).
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import mode
        from scipy.ndimage import gaussian_filter1d

        n_notes = len(self.raw_notes_data)
        time_norm = np.linspace(0, 1, n_notes)

        s_temporal_total = np.zeros(n_notes)
        s_variability_steps = np.zeros(n_notes)
        individual_temporal_curves = []

        # --- PART 1: TEMPORAL SURPRISE (S_t) ---
        for p in self.top_patterns:
            indices = sorted(p['indices'])
            p_curve = np.zeros(n_notes)
            if len(indices) > 2:
                deltas = np.diff(indices)
                m_res = mode(deltas, keepdims=True)
                tau_hat = m_res.mode[0] if m_res.mode.size > 0 else 0
                med = np.median(deltas)
                sigma = np.median(np.abs(deltas - med)) + 1e-5
                raw_surprises = [math.log1p(abs(d - tau_hat) / sigma) for d in deltas]
                mean_s = np.mean(raw_surprises)
                for i in range(len(deltas)):
                    p_curve[indices[i]:indices[i+1]] = raw_surprises[i] - mean_s
            individual_temporal_curves.append(p_curve)
            s_temporal_total += p_curve

        # --- PART 2: VARIABILITY SURPRISE (S_v) ---
        all_phi = np.array([p['importance'] for p in self.top_patterns])
        phi_max = np.max(all_phi) if len(all_phi) > 0 else 1.0
        soglia_relativa, k_pendenza = 0.20, 12

        def calculate_structural_weight(phi):
            rel_p = phi / phi_max
            return 1 / (1 + np.exp(-k_pendenza * (rel_p - soglia_relativa)))

        raw_weights = np.array([calculate_structural_weight(p['importance']) for p in self.top_patterns])
        w_max_observed = np.max(raw_weights) if np.max(raw_weights) > 0 else 1.0
        final_weights = (raw_weights / w_max_observed) * 100

        for i, p in enumerate(self.top_patterns):
            p_len = len(p['pattern']) + 1
            weight = final_weights[i]
            if weight < 0.5: continue
            for idx in p['indices']:
                s_variability_steps[idx : idx + p_len] = np.maximum(s_variability_steps[idx : idx + p_len], weight)

        # --- PART 3: S(t) AND ENGAGEMENT (C) ---
        s_raw_total = s_temporal_total + s_variability_steps
        s_final_smooth = gaussian_filter1d(s_raw_total, sigma=3)
        C_val = np.mean(np.abs(s_final_smooth))
        self.average_cognitive_energy = C_val

        # --- VISUALIZATION ---
        fig, axes = plt.subplots(4, 1, figsize=(12, 20), sharex=True)

        # 1. Individual temporal components
        for i, curve in enumerate(individual_temporal_curves[:10]):
            axes[0].plot(time_norm, curve, alpha=0.6)
        axes[0].set_title("Individual temporal components ($S_t(p_i)$)", fontweight='normal')
        axes[0].set_ylabel(r"Intensity ($\phi$)")

        # 2. Temporal surprise
        axes[1].plot(time_norm, s_temporal_total, color='#e67e22', linewidth=1.5)
        axes[1].axhline(0, color='black', linewidth=0.5, linestyle='--')
        axes[1].set_title("Temporal surprise ($S_t(t)$)", fontweight='normal')
        axes[1].set_ylabel(r"Intensity ($\phi$)")

        # 3. Variability surprise
        axes[2].plot(time_norm, s_variability_steps, color='#3498db', linewidth=2, drawstyle='steps-post')
        axes[2].set_title("Variability surprise ($S_v(t)$)", fontweight='normal')
        axes[2].set_ylabel(r"Intensity ($\phi$)")
        axes[2].set_ylim(-5, 110)

        # 4. Total surprise S(t) and C
        axes[3].plot(time_norm, s_final_smooth, color='black', linewidth=1.5, label="$S(t)$")
        axes[3].axhline(C_val, color='black', linewidth=1.5, linestyle='--', label=f"C = {C_val:.4f}")
        axes[3].axhline(0, color='red', linewidth=0.5)
        axes[3].set_title(f"Total surprise ($S(t)$) and engagement (C = {C_val:.4f})", fontweight='normal')
        axes[3].set_xlabel("Relative time of the composition")
        axes[3].set_ylabel(r"Intensity ($\phi$)")
        axes[3].set_xlim(0, 1)

        axes[3].legend(loc='upper right', frameon=True, fontsize='small')

        for ax in axes:
            ax.grid(True, alpha=0.2)
            ax.margins(y=0.15)

        plt.subplots_adjust(hspace=0.5, bottom=0.12, top=0.92, left=0.1, right=0.95)

        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] S(t) report generated: {output_name}")
        plt.show()

        return C_val

    def build_transition_tensor(self, k):
        """
        Builds a SPARSE representation of the transition tensor.
        Implements the local convergence count (ci) over subspaces:
        ci_{i-1 -> i} counts how many different past histories (j)
        led to the final transition (i-1 -> i).
        """
        seq = np.array(self.intervals)
        n = len(seq)

        self.tensor = {}
        self.context_totals = {}

        # Key:
        #   - if k=1: the target 'i' (how many different notes lead to i)
        #   - if k>1: the pair (i-1, i), i.e. the last observed transition
        # Value: set of "past histories" (context prefixes) j1...jk-1
        self.subspace_convergence = {}

        print(f"[PROCESS] Populating sparse tensor (k={k})...")

        for i in range(n - k):
            full_path = tuple(seq[i : i + k + 1])
            context = full_path[:-1]
            target = full_path[-1]

            # 1. Joint frequencies (sparse tensor)
            self.tensor[full_path] = self.tensor.get(full_path, 0) + 1

            # 2. Context totals (denominator for p_m)
            self.context_totals[context] = self.context_totals.get(context, 0) + 1

            # 3. Local convergence (ci) over subspaces
            if k == 1:
                sub_key = target
                if sub_key not in self.subspace_convergence:
                    self.subspace_convergence[sub_key] = set()
                self.subspace_convergence[sub_key].add(context)
            else:
                pivot_transition = context[-1]
                sub_key = (pivot_transition, target)

                if sub_key not in self.subspace_convergence:
                    self.subspace_convergence[sub_key] = set()

                prefix = context[:-1]
                self.subspace_convergence[sub_key].add(prefix)

        self.ci_subspace_dict = {
            key: len(val)
            for key, val in self.subspace_convergence.items()
        }

        print(f"[INFO] Sparse tensor built: {len(self.tensor)} active transitions.")
        print(f"[INFO] Computed {len(self.ci_subspace_dict)} local convergence subspaces.")

        return self.tensor

    def _get_max_overlap(self, p1, p2):
        """Finds the longest common subsequence between two patterns (min 2 transitions/3 notes)."""
        n1, n2 = len(p1), len(p2)
        max_sub = ()
        for i in range(n1):
            for j in range(i + 3, n1 + 1):
                sub = p1[i:j]
                for k in range(len(p2) - len(sub) + 1):
                    if p2[k:k+len(sub)] == sub:
                        if len(sub) > len(max_sub):
                            max_sub = sub
        return max_sub

    def _calculate_sub_psi(self, sub_seq_intervals):
        """
        Computes psi using the local subspace convergence (ci_{i-1 -> i})
        and the normalization over the total pattern length (L notes).
        """
        k = int(self.best_k)
        n_intervalli = len(sub_seq_intervals)
        # L is the number of notes (number of intervals + 1)
        L = n_intervalli + 1

        num_transizioni_effettive = n_intervalli - k

        # The pattern is too short to be evaluated at the current order k
        if num_transizioni_effettive <= 0:
            return 0

        prod_val = 1.0

        for i in range(k, n_intervalli):
            context = tuple(sub_seq_intervals[i-k : i])
            target = sub_seq_intervals[i]

            full_path = context + (target,)

            # 1. Local probability p_m
            count_trans = self.tensor.get(full_path, 0)
            denom = self.context_totals.get(context, 0)
            p_m = count_trans / denom if denom > 0 else 0

            # 2. Local convergence ci for the subspace
            if k == 1:
                sub_key = target
            else:
                pivot_transition = context[-1]
                sub_key = (pivot_transition, target)

            c_i = self.ci_subspace_dict.get(sub_key, 0)

            # 3. Product accumulation (ci * pm)
            prod_val *= (c_i * p_m)

        if prod_val > 0:
            # L-th root: makes patterns of different lengths and orders comparable
            return math.pow(float(prod_val), 1/L)
        else:
            return 0

    def extract_patterns(self, min_len=4, max_len=12):
        """
        Extracts the significant patterns based on the Phi value (F * Psi).
        Uses note_to_csv_map to correctly handle files with pause rows and
        extra columns.
        """
        k = self.best_k
        min_len = max(min_len, self.best_k + 2)

        if self.tensor is None:
            self.build_transition_tensor(k)

        # 1. CANDIDATE GENERATION FROM INTERVALS
        mapped_seq = tuple(self.intervals)
        n = len(mapped_seq)

        candidates = {}
        # L notes -> L-1 intervals
        for L_int in range(min_len - 1, max_len):
            for i in range(n - L_int + 1):
                pat = mapped_seq[i:i+L_int]

                if all(v == 0 for v in pat):
                    continue

                if self._is_alternating(pat):
                    continue

                if pat not in candidates:
                    psi_orig = self._calculate_sub_psi(pat)
                    if psi_orig > 0:
                        candidates[pat] = {'psi': psi_orig, 'indices': [], 'count': 0}

                if pat in candidates:
                    candidates[pat]['count'] += 1
                    candidates[pat]['indices'].append(i)

        sorted_candidates = []
        for pat, data in candidates.items():
            sorted_candidates.append({
                'pat': pat,
                'psi': data['psi'],
                'count': data['count'],
                'indices': data['indices']
            })

        sorted_candidates.sort(key=lambda x: (x['count'], -len(x['pat'])), reverse=True)

        # 2. GROUPING AND LOGARITHMIC SATURATION
        final_scored = []
        used_as_follower = set()

        for i, leader in enumerate(sorted_candidates):
            if i in used_as_follower:
                continue

            leader_pat = leader['pat']
            somma_psi_overlap = leader['psi']

            for j, follower in enumerate(sorted_candidates):
                if i == j or j in used_as_follower:
                    continue

                overlap = self._get_max_overlap(leader_pat, follower['pat'])

                if len(overlap) >= 2:
                    psi_overlap = self._calculate_sub_psi(overlap)
                    somma_psi_overlap += psi_overlap
                    used_as_follower.add(j)

            # psi_acc = psi_leader + log2(2 + sum of overlapping psi)
            psi_accumulato = leader['psi'] + math.log2(2 + somma_psi_overlap)

            # 3. FINAL PHI (importance = weighted F * saturated strength)
            # Frequency normalized over the total number of notes: (F * 1000 / N)
            n_note_totali = len(self.raw_notes_data)
            f_pesata = (leader['count'] * 1000) / n_note_totali

            phi_finale = f_pesata * psi_accumulato

            real_csv_indices = [self.note_to_csv_map[idx] for idx in leader['indices']]

            final_scored.append({
                'pattern': leader_pat,
                'importance': phi_finale,
                'f_pesata': f_pesata,
                'indices': leader['indices'],
                'csv_indices': real_csv_indices,
                'count': leader['count'],
                'psi_total': psi_accumulato
            })

        # 4. FILTERING AND FINAL RANKING
        final_scored.sort(key=lambda x: x['importance'], reverse=True)

        # Adaptive threshold proportional to the maximum observed Phi
        all_phi_values = [p['importance'] for p in final_scored]
        max_phi_detected = max(all_phi_values) if all_phi_values else 0

        dynamic_threshold = max_phi_detected * 0.02

        self.phi_threshold = dynamic_threshold
        self.top_patterns = [p for p in final_scored if p['importance'] > self.phi_threshold]

        # Fallback: keep at least the best pattern
        if not self.top_patterns and final_scored:
            self.top_patterns = final_scored[:1]

        print(f"\n[PHI RANKING] Extracted {len(self.top_patterns)} patterns above threshold {self.phi_threshold}:")
        if not self.top_patterns:
            print("  [!] No patterns found. Check min_len parameters or threshold.")
        else:
            for idx, p in enumerate(self.top_patterns):
                print(f"  {idx+1}. Phi: {p['importance']:.4f} | Freq: {p['count']} | Pattern (Intervals): {list(p['pattern'])}")

        return self.top_patterns

    def plot_scientific_report(self, output_name="integrated_report.png"):
        """
        UNIFIED PLOT: displays the top patterns on a single-flow timeline.
        Includes a surprise/relief gradient based on the temporal distance
        between occurrences. Handles a variable number of patterns.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        data = self.top_patterns
        if not data:
            print("[ERROR] No patterns to display.")
            return

        fig_height = max(8, len(data) * 1.5)
        fig, ax = plt.subplots(figsize=(16, fig_height))

        cmap = plt.get_cmap('tab10')
        n_patterns = len(data)

        ax.set_ylim(0.5, n_patterns + 0.5)

        for idx, p in enumerate(data):
            y_base = n_patterns - idx
            indices = sorted(p['indices'])
            color = cmap(idx % 10)

            # 1. Horizontal guide line
            ax.axhline(y_base, color='gray', alpha=0.1, zorder=1)

            # 2. Scatter: each occurrence of the pattern on the timeline
            ax.scatter(indices, [y_base]*len(indices), color=color,
                       s=150, zorder=4, edgecolors='white', linewidth=1.5,
                       label=f"P{idx+1}: {list(p['pattern'])}")

            # 3. Tension and temporal surprise visualization
            if len(indices) > 1:
                deltas = np.diff(indices)
                mean_t = np.mean(deltas)

                y_min_span = y_base - 0.4
                y_max_span = y_base + 0.4

                for i in range(len(deltas)):
                    start_x = indices[i]
                    end_x = indices[i+1]

                    diff = abs(deltas[i] - mean_t)

                    alpha_val = min(0.4, diff / (mean_t * 1.5 + 1e-5))

                    ax.fill_between([start_x, end_x], y_min_span, y_max_span,
                                   color=color, alpha=alpha_val, zorder=2)

                    tension = (deltas[i] / (mean_t + 1e-5)) * 0.15
                    ax.plot([start_x, end_x], [y_base-0.25, y_base-0.25 + tension],
                            color=color, alpha=0.6, linewidth=1.2, zorder=3)

        ax.set_yticks(range(1, n_patterns + 1))
        pattern_labels = [f"Pattern {n_patterns-i+1}" for i in range(1, n_patterns+1)][::-1]
        ax.set_yticklabels(pattern_labels, fontweight='bold')

        ax.set_xlabel("Temporal progression (note index in the unified flow)", fontsize=12)
        ax.set_title(f"COGNITIVE ARCHITECTURE: {os.path.basename(self.path).upper()}",
                     fontsize=15, fontweight='black', pad=25)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False,
                  title=fr"Patterns above threshold ($\Phi > {self.phi_threshold:.2f}$)", title_fontproperties={'weight':'bold'})

        plt.tight_layout()
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Report plot generated: {output_name}")
        plt.show()

    def export_patterns_to_midi(self, output_name_base):
        """
        Exports ALL top patterns using note_to_csv_map to retrieve the
        ORIGINAL durations, velocities and note spacings (transition).
        Handles overlaps (polyphony) and names the file with the next
        prime number.
        """
        try:
            from midiutil import MIDIFile
        except ImportError:
            print("[ERROR] Install midiutil: pip install midiutil")
            return

        def is_prime(n):
            if n < 2: return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0: return False
            return True

        def get_next_prime(n):
            candidate = n + 1
            while not is_prime(candidate):
                candidate += 1
            return candidate

        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
        if not os.path.exists(downloads_path):
            os.makedirs(downloads_path)

        clean_base = output_name_base.replace(".mid", "")

        existing_files = [f for f in os.listdir(downloads_path) if f.startswith(clean_base) and f.endswith('.mid')]
        last_prime = 1
        for f in existing_files:
            try:
                parts = f.replace(clean_base, "").replace(".mid", "").split("_")
                num_part = parts[-1] if parts[-1] else "0"
                val = int(num_part)
                if is_prime(val) and val > last_prime:
                    last_prime = val
            except:
                continue

        next_p = get_next_prime(last_prime)
        final_filename = f"{clean_base}_{next_p}.mid"
        full_path = os.path.join(downloads_path, final_filename)

        midi = MIDIFile(1)
        midi.addTempo(0, 0, 120)

        pattern_start_cursor = 0

        for idx, p in enumerate(self.top_patterns):
            start_note_idx = p['indices'][0]
            pattern_len = len(p['pattern']) + 1

            segment_notes = self.raw_notes_data[start_note_idx : start_note_idx + pattern_len]

            # Pitch offset to distinguish the patterns aurally (range 0-127)
            base_listening_pitch = 60 + ((idx % 12) * 4)
            pitch_offset = base_listening_pitch - segment_notes[0]['pitch']

            note_time_cursor = pattern_start_cursor

            for note_data in segment_notes:
                pitch = int(max(0, min(127, note_data['pitch'] + pitch_offset)))

                vel = int(note_data['velocity'])
                dur = float(note_data['duration'])

                trans = float(note_data['transition'])

                # If dur > trans, the note keeps sounding while the next one starts (overlap)
                midi.addNote(0, 0, pitch, note_time_cursor, dur, vel)

                # The cursor advances by the transition, not the duration
                note_time_cursor += trans

            pattern_start_cursor = note_time_cursor + 4

        try:
            with open(full_path, "wb") as output_file:
                midi.writeFile(output_file)
            print(f"[SUCCESS] MIDI ({len(self.top_patterns)} patterns) saved with real polyphony: {full_path}")
        except Exception as e:
            print(f"[ERROR] MIDI writing failed: {e}")


# --- MAIN ---
if __name__ == "__main__":
    print(">>> [SCIENTIFIC ANALYSIS]")
    path_file = input("PATH CSV > ").strip().replace("'", "").replace('"', "")

    downloads_path = "/Users/raph/Downloads"
    nome_base = os.path.basename(path_file).replace(".csv", "")

    try:
        engine = MusicalMarkovEngine(path_file)
        engine.load_and_clean()

        # 1. Markov analysis: optimal memory order of the chain
        engine.find_optimal_k()

        # 2. Pattern discovery (dynamic threshold on Phi)
        print("\n2. Extracting and weighting patterns...")
        engine.extract_patterns()

        # 3. Plots
        print("\n3. Generating reports...")
        report_path = os.path.join(downloads_path, f"Analysis_{nome_base}.png")
        engine.plot_scientific_report(output_name=report_path)

        print("\n4. Generating Markovian surprise plot and computing C...")
        surprise_path = os.path.join(downloads_path, f"S_t_{nome_base}.png")
        s_media = engine.plot_markovian_surprise(output_name=surprise_path)

        # 4. MIDI export
        midi_name = f"Pattern_{nome_base}.mid"
        engine.export_patterns_to_midi(midi_name)

        # --- FINAL RESULTS ---
        print(f"\n{'='*75}")
        print(f"{'AVERAGE COGNITIVE SURPRISE (C)':^75}")
        print(f"{'-'*75}")
        print(f"-> Mean value of S(t): {s_media:.4f}")
        print(f"{'='*75}")

        print(f"\n{'='*75}")
        print(f"{'TOP DETECTED PATTERNS (Phi threshold: ' + f'{engine.phi_threshold:.2f}' + ')':^75}")
        print(f"{'-'*75}")
        print(f"{'PATTERN (INTERVALS)':<35} | {'FREQ':<5} | {'PHI':<10}")
        print(f"{'-'*75}")

        for p in engine.top_patterns:
            print(f"{str(list(p['pattern'])):<35} | {p['count']:<5} | {p['importance']:.4f}")
        print(f"{'='*75}")

        print(f"\nAnalysis completed successfully.")
        print(f"Plots and MIDI saved to: {downloads_path}")

    except Exception as e:
        import traceback
        print(f"\nError during the analysis:")
        traceback.print_exc()
