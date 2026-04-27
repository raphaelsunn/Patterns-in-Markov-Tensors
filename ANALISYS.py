import os
import pandas as pd
import numpy as np
import math
from scipy.linalg import eig
from collections import Counter
from scipy.stats import mode
from scipy.ndimage import gaussian_filter1d


class MusicalMarkovEngine:
    """
    Motore di analisi statistica per sequenze musicali.
    Gestisce la conversione in intervalli e il calcolo delle metriche entropiche.
    """
    
    def __init__(self, path):
        self.path = path
        self.notes = []
        self.intervals = []
        self.best_k = 0
        self.h_chain = 0.0
        self.raw_notes_data = []
        # --- AGGIUNTA PUNTO 1 ---
        self.octaves = 0
        self.alphabet_size = 0
        self.base_midi = 0
        self.tensor = None 
        self.ci_vector = None

    def load_and_clean(self):
        """
        Caricamento e calcolo parametri dell'alfabeto.
        Supporta sia il vecchio formato (Schoenberg) che il nuovo (Rautavaara).
        Ora estrae anche il valore 'transition' per gestire le sovrapposizioni ritmiche.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Percorso non trovato: {self.path}")
            
        df = pd.read_csv(self.path)
        
        # Identificazione dinamica delle colonne
        col_pitch = 'event' if 'event' in df.columns else 'pitch'
        col_vel = 'velocity' 
        col_dur = 'duration'
        col_trans = 'transition' # Colonna fondamentale per la polifonia
        col_abs_tick = 'absolute_tick' 
        
        cleaned_data = []
        # Mappa per collegare l'indice della nota "sonora" alla riga originale del CSV
        self.note_to_csv_map = [] 

        for idx, row in df.iterrows():
            raw_val = str(row[col_pitch]).strip().lower()
            
            # Saltiamo le pause per la catena di Markov melodica
            if raw_val == 'pause': 
                continue
                
            p_clean = self._clean_val(row[col_pitch])
            if p_clean is None: 
                continue 
            
            # Pulizia sicura dei valori numerici per evitare TypeError: NoneType
            # Se il valore pulito è None, assegniamo un default (90 per vel, 1.0 per dur/trans)
            temp_vel = self._clean_val(row[col_vel]) if col_vel in df.columns else 90
            temp_dur = self._clean_val(row[col_dur]) if col_dur in df.columns else 1.0
            temp_trans = self._clean_val(row[col_trans]) if col_trans in df.columns else 1.0

            # Estraiamo i dati della nota sonora assicurandoci che siano float/int validi
            note_info = {
                'pitch': int(p_clean),
                'velocity': int(temp_vel) if temp_vel is not None else 90,
                'duration': float(temp_dur) if temp_dur is not None else 1.0,
                'transition': float(temp_trans) if temp_trans is not None else 1.0,
                'csv_row_index': idx 
            }
            
            # Se esiste la colonna absolute_tick, la salviamo
            if col_abs_tick in df.columns:
                val_abs = self._clean_val(row[col_abs_tick])
                if val_abs is not None:
                    note_info['absolute_tick'] = float(val_abs)
                
            cleaned_data.append(note_info)
            self.note_to_csv_map.append(idx)
        
        if len(cleaned_data) < 2:
            raise ValueError(f"Dati insufficienti nel file CSV per l'analisi (minimo 2 note richieste).")
            
        self.raw_notes_data = cleaned_data
        

        
        
        # --- CALCOLO PARAMETRI ALFABETO ---
        pitches = np.array([n['pitch'] for n in cleaned_data])
        p_min, p_max = pitches.min(), pitches.max()
        
        # Calcolo ottave 'o'
        self.octaves = math.ceil((p_max - p_min) / 12)
        if self.octaves == 0: self.octaves = 1
        
        # Dimensione Alfabeto N
        self.alphabet_size = 12 * self.octaves
        # Base per la mappatura
        self.base_midi = (p_min // 12) * 12
        
        # Sequenza delle classi di altezza (Pitch Class) per analisi armonica
        self.notes = [n['pitch'] % 12 for n in cleaned_data]
        
        # Generazione della sequenza di intervalli (il cuore del tensore sparso)
        # Gli intervalli ignorano le pause temporali, misurando solo il salto melodico
        self.intervals = [
            cleaned_data[i+1]['pitch'] - cleaned_data[i]['pitch'] 
            for i in range(len(cleaned_data)-1)
        ]
        
        print(f"[INFO] Caricamento completato: {len(cleaned_data)} note processate.")
        print(f"[INFO] Alfabeto impostato: {self.alphabet_size} note ({self.octaves} ottave).")
        
    def _clean_val(self, val):
        """
        Estrae il numero puro da stringhe complesse.
        Gestisce formati come:
        - '64 (p)' -> 64.0
        - '1.497 (~9/8)' -> 1.497
        - '0.5 (1/2)' -> 0.5
        - 127 -> 127.0
        """
        if pd.isna(val): 
            return None
        
        # Converte in stringa e pulisce spazi bianchi
        s_val = str(val).strip()
        
        # Se la stringa contiene una parentesi (es. "1.497 (~9/8)"), 
        # prendiamo solo la parte prima della parentesi
        if '(' in s_val:
            s_val = s_val.split('(')[0]
            
        # Rimuove il simbolo di approssimazione '~' se presente
        s_val = s_val.replace('~', '').strip()
        
        try:
            # Tenta la conversione finale in float per i calcoli matematici
            return float(s_val)
        except ValueError:
            # Se non è un numero (es. la stringa 'pause'), restituisce None
            return None

    def build_transition_counts(self, sequence, k):
        """Costruisce i conteggi delle transizioni per un ordine k specifico."""
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
        Trova l'ordine Markoviano ottimale tramite Past-Future Mutual Information (PFMI).
        Sostituisce la Cross-Entropy per una misura pura dell'informazione.
        """
        seq = self.intervals
        n = len(seq)
        
        def calculate_entropy(data_list):
            if not data_list: return 0
            counts = Counter(data_list)
            total = sum(counts.values())
            probs = [c/total for c in counts.values()]
            return -sum(p * math.log2(p) for p in probs if p > 0)

        # H(next): Entropia della singola nota (unigramma)
        h_next = calculate_entropy(seq)
        
        k_results = {}
        print(f"[PROCESS] Calcolo Mutua Informazione per k=1..{max_k}...")
        
        for k in range(1, max_k + 1):
            # Definiamo passato (k note) e congiunta (k+1 note)
            histories = [tuple(seq[i:i+k]) for i in range(n - k)]
            joints = [tuple(seq[i:i+k+1]) for i in range(n - k)]
            
            h_hist = calculate_entropy(histories)
            h_joint = calculate_entropy(joints)
            
            # Formula: I(k) = H(Past) + H(Future) - H(Past, Future)
            mi = h_hist + h_next - h_joint
            k_results[k] = mi

        # Logica del plateau (Gomito): 
        # Troviamo il punto dove l'incremento di informazione diventa marginale
        self.best_k = 1
        for k in range(2, max_k + 1):
            incremento = k_results[k] - k_results[k-1]
            if incremento > 0.03: # Soglia di 0.05 bit (regolabile)
                self.best_k = k
            else:
                break
        
        print(f"[INFO] Ordine ottimale (Plateau MI): k={self.best_k} (MI: {k_results[self.best_k]:.4f} bits)")
        return k_results
    
    def _is_alternating(self, pattern):
        """
        Versione potenziata: scarta pattern con scarsa variabilità interna.
        Scarta: ABAB, AAABAA, [1, -1, -1, 1] ecc.
        """
        if len(pattern) < 3:
            return False
            
        # 1. Conta quanti intervalli unici ci sono
        unique_intervals = set(pattern)
        
        # Se il pattern usa solo 2 intervalli (es. solo 1 e -1) ed è corto, 
        # spesso è solo rumore meccanico o un trillo sporco.
        if len(unique_intervals) <= 2:
            # Se il pattern è lungo (es. 6+) e usa solo 2 intervalli, è quasi certamente un ciclo
            if len(pattern) >= 5:
                return True
        
        # 2. Controllo alternanza pura (ABAB...)
        evens = pattern[0::2]
        odds = pattern[1::2]
        if len(set(evens)) == 1 and len(set(odds)) == 1:
            return True
            
        # 3. Controllo micro-ciclicità (metà identiche)
        half = len(pattern) // 2
        if len(pattern) >= 4 and pattern[:half] == pattern[half:]:
            return True

        return False

    def compute_stationary_entropy(self):
        """
        Calcola l'Entropia della Catena (H) risolvendo la distribuzione stazionaria pi.
        Valido per l'ordine k calcolato.
        """
        k = self.best_k
        seq = self.intervals
        counts = self.build_transition_counts(seq, k)
        
        states = sorted(list(counts.keys()))
        state_to_idx = {s: i for i, s in enumerate(states)}
        n = len(states)
        
        # Matrice di transizione tra stati
        P = np.zeros((n, n))
        for i, s in enumerate(states):
            c_dict = counts[s]
            total = sum(c_dict.values())
            for note, c in c_dict.items():
                # Il prossimo stato è lo shift del contesto + la nuova nota
                next_s = s[1:] + (note,) if k > 0 else ()
                if next_s in state_to_idx:
                    P[i, state_to_idx[next_s]] = c / total
        
        # Risoluzione pi P = pi
        if n > 1:
            try:
                vals, vecs = eig(P, left=True, right=False)
                idx_1 = np.argmin(np.abs(vals - 1.0))
                pi = np.real(vecs[:, idx_1])
                pi = pi / np.sum(pi)
            except:
                pi = np.ones(n) / n
        else:
            pi = np.array([1.0])
            
        # Calcolo H = - sum pi_i * sum p_ij log p_ij
        h_chain = 0.0
        for i, s in enumerate(states):
            c_dict = counts[s]
            total = sum(c_dict.values())
            entropy_i = -sum((c/total) * math.log2(c/total) for c in c_dict.values())
            h_chain += pi[i] * entropy_i
            
        self.h_chain = h_chain
        return h_chain
    
    def plot_dynamic_thermodynamics(self, output_name="termodinamica_informativa.png"):
        """
        Calcola H(t), R(t) e le loro derivate dH/dt, dR/dt.
        Usa una finestra mobile per catturare l'evoluzione dell'entropia termodinamica.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Costanti fisiche
        kb = 1.380649e-23  # Costante di Boltzmann (J/K)
        N = self.alphabet_size
        if N <= 1: N = 12 # Fallback
        h_max = math.log(N) # Entropia massima teorica

        seq = self.intervals
        n_total = len(seq)
        
        # Dimensione della finestra (es. 30 note per avere una statistica locale stabile)
        window_size = 30
        step = 1 # Calcolo punto per punto per derivate precise
        
        times = []
        h_values = []
        r_values = []

        # 1. CALCOLO H(t) e R(t)
        for i in range(0, n_total - window_size, step):
            window = seq[i : i + window_size]
            counts = Counter(window)
            total = sum(counts.values())
            
            # Calcolo Entropia di Shannon locale: H = - sum p * ln(p)
            # Moltiplicata per kb per la scala termodinamica
            h_local_shannon = 0.0
            for c in counts.values():
                p = c / total
                h_local_shannon -= p * math.log(p)
            
            h_kb = kb * h_local_shannon
            
            # Calcolo Ridondanza locale: R = 1 - (H / H_max)
            # Usiamo h_local_shannon per il rapporto (kb si cancellerebbe)
            r_local = 1 - (h_local_shannon / h_max) if h_max > 0 else 0
            
            times.append(i + window_size)
            h_values.append(h_kb)
            r_values.append(r_local)

        times = np.array(times)
        h_values = np.array(h_values)
        r_values = np.array(r_values)

        # 2. CALCOLO DERIVATE (dH/dt e dR/dt)
        # Usiamo il gradiente numerico (differenze finite)
        dh_dt = np.gradient(h_values)
        dr_dt = np.gradient(r_values)

        # 3. VISUALIZZAZIONE
        fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

        # Pannello superiore: dH/dt (Variazione del disordine)
        # Moltiplichiamo dH/dt per un fattore di scala visivo se kb lo rende troppo piccolo
        axes[0].plot(times, dh_dt, color='purple', linewidth=1.5, label="dH/dt (Variazione Entropia)")
        axes[0].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[0].fill_between(times, dh_dt, 0, where=(dh_dt > 0), color='purple', alpha=0.1)
        axes[0].set_title(f"DINAMICA DELL'ENTROPIA TERMODINAMICA ($k_b$ scale)", fontweight='bold')
        axes[0].set_ylabel("dH/dt (J/K per nota)")
        axes[0].legend()

        # Pannello inferiore: dR/dt (Variazione della coerenza strutturale)
        axes[1].plot(times, dr_dt, color='teal', linewidth=1.5, label="dR/dt (Variazione Ridondanza)")
        axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1].fill_between(times, dr_dt, 0, where=(dr_dt > 0), color='teal', alpha=0.1)
        axes[1].set_title("DINAMICA DELLA RIDONDANZA (COERENZA)", fontweight='bold')
        axes[1].set_xlabel("Progressione Temporale (Indice Nota)")
        axes[1].set_ylabel("dR/dt")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_name, dpi=300)
        print(f"[SUCCESS] Grafico dH/dt e dR/dt generato: {output_name}")
        plt.show()

        # Restituiamo i valori medi delle derivate per il terminale
        return np.mean(np.abs(dh_dt)), np.mean(np.abs(dr_dt))

    def plot_markovian_surprise(self, output_name="sorpresa_markoviana.png"):
        """
        Genera il grafico della Sorpresa Markoviana S(t) in formato quadrato (1:1).
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import mode
        from scipy.ndimage import gaussian_filter1d

        # --- SETUP DATI E ASSE TEMPORALE ---
        n_notes = len(self.raw_notes_data)
        time_norm = np.linspace(0, 1, n_notes) 
        
        s_temporal_total = np.zeros(n_notes)
        s_variability_steps = np.zeros(n_notes)
        individual_temporal_curves = []

        # --- PARTE 1: CALCOLO SORPRESA TEMPORALE (S_t) ---
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

        # --- PARTE 2: CALCOLO SORPRESA DI VARIABILITÀ (S_v) ---
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

        # --- PARTE 3: CALCOLO S(t) E COINVOLGIMENTO (C) ---
        s_raw_total = s_temporal_total + s_variability_steps
        s_final_smooth = gaussian_filter1d(s_raw_total, sigma=3)
        C_val = np.mean(np.abs(s_final_smooth))
        self.average_cognitive_energy = C_val

        # --- VISUALIZZAZIONE (FORMATO QUADRATO 1:1) ---
        # Impostiamo figsize con valori uguali per il formato quadrato
        # Aumentando il primo valore (20) e diminuendo il secondo (12), 
        # i grafici si allungano orizzontalmente.
        fig, axes = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
        
        # 1. Componenti temporali individuali
        for i, curve in enumerate(individual_temporal_curves[:10]):
            axes[0].plot(time_norm, curve, alpha=0.6)
        axes[0].set_title("Componenti temporali individuali ($S_t(p_i)$)", fontweight='normal')
        axes[0].set_ylabel(r"Intensità ($\phi$)")

        # 2. Sorpresa temporale
        axes[1].plot(time_norm, s_temporal_total, color='#e67e22', linewidth=1.5)
        axes[1].axhline(0, color='black', linewidth=0.5, linestyle='--')
        axes[1].set_title("Sorpresa temporale ($S_t(t)$)", fontweight='normal')
        axes[1].set_ylabel(r"Intensità ($\phi$)")

        # 3. Sorpresa di variabilità
        axes[2].plot(time_norm, s_variability_steps, color='#3498db', linewidth=2, drawstyle='steps-post')
        axes[2].set_title("Sorpresa di variabilità $S_v(t)$", fontweight='normal')
        axes[2].set_ylabel(r"Intensità ($\phi$)")
        axes[2].set_ylim(-5, 110) 

        # 4. Sorpresa totale S(t) e C
        axes[3].plot(time_norm, s_final_smooth, color='black', linewidth=1.5, label="$S(t)$")
        axes[3].axhline(C_val, color='black', linewidth=1.5, linestyle='--', label=f"C = {C_val:.4f}")
        axes[3].axhline(0, color='red', linewidth=0.5)
        axes[3].set_title(f"Sorpresa totale ($S(t)$) e Coinvolgimento (C = {C_val:.4f})", fontweight='normal')
        axes[3].set_xlabel("Tempo relativo della composizione (s)")
        axes[3].set_ylabel(r"Intensità ($\phi$)")
        axes[3].set_xlim(0, 1)
        
        # --- REINSERIMENTO LEGENDA ---
        axes[3].legend(loc='upper right', frameon=True, fontsize='small')

        # --- REGOLE ESTETICHE E LAYOUT QUADRATO ---
        for ax in axes:
            ax.grid(True, alpha=0.2)
            ax.margins(y=0.15) 

        # Regolazione spazi per formato quadrato (hspace ridotto per non comprimere troppo i grafici)
        plt.subplots_adjust(hspace=0.5, bottom=0.12, top=0.92, left=0.1, right=0.95)
        
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Report S(t) quadrato generato: {output_name}")
        plt.show()

        return C_val
    
    def build_transition_tensor(self, k):
        """
        Costruisce una rappresentazione SPARSA del tensore delle transizioni.
        Implementa il calcolo della convergenza locale (ci) per sottospazi.
        Logica: ci_{i-1 -> i} = conta quante diverse storie passate (j) 
        hanno portato alla transizione finale (i-1 -> i).
        """
        seq = np.array(self.intervals)
        n = len(seq)
        
        # Inizializzazione tensore e totali contesto
        self.tensor = {} 
        self.context_totals = {}
        
        # Struttura per il calcolo della convergenza locale (ci) nel sottospazio
        # Chiave: 
        #   - se k=1: il target 'i' (per contare quante note diverse portano a i)
        #   - se k>1: la coppia (i-1, i) ovvero l'ultima transizione osservata
        # Valore: set di "storie passate" (prefissi del contesto) j1...jk-1
        self.subspace_convergence = {}

        print(f"[PROCESS] Popolamento tensore sparso (k={k})...")

        for i in range(n - k):
            # Coordinate: contesto (j1...jk-1, i-1) -> target (i)
            full_path = tuple(seq[i : i + k + 1])
            context = full_path[:-1]
            target = full_path[-1]

            # 1. Aggiorna Frequenze congiunte (Tensore Sparso)
            self.tensor[full_path] = self.tensor.get(full_path, 0) + 1

            # 2. Aggiorna Totali Contesto (denominatore per p_m)
            self.context_totals[context] = self.context_totals.get(context, 0) + 1

            # 3. Calcolo Convergenza Locale (ci) per sottospazi
            # Implementazione della formula: fissiamo l'ultima transizione e variamo il passato
            if k == 1:
                # Per k=1, il sottospazio è definito solo dal target 'i'
                # Contiamo quanti diversi predecessori (j) portano a i
                sub_key = target
                # Aggiungiamo l'intero contesto (la nota singola j)
                if sub_key not in self.subspace_convergence:
                    self.subspace_convergence[sub_key] = set()
                self.subspace_convergence[sub_key].add(context)
            else:
                # Per k>1, il sottospazio è definito dalla transizione i-1 -> i
                # Dove context[-1] è i-1 e target è i
                pivot_transition = context[-1]
                sub_key = (pivot_transition, target)
                
                if sub_key not in self.subspace_convergence:
                    self.subspace_convergence[sub_key] = set()
                
                # Identifichiamo la "storia passata" (prefisso j1...jk-1)
                prefix = context[:-1]
                # Aggiungiamo al set: contiamo solo le celle non nulle (osservate)
                self.subspace_convergence[sub_key].add(prefix)

        # Trasformiamo i set in conteggi numerici (la nostra ci locale)
        self.ci_subspace_dict = {
            key: len(val) 
            for key, val in self.subspace_convergence.items()
        }

        print(f"[INFO] Tensore sparso creato: {len(self.tensor)} transizioni attive.")
        print(f"[INFO] Calcolati {len(self.ci_subspace_dict)} sottospazi di convergenza locale.")
        
        return self.tensor
    
    def _get_max_overlap(self, p1, p2):
        """Trova la più lunga sottosequenza comune tra due pattern (min 2 transizioni/3 note)."""
        n1, n2 = len(p1), len(p2)
        max_sub = ()
        # Cerchiamo sottosequenze a partire da lunghezza 3 (2 transizioni)
        for i in range(n1):
            for j in range(i + 3, n1 + 1): 
                sub = p1[i:j]
                # Controllo se 'sub' esiste in p2
                for k in range(len(p2) - len(sub) + 1):
                    if p2[k:k+len(sub)] == sub:
                        if len(sub) > len(max_sub):
                            max_sub = sub
        return max_sub
    
    def _calculate_sub_psi(self, sub_seq_intervals):
        """
        Calcola psi usando la convergenza locale per sottospazi (ci_{i-1 -> i})
        e la normalizzazione sulla lunghezza totale del pattern (L note).
        """
        k = int(self.best_k)
        n_intervalli = len(sub_seq_intervals)
        # L è il numero di note (numero intervalli + 1)
        L = n_intervalli + 1
        
        # Le transizioni valutabili con un passato di ordine k sono (n_intervalli - k)
        num_transizioni_effettive = n_intervalli - k
        
        # Se il pattern è troppo corto per l'ordine k attuale, non è valutabile
        if num_transizioni_effettive <= 0: 
            return 0
        
        prod_val = 1.0
        
        # Ciclo attraverso le transizioni del pattern partendo dal primo target con storia k
        for i in range(k, n_intervalli):
            # Contesto di lunghezza k (j1, ..., jk-1, i-1)
            context = tuple(sub_seq_intervals[i-k : i])
            # Target (i)
            target = sub_seq_intervals[i]
            
            # Percorso completo nel tensore
            full_path = context + (target,)
            
            # 1. Recupero Probabilità locale p_m
            count_trans = self.tensor.get(full_path, 0)
            denom = self.context_totals.get(context, 0)
            p_m = count_trans / denom if denom > 0 else 0
            
            # 2. Recupero Convergenza Locale ci per sottospazio
            # Implementazione della formula: ci_{i-1 -> i}
            # Se k=1 la chiave è solo il target, se k>1 è (i-1, i)
            if k == 1:
                sub_key = target
            else:
                pivot_transition = context[-1] # Questo è i-1
                sub_key = (pivot_transition, target)
            
            c_i = self.ci_subspace_dict.get(sub_key, 0)
            
            # 3. Accumulo del prodotto energetico (ci * pm)
            prod_val *= (c_i * p_m)
            
        if prod_val > 0:
            # --- NORMALIZZAZIONE L-ESIMA ---
            # Eleviamo alla potenza di 1/L (numero di note) per rendere 
            # confrontabili pattern di lunghezze e ordini diversi.
            return math.pow(float(prod_val), 1/L)
        else:
            return 0

    def extract_patterns(self, min_len=4, max_len=12):
        """
        Estrae i pattern significativi basandosi sul valore Phi (F * Psi).
        Utilizza la mappatura note_to_csv_map per gestire correttamente i file 
        con righe pause e colonne extra.
        """
        k = self.best_k
        # Il pattern deve essere lungo almeno k+2 note per avere senso statistico
        min_len = max(min_len, self.best_k + 2)
        
        if self.tensor is None: 
            self.build_transition_tensor(k)
        
        # 1. GENERAZIONE CANDIDATI DAGLI INTERVALLI
        # Lavoriamo sulla sequenza pura di intervalli (invarianza di trasposizione)
        mapped_seq = tuple(self.intervals)
        n = len(mapped_seq)
        
        candidates = {}
        # L note -> L-1 intervalli
        for L_int in range(min_len - 1, max_len): 
            for i in range(n - L_int + 1):
                pat = mapped_seq[i:i+L_int]
                
                # Salta pattern composti solo da note ribattute (intervallo 0)
                if all(v == 0 for v in pat): 
                    continue 

                if self._is_alternating(pat):
                    continue
                
                if pat not in candidates:
                    # Calcolo psi originale (Media geometrica delle transizioni)
                    psi_orig = self._calculate_sub_psi(pat)
                    if psi_orig > 0:
                        # Utilizziamo la mappatura per salvare gli indici REALI nel CSV
                        candidates[pat] = {'psi': psi_orig, 'indices': [], 'count': 0}
                
                if pat in candidates:
                    candidates[pat]['count'] += 1
                    # Salviamo l'indice della nota nel flusso melodico
                    candidates[pat]['indices'].append(i)

        # Trasformiamo in lista e ordiniamo per PSI decrescente per definire i Leader
        sorted_candidates = []
        for pat, data in candidates.items():
            sorted_candidates.append({
                'pat': pat,
                'psi': data['psi'],
                'count': data['count'],
                'indices': data['indices']
            })
        
        # Ordinamento primario per forza statistica (Psi)
        sorted_candidates.sort(key=lambda x: (x['count'], -len(x['pat'])), reverse=True)

        # 2. ACCORPAMENTO E CALCOLO CON SATURAZIONE LOGARITMICA
        final_scored = []
        used_as_follower = set()

        for i, leader in enumerate(sorted_candidates):
            if i in used_as_follower: 
                continue
            
            leader_pat = leader['pat']
            # Inizializziamo la somma delle psi partendo da quella del leader (più frequente)
            somma_psi_overlap = leader['psi']
            
            for j, follower in enumerate(sorted_candidates):
                if i == j or j in used_as_follower: 
                    continue
                
                # Cerca sovrapposizione significativa
                overlap = self._get_max_overlap(leader_pat, follower['pat'])
                
                if len(overlap) >= 2: 
                    # Calcola il contributo psi della parte sovrapposta (radice L-esima già inclusa)
                    psi_overlap = self._calculate_sub_psi(overlap)
                    somma_psi_overlap += psi_overlap
                    used_as_follower.add(j)
            
            # --- APPLICAZIONE FORMULA SATURAZIONE LOGARITMICA ---
            # psi_acc = psi_leader + ln(1 + somma_psi_overlap)
            # Il logaritmo naturale (math.log) "raffredda" l'esplosione di Phi:
            # i primi follower pesano molto, i successivi sempre meno.
            psi_accumulato = leader['psi'] + math.log2(2 + somma_psi_overlap)
            
            # 3. DEFINIZIONE PHI FINALE (Importanza = F pesata * Forza saturata)
            # Normalizziamo la frequenza (count) sul numero totale di note del brano
            # Formula richiesta: (F * 1000 / N_note) * Psi_accumulato
            n_note_totali = len(self.raw_notes_data)
            f_pesata = (leader['count'] * 1000) / n_note_totali
            
            phi_finale = f_pesata * psi_accumulato
            
            # Mappiamo gli indici melodici agli indici reali del CSV per il report
            real_csv_indices = [self.note_to_csv_map[idx] for idx in leader['indices']]
            
            final_scored.append({
                'pattern': leader_pat,
                'importance': phi_finale,
                'f_pesata': f_pesata, # Opzionale: per debug
                'indices': leader['indices'],
                'csv_indices': real_csv_indices,
                'count': leader['count'],
                'psi_total': psi_accumulato
            })

        # 4. FILTRAGGIO E RANKING FINALE
        # Ordiniamo per Phi decrescente
        final_scored.sort(key=lambda x: x['importance'], reverse=True)

        # --- MODIFICA SOGLIA DINAMICA ADATTIVA ---
        # 1. Troviamo il Phi massimo tra tutti i candidati per decidere la sensibilità
        all_phi_values = [p['importance'] for p in final_scored]
        max_phi_detected = max(all_phi_values) if all_phi_values else 0
        
        # 2. Logica Adattiva: la soglia scala con il massimo, ma resta nel range [3.0, 10.0]
        # Esempio: se max_phi è 500 (Raut), la soglia tende a 10. Se è 50 (Sch), tende a 3.
        #dynamic_threshold = max(3, min(max_phi_detected * 0.02, 10))
        dynamic_threshold = max_phi_detected * 0.02

        self.phi_threshold = dynamic_threshold
        self.top_patterns = [p for p in final_scored if p['importance'] > self.phi_threshold]
        
        # Fallback: se nessun pattern supera 5, prendiamo almeno il migliore per non restituire nulla
        if not self.top_patterns and final_scored:
            self.top_patterns = final_scored[:1]

        # Debugging e Output a console
        print(f"\n[RANKING PHI] Estratti {len(self.top_patterns)} pattern sopra soglia {self.phi_threshold}:")
        if not self.top_patterns:
            print("  [!] Nessun pattern trovato. Controllare parametri min_len o soglia.")
        else:
            for idx, p in enumerate(self.top_patterns):
                print(f"  {idx+1}. Phi: {p['importance']:.4f} | Freq: {p['count']} | Pattern (Intervalli): {list(p['pattern'])}")

        return self.top_patterns

    

    
    def plot_scientific_report(self, output_name="report_integrato.png"):
        """
        GRAFICO UNIFICATO: Visualizza i top pattern su una timeline a flusso unico.
        Include gradiente di sorpresa/sollievo basato sulla distanza temporale tra occorrenze.
        Versione dinamica per numero variabile di pattern (Phi > 5).
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        data = self.top_patterns
        if not data: 
            print("[ERROR] Nessun pattern da visualizzare.")
            return

        # --- MODIFICA DINAMICA ALTEZZA ---
        # L'altezza si adatta al numero di pattern estratti (minimo 8 pollici)
        fig_height = max(8, len(data) * 1.5)
        fig, ax = plt.subplots(figsize=(16, fig_height))
        
        # --- MODIFICA DINAMICA COLORI ---
        # Usiamo tab10 che gestisce bene molti pattern ciclando i colori
        cmap = plt.get_cmap('tab10') 
        n_patterns = len(data)
        
        # Impostiamo i limiti verticali in base al numero di pattern estratti
        ax.set_ylim(0.5, n_patterns + 0.5)
        
        for idx, p in enumerate(data):
            # Posizionamento verticale (Pattern 1 in alto)
            y_base = n_patterns - idx 
            indices = sorted(p['indices'])
            color = cmap(idx % 10) # Seleziona il colore in base all'indice
            
            # 1. Linea guida orizzontale per la lettura
            ax.axhline(y_base, color='gray', alpha=0.1, zorder=1)
            
            # 2. Scatter dei punti: ogni occorrenza del pattern sulla timeline
            ax.scatter(indices, [y_base]*len(indices), color=color, 
                       s=150, zorder=4, edgecolors='white', linewidth=1.5, 
                       label=f"P{idx+1}: {list(p['pattern'])}")
            
            # 3. Calcolo e visualizzazione della Tensione e Sorpresa Temporale
            if len(indices) > 1:
                deltas = np.diff(indices)
                mean_t = np.mean(deltas)
                
                # Definiamo l'altezza della fascia colorata attorno alla riga del pattern
                y_min_span = y_base - 0.4
                y_max_span = y_base + 0.4

                for i in range(len(deltas)):
                    start_x = indices[i]
                    end_x = indices[i+1]
                    
                    # Sorpresa: quanto la distanza attuale devia dalla media del pattern
                    diff = abs(deltas[i] - mean_t)
                    
                    # L'opacità (alpha) aumenta quanto più l'evento è "sorprendente" (fuori tempo medio)
                    alpha_val = min(0.4, diff / (mean_t * 1.5 + 1e-5))
                    
                    # Colorazione dell'intervallo tra due occorrenze
                    ax.fill_between([start_x, end_x], y_min_span, y_max_span,
                                   color=color, alpha=alpha_val, zorder=2)
                    
                    # Linea di "tensione" dinamica: pendenza basata sul rapporto col tempo medio
                    tension = (deltas[i] / (mean_t + 1e-5)) * 0.15
                    ax.plot([start_x, end_x], [y_base-0.25, y_base-0.25 + tension], 
                            color=color, alpha=0.6, linewidth=1.2, zorder=3)

        # Formattazione assi e etichette
        ax.set_yticks(range(1, n_patterns + 1))
        # Invertiamo le etichette per far corrispondere correttamente P1, P2...
        pattern_labels = [f"Pattern {n_patterns-i+1}" for i in range(1, n_patterns+1)][::-1]
        ax.set_yticklabels(pattern_labels, fontweight='bold')
        
        ax.set_xlabel("Progressione Temporale (Indice Nota nel Flusso Unificato)", fontsize=12)
        ax.set_title(f"ARCHITETTURA COGNITIVA: {os.path.basename(self.path).upper()}", 
                     fontsize=15, fontweight='black', pad=25)
        
        # Legenda esterna con gli intervalli dei pattern
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, 
                  title=fr"Pattern Sopra Soglia ($\Phi > {self.phi_threshold:.2f}$)", title_fontproperties={'weight':'bold'})
        
        plt.tight_layout()
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Report grafico generato: {output_name}")
        plt.show()
    

    def export_patterns_to_midi(self, output_name_base):
        """
        Esporta TUTTI i top pattern usando la mappatura note_to_csv_map per recuperare
        le durate, velocity e i distacchi (transition) ORIGINALI.
        Gestisce le sovrapposizioni (polifonia) e nomina il file con il numero primo successivo.
        """
        try:
            from midiutil import MIDIFile
        except ImportError:
            print("[ERROR] Installa midiutil: pip install midiutil")
            return

        # --- LOGICA NUMERI PRIMI PER IL NOME FILE ---
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

        # Pulizia del nome base per la scansione
        clean_base = output_name_base.replace(".mid", "")
        
        # Scansione per trovare l'ultimo numero primo usato nel nome file
        existing_files = [f for f in os.listdir(downloads_path) if f.startswith(clean_base) and f.endswith('.mid')]
        last_prime = 1
        for f in existing_files:
            try:
                # Estraiamo la parte numerica finale (es. Nome_5.mid -> 5)
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
        # --------------------------------------------

        # Creazione MIDI (1 traccia)
        midi = MIDIFile(1)
        midi.addTempo(0, 0, 120)
        
        # pattern_start_cursor tiene traccia di dove inizia ogni blocco di pattern
        pattern_start_cursor = 0 

        # Ciclo su tutti i pattern sopra soglia
        for idx, p in enumerate(self.top_patterns):
            # Recuperiamo la prima occorrenza del pattern nel flusso sonoro
            start_note_idx = p['indices'][0]
            pattern_len = len(p['pattern']) + 1
            
            # Segmento di dati reali dal CSV
            segment_notes = self.raw_notes_data[start_note_idx : start_note_idx + pattern_len]
            
            # Offset del pitch per distinguere i pattern uditivamente (nel range 0-127)
            base_listening_pitch = 60 + ((idx % 12) * 4) 
            pitch_offset = base_listening_pitch - segment_notes[0]['pitch']
            
            # Cursore temporale interno al pattern
            note_time_cursor = pattern_start_cursor
            
            for note_data in segment_notes:
                # 1. Calcolo Pitch con offset
                pitch = int(max(0, min(127, note_data['pitch'] + pitch_offset)))
                
                # 2. Velocity e Durata originali
                vel = int(note_data['velocity'])
                dur = float(note_data['duration'])
                
                # 3. Transizione originale (distanza dall'inizio della nota attuale alla prossima)
                # Recuperata grazie alla modifica fatta in load_and_clean
                trans = float(note_data['transition'])
                
                # Aggiunta della nota al MIDI
                # Se dur > trans, la nota continuerà a suonare mentre inizia la successiva (sovrapposizione)
                midi.addNote(0, 0, pitch, note_time_cursor, dur, vel)
                
                # IL SEGRETO: Il cursore si muove in base alla transizione, non alla durata
                note_time_cursor += trans
            
            # Spostiamo l'inizio del prossimo pattern 4 battiti dopo la fine dell'ultimo cursore locale
            pattern_start_cursor = note_time_cursor + 4

        try:
            with open(full_path, "wb") as output_file:
                midi.writeFile(output_file)
            print(f"[SUCCESS] MIDI ({len(self.top_patterns)} pattern) salvato con polifonia reale: {full_path}")
        except Exception as e:
            print(f"[ERROR] Scrittura MIDI fallita: {e}")

    def analyze_temporal_surprise(self):
        """
        Calcola la sorpresa temporale per i top pattern identificati.
        S = |t_i - t_media| dove t è la distanza tra occorrenze successive.
        """
        if not hasattr(self, 'top_patterns') or not self.top_patterns:
            print("[WARNING] Nessun pattern trovato per l'analisi della sorpresa.")
            return

        # Analizziamo solo i pattern che hanno almeno 2 occorrenze
        print(f"[INFO] Analisi temporale della sorpresa su pattern ricorrenti...")

        for p in self.top_patterns:
            indices = sorted(p['indices'])
            if len(indices) < 2:
                p['surprises'] = []
                p['mean_t'] = 0
                p['avg_surprise'] = 0
                continue
            
            deltas = np.diff(indices) 
            mean_t = np.mean(deltas)
            surprises = [abs(d - mean_t) for d in deltas]
            
            p['surprises'] = surprises
            p['avg_surprise'] = np.mean(surprises)
            p['mean_t'] = mean_t

    

# --- INTEGRAZIONE NEL MAIN ---
if __name__ == "__main__":
    print(">>> [SCIENTIFIC ANALYSIS] MODULO 1, 2 & 3")
    path_file = input("PATH CSV > ").strip().replace("'", "").replace('"', "")
    
    # Definiamo la cartella di output fissa su Downloads
    downloads_path = "/Users/raph/Downloads"
    nome_base = os.path.basename(path_file).replace(".csv", "")

    try:
        engine = MusicalMarkovEngine(path_file)
        engine.load_and_clean()
        
        # 1. Analisi Markoviana
        # Trova l'ordine ottimale di memoria della catena
        engine.find_optimal_k()
        engine.compute_stationary_entropy()
        
        # 2. Pattern Discovery
        # Estrazione basata sulla soglia dinamica (3-10) calcolata su Phi
        print("\n3. Estrazione e pesatura dei pattern...")
        engine.extract_patterns() 
        
        # 3. Analisi Temporale e Grafica
        # Calcolo delle deviazioni temporali (sorpresa t) per i report
        print("\n4. Analisi della sorpresa e generazione report...")
        engine.analyze_temporal_surprise()
        
        # Percorso per il grafico scientifico standard (Report Architettura)
        report_path = os.path.join(downloads_path, f"Analisi_{nome_base}.png")
        engine.plot_scientific_report(output_name=report_path)

        # 4. Generazione grafico Sorpresa Markoviana S(t) e calcolo valore medio
        print("\n5. Generazione grafico Sorpresa Markoviana e calcolo media...")
        surprise_path = os.path.join(downloads_path, f"S_t_{nome_base}.png")
        # Catturiamo il valore medio (Energia Cognitiva Media)
        s_media = engine.plot_markovian_surprise(output_name=surprise_path)

        # 5. Analisi Termodinamica Dinamica (dH/dt, dR/dt)
        # Calcolo delle derivate temporali basate sulla costante di Boltzmann
        print("\n6. Analisi termodinamica delle derivate H ed R...")
        #PER ORA NON LI BOGLI SALVARE
        #thermo_path = os.path.join(downloads_path, f"Dinamica_Informativa_{nome_base}.png")
        #mean_dh, mean_dr = engine.plot_dynamic_thermodynamics(output_name=thermo_path)


        # 6. Esportazione MIDI in Downloads
        # Genera il file MIDI dei pattern con la polifonia reale e nomi basati su numeri primi
        midi_name = f"Pattern_{nome_base}.mid"
        engine.export_patterns_to_midi(midi_name)


        # --- OUTPUT RISULTATI FINALI NEL TERMINALE ---
        
        # Sezione 1: Energia Cognitiva Media (S Media)
        print(f"\n{'='*75}")
        print(f"{'ANALISI ENERGETICA COGNITIVA MEDIA (S)':^75}")
        print(f"{'-'*75}")
        print(f"👉 Valore Medio di S(t) (Energia per Nota): {s_media:.4f}")
        print(f"{'='*75}")
        
        # Entropia stazionaria
        print(f"👉 Entropia Stazionaria della Catena (H): {engine.h_chain:.4f} bits/nota")

        # Sezione 2: Termodinamica delle Derivate (H ed R)
        print(f"\n{'='*75}")
        print(f"{'DINAMICA TERMODINAMICA (dH/dt, dR/dt)':^75}")
        print(f"{'-'*75}")
        print(f"👉 Velocità media variazione Entropia (|dH/dt|): {mean_dh:.2e} J/K")
        print(f"👉 Velocità media variazione Ridondanza (|dR/dt|): {mean_dr:.4f}")
        print(f"{'='*75}")

        # Sezione 3: Dettaglio Top Pattern (Ranking Phi e Sorpresa t)
        print(f"\n{'='*75}")
        print(f"{'TOP PATTERN RILEVATI (Soglia Phi: ' + f'{engine.phi_threshold:.2f}' + ')':^75}")
        print(f"{'-'*75}")
        print(f"{'PATTERN (INTERVALLI)':<35} | {'FREQ':<5} | {'SORPRESA MEDIA (t)':<10}")
        print(f"{'-'*75}")
        
        # Stampiamo tutti i pattern estratti che hanno superato la soglia dinamica
        for p in engine.top_patterns:
            surprise_val = p.get('avg_surprise', 0)
            print(f"{str(list(p['pattern'])):<35} | {p['count']:<5} | {surprise_val:.2f}")
        print(f"{'='*75}")
        
        print(f"\n✔ Analisi completata con successo.")
        print(f"✔ Grafici e MIDI salvati in: {downloads_path}")

    except Exception as e:
        import traceback
        print(f"\n❌ Errore durante l'esecuzione dell'analisi:")
        traceback.print_exc()