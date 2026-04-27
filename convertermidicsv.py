# questo codice rispetto al precedente midi2csv considera gli overlapping, distinguendo tra durata della nota e durata della transizione. 
# in questo modo una nota può cintinuare a suonare (sostenere) mentre altre si sovrappongono. prox step accordi... 

import mido
import pandas as pd
import os


# ============================================================
# MAPPE SERIALI
# ============================================================

DURATION_MAP = {
    1.125: "9/8",
    1.0: "1",
    0.875: "7/8",
    0.75: "3/4",
    0.5: "1/2",
    0.25: "1/4",
    0.1875: "3/16",
    0.125: "1/8",
    0.09375: "3/32",
    0.0625: "1/16",
    0.03125: "1/32",
    0.015625: "1/64"
}

DURATION_VALUES = list(DURATION_MAP.keys())
MIN_DUR = 0.0625   # 1/16


VELOCITY_MAP = {
    (0, 30): "ppppp",
    (31, 40): "pppp",
    (41, 50): "ppp",
    (51, 60): "pp",
    (61, 70): "p",
    (71, 80): "mp",
    (81, 90): "mf",
    (91, 100): "f",
    (101, 110): "ff",
    (111, 120): "fff",
    (121, 127): "ffff"
}


def quantize_duration(d):
    closest = min(DURATION_VALUES, key=lambda x: abs(x - d))
    sym = DURATION_MAP[closest]
    if abs(d - closest) > 1e-6:
        return f"{d} (~{sym})"
    return f"{d} ({sym})"


def quantize_velocity(v):
    for (lo, hi), sym in VELOCITY_MAP.items():
        if lo <= v <= hi:
            return f"{v} ({sym})"
    return f"{v} (ppppp)"


# ============================================================
# STEP 1 — PRE_PROCESSED (note_on-centric)
# ============================================================
# ============================================================
# STEP 1 — PRE_PROCESSED (Merge + Overlap + Robustezza)
# ============================================================

def midi_to_preprocessed(midi_path, out_csv):
    """
    Step 1: Estrazione note con durata REALE. 
    Usa Note-Off e Velocity=0 per definire la fine della nota.
    """
    try:
        mid = mido.MidiFile(midi_path, clip=True)
    except Exception as e:
        print(f"  ❌ Errore critico nel caricamento di {os.path.basename(midi_path)}: {e}")
        return 120

    ppqn = mid.ticks_per_beat
    all_notes = []

    # Analizziamo ogni traccia per accoppiare Note-On e Note-Off
    for track in mid.tracks:
        abs_tick = 0
        active_notes = {} # Dizionario temporaneo: nota -> (start_tick, velocity)
        
        for msg in track:
            abs_tick += msg.time
            
            # CASO A: Nota Premuta (Note-On con Velocity > 0)
            if msg.type == "note_on" and msg.velocity > 0:
                # Se la nota era già attiva, la chiudiamo per sicurezza prima di riaprirla
                if msg.note in active_notes:
                    s_tick, s_vel = active_notes.pop(msg.note)
                    all_notes.append((s_tick, abs_tick, msg.note, s_vel))
                
                # Registriamo l'inizio
                active_notes[msg.note] = (abs_tick, msg.velocity)
            
            # CASO B: Nota Rilasciata (Note-Off OPPURE Note-On con Velocity = 0)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active_notes:
                    s_tick, s_vel = active_notes.pop(msg.note)
                    # Salviamo la nota con il suo tick di fine REALE
                    all_notes.append((s_tick, abs_tick, msg.note, s_vel))
    
    if not all_notes:
        print(f"  ⚠️ Nessun evento nota trovato in {os.path.basename(midi_path)}")
        return ppqn

    # Creiamo un DataFrame con tutte le note trovate
    df_raw = pd.DataFrame(all_notes, columns=["start_tick", "end_tick", "note", "velocity"])
    
    # Calcoliamo la durata REALE (informazione intrinseca del MIDI)
    df_raw["duration"] = (df_raw["end_tick"] - df_raw["start_tick"]) / ppqn

    # --- LOGICA MONOFONICA (Manteniamo solo la nota più alta per ogni istante) ---
    # Ordiniamo e raggruppiamo per start_tick
    rows = []
    grouped = df_raw.groupby("start_tick")
    for t, group in grouped:
        # Tra le note che iniziano nello stesso momento, scegliamo quella con pitch (note) maggiore
        best_note = group.loc[group["note"].idxmax()]
        rows.append({
            "start_tick": t,
            "note": int(best_note["note"]),
            "velocity": int(best_note["velocity"]),
            "duration": best_note["duration"]
        })

    # Salvataggio CSV pre-processed con durata reale
    df_final = pd.DataFrame(rows).sort_values("start_tick")
    #df_final.to_csv(out_csv, index=False)
    return df_final, ppqn # Restituisce il dataframe invece di salvarlo


# ============================================================
# STEP 2 — PROCESSED (overlap + transizioni)
# ============================================================

# ============================================================
# STEP 2 — PROCESSED (Robustezza + Pausa Iniziale + Overlap)
# ============================================================

def pre_to_processed(df, out_csv, ppqn):
    """
    Step 2: Generazione del file finale.
    Mantiene la durata reale e calcola la transizione come IOI (Inter-Onset Interval).
    """
    # Se il DataFrame è nullo o vuoto, esce senza fare nulla
    if df is None or df.empty:
        return

    # Assicuriamoci che l'ordine temporale sia perfetto
    df = df.sort_values("start_tick").reset_index(drop=True)
    rows = []

    # Gestione della pausa iniziale se presente
    first_tick = df.iloc[0]["start_tick"]
    if first_tick >= (MIN_DUR * ppqn):
        initial_pause = first_tick / ppqn
        rows.append({
            "event": "pause",
            "duration": quantize_duration(initial_pause),
            "transition": quantize_duration(initial_pause),
            "velocity": "0"
        })

    prev = None

    for i, r in df.iterrows():
        # 'dur' qui è la durata reale estratta dal MIDI nello Step 1
        dur = r.duration

        if dur < MIN_DUR:
            continue

        if prev is None:
            prev = {
                "note": r.note,
                "start": r.start_tick,
                "dur": dur,
                "vel": r.velocity
            }
            continue

        # CALCOLO TRANSIZIONE: Distanza tra inizio nota precedente e inizio attuale
        # Questo valore è indipendente dalla durata della nota stessa
        trans_dur = (r.start_tick - prev["start"]) / ppqn

        # Se la nota è la stessa della precedente, le fondiamo (Legato)
        if int(r.note) == int(prev["note"]):
            # Aumentiamo la durata della nota "sostenuta"
            prev["dur"] += dur
            # Saltiamo la scrittura della riga: la transizione verrà calcolata 
            # rispetto alla prossima nota diversa
            continue

        # Scrittura della nota precedente con i suoi valori distinti
        rows.append({
            "event": int(prev["note"]),
            "duration": quantize_duration(prev["dur"]),    # Quanto dura il suono
            "transition": quantize_duration(trans_dur),    # Dopo quanto tempo arriva la prossima nota
            "velocity": quantize_velocity(int(prev["vel"]))
        })

        # Aggiorniamo 'prev' con la nota attuale
        prev = {
            "note": r.note,
            "start": r.start_tick,
            "dur": dur,
            "vel": r.velocity
        }

    # Chiusura dell'ultima nota del brano
    if prev is not None:
        rows.append({
            "event": int(prev["note"]),
            "duration": quantize_duration(prev["dur"]),
            "transition": quantize_duration(prev["dur"]), # Fallback per l'ultima nota
            "velocity": quantize_velocity(int(prev["vel"]))
        })

    # Creazione del CSV finale "processed"
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# ============================================================
# MAIN — CARTELLA → CARTELLA STRUTTURATA (VERSIONE ROBUSTA)
# ============================================================

if __name__ == "__main__":
    print("\n=== MIDI → OVERLAP MONODY PIPELINE (ROBUST MERGE) ===\n")

    input_dir = input("Cartella INPUT (MIDI): ").strip().replace("'", "").replace('"', "")
    output_root = input("Cartella OUTPUT: ").strip().replace("'", "").replace('"', "")

    # Controllo esistenza cartella input
    if not os.path.isdir(input_dir):
        print(f"❌ Errore: La cartella di input non esiste: {input_dir}")
        exit()

    name = os.path.basename(os.path.normpath(input_dir))
    base_out = os.path.join(output_root, name)

    # Creazione sottocartelle
    # pre_dir = os.path.join(base_out, f"{name}_pre_processed")  # COMMENTATO
    proc_dir = os.path.join(base_out, f"{name}_processed")
    
    # os.makedirs(pre_dir, exist_ok=True)  # COMMENTATO
    os.makedirs(proc_dir, exist_ok=True)

    # Lista dei file MIDI
    midi_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".mid", ".midi"))]
    print(f"Total files found: {len(midi_files)}")

    for midi in midi_files:
        base = os.path.splitext(midi)[0]
        # pre_csv = os.path.join(pre_dir, f"{base}_pre.csv")  # COMMENTATO
        proc_csv = os.path.join(proc_dir, f"{base}_processed.csv")
        path_completo = os.path.join(input_dir, midi)

        # STEP 1: Ora restituisce il DataFrame direttamente in memoria
        # Passiamo None come secondo argomento perché la funzione non deve più scrivere il CSV
        df_temp, ppqn = midi_to_preprocessed(path_completo, None)

        # STEP 2: Passiamo direttamente il dataframe 'df_temp' alla funzione
        if df_temp is not None and not df_temp.empty:
            pre_to_processed(df_temp, proc_csv, ppqn)
            print(f"✔ Elaborato con successo: {midi}")
        else:
            print(f"  ⚠️ Saltato {midi}: Nessun dato valido estratto.")

    print("\n✔ Pipeline completata. Generati solo i file 'processed'.")