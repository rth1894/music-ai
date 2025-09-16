import os
import json
import pretty_midi
from pathlib import Path
from tqdm import tqdm

midi_full = './dataset_final/'
output = './preprocessed_notes/'

piano = [0, 1, 2, 3, 4, 5, 6, 7]

def is_piano(instrument):
    return instrument.program in piano and not instrument.is_drum

def get_notes(path, min_notes, log=None):
    step = 0.125
    try:
        midi = pretty_midi.PrettyMIDI(str(path))
    except Exception as e:
        if log is not None:
            log.write(f"{path}\n")
        return None

    notes = []
    for instr in midi.instruments:
        if not is_piano(instr):
            continue
        for note in instr.notes:
            notes.append({
                'pitch': note.pitch,
                'start': note.start,
                'end': note.end,
                'duration': note.end - note.start,
                'velocity': note.velocity
            })

    if len(notes) < min_notes:
        return None

    quantized = []
    for note in notes:
        start = round(note['start'] / step) * step
        end = round(note['end'] / step) * step
        duration = max((end - start), step)
        note['start'] = start
        note['end'] = end
        note['duration'] = duration
        quantized.append(note)

    encoded = []
    for note in quantized:
        pitch = note['pitch']
        duration = int(round(note['duration'] / step))
        velocity = note['velocity']
        encoded.append((pitch, duration, velocity))

    return encoded

def main():
    min_notes = 50
    target_count = 10000

    files = list(Path(midi_full).rglob('*.mid')) + list(Path(midi_full).rglob('*.midi'))
    os.makedirs(output, exist_ok=True)

    kept, skip, index = 0, 0, 0

    with open('fails.txt', 'a') as log, tqdm(total=target_count, desc="Preprocessing") as pbar:
        while kept < target_count and index < len(files):
            path = files[index]
            index += 1

            notes = get_notes(path, min_notes, log=log)
            if notes is None:
                skip += 1
                continue

            out_path = Path(output) / (path.stem + '.json')
            with open(out_path, 'w') as f:
                json.dump(notes, f)

            kept += 1
            pbar.update(1)

    print(f"Preprocessing complete\nKept: {kept}, Skipped: {skip}")

if __name__ == "__main__":
    main()
