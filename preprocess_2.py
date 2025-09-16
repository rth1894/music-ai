import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
from collections import Counter
import math

input_dir = './preprocessed_notes/'
output_dir = './top_preprocessed_notes/'
os.makedirs(output_dir, exist_ok=True)

TOP_N = 5000

def pitch_entropy(pitches):
    if len(pitches) == 0:
        return 0
    counter = Counter(pitches)
    total = len(pitches)
    entropy = -sum((count/total) * math.log2(count/total) for count in counter.values())
    return entropy

def compute_score(file_path):
    with open(file_path) as f:
        notes = json.load(f)

    if not notes or len(notes) < 50:
        return -1

    pitches = [n[0] for n in notes]
    durations = [n[1] for n in notes]
    velocities = [n[2] for n in notes]

    num_notes = len(notes)
    unique_pitches = len(set(pitches))
    avg_duration = np.mean(durations)
    avg_velocity = np.mean(velocities)
    entropy = pitch_entropy(pitches)

    score = (
        num_notes / 1000.0 +
        unique_pitches / 10.0 +
        entropy / 2.0 +
        (1.0 if 3 <= avg_duration <= 12 else 0) +
        (1.0 if 40 <= avg_velocity <= 100 else 0)
    )
    return score

def select_best_files():
    all_files = list(Path(input_dir).rglob('*.json'))
    file_scores = []
    for path in tqdm(all_files, desc="Scoring files"):
        score = compute_score(path)
        if score > 0:
            file_scores.append((path, score))

    file_scores.sort(key=lambda x: x[1], reverse=True)
    selected = file_scores[:TOP_N]
    for i, (src_path, _) in enumerate(selected):
        dst_path = Path(output_dir) / src_path.name
        shutil.copy(src_path, dst_path)

    print(f"Selected top {len(selected)} files to: {output_dir}")

if __name__ == "__main__":
    select_best_files()
