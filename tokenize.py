import os
import json
from pathlib import Path
from tqdm import tqdm

input_dir = './preprocessed_notes/'
output_dir = './tokenized_sequences/'
os.makedirs(output_dir, exist_ok=True)

def tokenize_note(pitch, duration, velocity):
    return f"note_{pitch}_dur_{duration}_vel_{velocity}"

def main():
    files = list(Path(input_dir).rglob('*.json'))
    total = 0

    for file in tqdm(files, desc="Tokenizing"):
        with open(file, 'r') as f:
            notes = json.load(f)

        tokens = []
        for note in notes:
            pitch, duration, velocity = note
            token = tokenize_note(pitch, duration, velocity)
            tokens.append(token)

        out_path = Path(output_dir) / (file.stem + '.txt')
        with open(out_path, 'w') as f:
            f.write(' '.join(tokens))

        total += 1

    print(f"Tokenization complete — Processed: {total} files")

if __name__ == "__main__":
    main()
