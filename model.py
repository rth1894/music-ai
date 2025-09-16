import os
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Embedding, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm


Ft_path = "./preprocessed_notes"


def load_notes_pandas(folder):
    dfs = []
    for file in Path(folder).rglob("*.json"):
        with open(file, "r") as f:
            try:
                data = json.load(f)
                if data:
                    df = pd.DataFrame(data, columns=["pitch", "duration", "velocity"])
                    dfs.append(df)
            except json.JSONDecodeError:
                print(f"Could not decode JSON file: {file}")

    if not dfs:
        return pd.DataFrame(columns=["pitch", "duration", "velocity"])

    all_notes = pd.concat(dfs, ignore_index=True)
    return all_notes


notes_df = load_notes_pandas(Ft_path)

if notes_df.empty:
    print("No valid notes found. Check your JSON files in:", Ft_path)
    exit()

notes_df["note_token"] = notes_df.apply(
    lambda row: f"{row['pitch']}_{row['duration']}_{row['velocity']}", axis=1)
tokens = list(notes_df["note_token"])

unique_tokens = sorted(set(tokens))
note_to_int = {note: i for i, note in enumerate(unique_tokens)}
int_to_note = {i: note for i, note in enumerate(unique_tokens)}
vocab_size = len(unique_tokens)
print(f"Vocabulary size: {vocab_size}")

encoded = [note_to_int[n] for n in tokens]

sequence_length = 50
X, y = [], []


for i in tqdm(range(len(encoded) - sequence_length), desc="Building sequences"):
    X.append(encoded[i:i + sequence_length])
    y.append(encoded[i + sequence_length])
X = np.array(X, dtype=np.int32)
y = np.array(y, dtype=np.int32)

limit = 50000
X = X[:limit]
y = y[:limit]



model = Sequential([
    Input(shape=(sequence_length,)),
    Embedding(input_dim=vocab_size, output_dim=256),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.3),
    LSTM(256),
    Dropout(0.3),
    Dense(vocab_size, activation='softmax')
    ])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

early_stop = EarlyStopping(
    monitor='loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X, y,
    batch_size=64,
    epochs=35,
    callbacks=[early_stop],
    verbose=1
)

model.save("music_token_model.h5")
print("Model saved as 'music_token_model.h5'")

with open("note_to_int.json", "w") as f:
    json.dump(note_to_int, f)
with open("int_to_note.json", "w") as f:
    json.dump(int_to_note, f)
print(" Token mappings saved.")
