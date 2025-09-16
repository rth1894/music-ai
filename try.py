import os
import json
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

source_dir = "./preprocessed_notes"
target_dir = "./top_preprocessed_notes"
num_files_to_copy = 5000

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

json_files = list(Path(source_dir).rglob("*.json"))
selected_files = random.sample(json_files, num_files_to_copy)

for file_path in selected_files:
    shutil.copy(file_path, Path(target_dir) / file_path.name)

def quantize(value, step, min_value=0, max_value=127):
    return min(max_value, max(min_value, int(round(value / step) * step)))

def load_notes_pandas(folder):
    dfs = []
    for file in Path(folder).rglob("*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                if data:
                    df = pd.DataFrame(data, columns=["pitch", "duration", "velocity"])
                    dfs.append(df)
        except:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

Ft_path = "./top_preprocessed_notes"
notes_df = load_notes_pandas(Ft_path)

if notes_df.empty:
    raise RuntimeError("No valid notes found in directory!")

notes_df["duration"] = notes_df["duration"].apply(lambda x: quantize(x, step=0.25))
notes_df["velocity"] = notes_df["velocity"].apply(lambda x: quantize(x, step=32))
notes_df["note_token"] = notes_df.apply(lambda row: f"{row['pitch']}_{row['duration']}_{row['velocity']}", axis=1)

tokens = list(notes_df["note_token"])
unique_tokens = sorted(set(tokens))
note_to_int = {note: i for i, note in enumerate(unique_tokens)}
int_to_note = {i: note for i, note in enumerate(unique_tokens)}
vocab_size = len(unique_tokens)
print(f"🎼 Vocabulary size: {vocab_size}")

encoded = [note_to_int[token] for token in tokens]

sequence_length = 100
max_samples = 100000

X, y = [], []
for i in tqdm(range(min(len(encoded) - sequence_length, max_samples)), desc="Building sequences"):
    X.append(encoded[i:i+sequence_length])
    y.append(encoded[i+sequence_length])

X = np.array(X, dtype=np.int32)
y = np.array(y, dtype=np.int32)

shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

model = Sequential([
    Input(shape=(sequence_length,)),
    Embedding(input_dim=vocab_size, output_dim=256),
    LSTM(512, return_sequences=True),
    Dropout(0.4),
    LSTM(512),
    Dropout(0.4),
    Dense(vocab_size, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_please.keras", monitor='val_loss', save_best_only=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=30,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

model.save("please.keras")

with open("note_to_int.json", "w") as f:
    json.dump(note_to_int, f)
with open("int_to_note.json", "w") as f:
    json.dump(int_to_note, f)

print("✅ Model and vocabulary saved.")

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.legend()
plt.show()
