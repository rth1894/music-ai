from pathlib import Path
import time
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Embedding, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import math

token_path = "./tokenized_sequences_all"

def load_tokenized_notes_per_file(folder):
    all_songs = []
    for file in Path(folder).rglob("*.txt"):
        with open(file, "r") as f:
            content = f.read().strip()
            if content:
                tokens = content.split()
                all_songs.append(tokens)
    return all_songs

songs = load_tokenized_notes_per_file(token_path)

if not songs:
    print("No tokenized songs found in ", token_path)
    exit()

all_tokens_flat = [token for song in songs for token in song]
unique_tokens = sorted(set(all_tokens_flat))
note_to_int = {note: i for i, note in enumerate(unique_tokens)}
int_to_note = {i: note for i, note in enumerate(unique_tokens)}
vocab_size = len(unique_tokens)
print(f"Vocabulary size: {vocab_size}")

encoded_songs = [[note_to_int[n] for n in song] for song in songs]

sequence_length = 130
X, y = [], []

for encoded in tqdm(encoded_songs, desc="Building sequences"):
    if len(encoded) <= sequence_length:
        continue
    for i in range(len(encoded) - sequence_length):
        X.append(encoded[i:i + sequence_length])
        y.append(encoded[i + sequence_length])

X = np.array(X, dtype=np.int32)
y = np.array(y, dtype=np.int32)

print(f"Total sequences: {len(X)}")

#"""
limit = 90000
X, y = X[:limit], y[:limit]
#"""

sequence_counts = []
for song in songs:
    length = len(song)
    count = max(0, length - sequence_length)
    sequence_counts.append(count)

total_sequences = sum(sequence_counts)
print(f"Estimated total sequences: {total_sequences}")
print(f"Average sequences per file: {total_sequences / len(songs):.2f}")
time.sleep(10)

model = Sequential([
    Input(shape=(sequence_length,)),
    Embedding(input_dim=vocab_size, output_dim=256),
    Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2,
        kernel_regularizer=regularizers.l2(1e-4), recurrent_regularizer=regularizers.l2(1e-4))),
    Dropout(0.4),
    LSTM(128, recurrent_dropout=0.2, kernel_regularizer=regularizers.l2(1e-4), recurrent_regularizer=regularizers.l2(1e-4)),
    Dropout(0.4),
    Dense(vocab_size, activation='softmax', kernel_regularizer=regularizers.l2(1e-4))
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1),
    metrics=['accuracy']
)
model.summary()

class PerplexityCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key in ['loss', 'val_loss']:
            if key in logs and logs[key] is not None:
                ppl = math.exp(logs[key])
                print(f"{key} perplexity: {ppl:.2f}")

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

history = model.fit(
    X, y,
    batch_size=128,
    epochs=35,
    validation_split=0.15,
    callbacks=[early_stop, reduce_lr, PerplexityCallback()],
    verbose=1,
    shuffle=True
)

model.save("music_all_tokens_high_sequence_model.keras")
with open("note_to_int.json", "w") as f:
    json.dump(note_to_int, f)
with open("int_to_note.json", "w") as f:
    json.dump(int_to_note, f)
print("Model and token mappings saved.")

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training & Validation Loss / Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("training_history.png", dpi=300, bbox_inches='tight')
plt.show()
