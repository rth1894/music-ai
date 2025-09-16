import json
import numpy as np
import pretty_midi
import random
from keras.models import load_model


try:
    model = load_model("music_all_tokens_model.keras")
except FileNotFoundError:
    print("\n\n-----Model not found, check path again!-------\n\n")

with open("note_to_int.json") as f:
    note_to_int = json.load(f)
with open("int_to_note.json") as f:
    int_to_note = json.load(f)

vocab_size = len(note_to_int)
sequence_length = 50

start = random.randint(0, len(list(note_to_int)) - sequence_length)
seed = list(note_to_int.values())[start:start + sequence_length]
generated = seed.copy()

n = 500

def sample_with_temperature(probabilities, temperature=1.5):
    if temperature <= 0:
        return np.argmax(probabilities)
    predictions = np.log(probabilities + 1e-9) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(predictions), p=predictions)

for _ in range(n):
    input_sequence = np.array(generated[-sequence_length:]).reshape(1, sequence_length)
    predicted_probs = model.predict(input_sequence, verbose=0)[0]
    predicted_index = sample_with_temperature(predicted_probs, temperature=1.2)
    generated.append(predicted_index)

generated_notes = [int_to_note[str(i)] for i in generated]


def tokens_to_midi(tokens, output_path="generated.mid", step=0.125):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    time = 0
    for token in tokens:
        """
        pitch, duration, velocity = float(map(float, token.split('_')))
        note = pretty_midi.Note(
            velocity=int(velocity * 127),
            pitch=int(pitch * 127),
            start=time,
            end=time + duration * step
        )
        """

        pitch = int(token.split('_')[1])
        duration = float(token.split('_')[3])
        velocity = int(token.split('_')[5])

        note = pretty_midi.Note(
            velocity = velocity,
            pitch = pitch,
            start = time,
            end = int(time + duration * step)
        )

        instrument.notes.append(note)
        time += int(duration * step)

    midi.instruments.append(instrument)
    midi.write(output_path)
    print(f"saved to: {output_path}")
print(generated_notes[:10])

tokens_to_midi(generated_notes, output_path="generated_all.mid")
