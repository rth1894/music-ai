import mido
import os

midi_root = './lmd_full'

for root, dirs, files in os.walk(midi_root):
    for file in files:
        if not file.endswith('.mid'):
            continue

        midi_path = os.path.join(root, file)

        try:
            mid = mido.MidiFile(midi_path)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

        print(f"Processing: {file}")

        for i, track in enumerate(mid.tracks):
            for msg in track:
                if msg.type == 'note_on':
                    print(f"\tnote on: pitch={msg.note}, velocity={msg.velocity}, time={msg.time}")
                elif msg.type == 'note_off':
                    print(f"\tnote off: pitch={msg.note}, velocity={msg.velocity}, time={msg.time}")
                elif msg.type == 'pitchwheel':
                    print(f"\tpitch bend: value={msg.pitch}")
