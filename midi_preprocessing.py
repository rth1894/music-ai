
import os
import glob
import pickle
import numpy as np
import mido
import random
import json
from collections import Counter

midi_full = "D:/music-evolution/Data/archive"
output = "D:/music-evolution/Data/process1_notes"
piano_programs = [0, 1, 2, 3, 4, 5, 6, 7]  
max_files = 15000
min_sequence_length = 50  
max_sequence_length = 500  
repetition_threshold = 0.7  

def calculate_repetition_ratio(sequence):
    """Calculate how repetitive a sequence is"""
    if len(sequence) < 10:
        return 1.0

    counter = Counter(sequence)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(sequence)

def extract_notes_from_midi(file_path):
    """Extract piano notes using mido library"""
    try:
        mid = mido.MidiFile(file_path)
        notes = []
        current_time = 0

        active_notes = {}

        for track in mid.tracks:
            current_time = 0

            for msg in track:
                current_time += msg.time

                if msg.type == 'program_change':
                    if msg.program not in piano_programs:
                        break  

                elif msg.type == 'note_on' and msg.velocity > 0:
 
                    active_notes[msg.note] = {
                        'start_time': current_time,
                        'velocity': msg.velocity
                    }

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):

                    if msg.note in active_notes:
                        start_info = active_notes[msg.note]
                        duration = current_time - start_info['start_time']

                        if duration > 0:  
                            notes.append({
                                'pitch': msg.note,
                                'duration': duration,
                                'velocity': start_info['velocity'],
                                'start_time': start_info['start_time']
                            })

                        del active_notes[msg.note]

        notes.sort(key=lambda x: x['start_time'])

        return notes if len(notes) >= min_sequence_length else None

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def normalize_sequence(notes):
    """Normalize notes while preserving musical structure and avoiding repetition"""
    if not notes:
        return None

    intervals = []
    durations = []
    dynamics = []

    for i, note in enumerate(notes):

        if i == 0:
            intervals.append(0) 
        else:
            interval = note['pitch'] - notes[i-1]['pitch']
 
            intervals.append(max(-24, min(24, interval)))

        duration_ms = note['duration']
        if duration_ms < 100:
            durations.append('short')    
        elif duration_ms < 300:
            durations.append('medium')   
        elif duration_ms < 600:
            durations.append('long')     
        else:
            durations.append('very_long') 
       
        vel = note['velocity']
        if vel < 40:
            dynamics.append('pp')
        elif vel < 60:
            dynamics.append('p')
        elif vel < 80:
            dynamics.append('mp')
        elif vel < 100:
            dynamics.append('mf')
        elif vel < 120:
            dynamics.append('f')
        else:
            dynamics.append('ff')

    sequence = []
    for i in range(len(intervals)):

        token = f"I{intervals[i]}_D{durations[i]}_V{dynamics[i]}"
        sequence.append(token)

    # Check for excessive repetition
    if calculate_repetition_ratio(sequence) > repetition_threshold:
        return None

    return sequence[:max_sequence_length]

def add_musical_context(sequence):
    """Add musical context to reduce repetition"""
    if len(sequence) < 4:
        return sequence

    contextualized = []

    for i, token in enumerate(sequence):
        # Add positional context
        if i == 0:
            contextualized.append(f"START_{token}")
        elif i == len(sequence) - 1:
            contextualized.append(f"END_{token}")
        elif i % 8 == 0:  # Every 8th note (measure-like)
            contextualized.append(f"BEAT_{token}")
        else:
            contextualized.append(token)

    return contextualized

def preprocess_midi_dataset():
    """Main preprocessing function"""

    # Create output directory
    os.makedirs(output, exist_ok=True)

    # Get all MIDI files
    midi_files = []
    for ext in ['*.mid', '*.midi']:
        midi_files.extend(glob.glob(os.path.join(midi_full, '**', ext), recursive=True))

    print(f"Found {len(midi_files)} MIDI files")

    # Randomly sample files to limit processing
    if len(midi_files) > max_files:
        random.seed(42)  # For reproducibility
        midi_files = random.sample(midi_files, max_files)
        print(f"Limited to {max_files} files for processing")

    processed_sequences = []
    failed_files = 0

    for i, file_path in enumerate(midi_files):
        if i % 200 == 0:
            print(f"Processing file {i+1}/{len(midi_files)}")

        notes = extract_notes_from_midi(file_path)

        if notes:
            sequence = normalize_sequence(notes)
            if sequence and len(sequence) >= min_sequence_length:
                # Add musical context to further reduce repetition
                contextualized = add_musical_context(sequence)

                processed_sequences.append({
                    'sequence': contextualized,
                    'filename': os.path.basename(file_path),
                    'length': len(contextualized),
                    'original_length': len(notes)
                })
            else:
                failed_files += 1
        else:
            failed_files += 1

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(processed_sequences)} files")
    print(f"Failed/rejected: {failed_files} files")

    if not processed_sequences:
        print("No valid sequences found! Check your MIDI files and piano detection.")
        return None

    # Save processed data
    output_file = os.path.join(output, 'processed_piano_sequences.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(processed_sequences, f)

    # Create vocabulary from all sequences
    vocab = set()
    for seq_data in processed_sequences:
        vocab.update(seq_data['sequence'])

    vocab = sorted(list(vocab))

    # Create token mappings
    token_to_int = {token: i for i, token in enumerate(vocab)}
    int_to_token = {i: token for token, i in token_to_int.items()}

    # Save vocabulary
    vocab_data = {
        'token_to_int': token_to_int,
        'int_to_token': int_to_token,
        'vocab_size': len(vocab)
    }

    with open(os.path.join(output, 'vocabulary.pkl'), 'wb') as f:
        pickle.dump(vocab_data, f)

    # Save metadata
    lengths = [seq['length'] for seq in processed_sequences]
    metadata = {
        'total_sequences': len(processed_sequences),
        'vocab_size': len(vocab),
        'avg_length': np.mean(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'failed_files': failed_files,
        'repetition_threshold': repetition_threshold
    }

    with open(os.path.join(output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nData saved to: {output_file}")
    print(f"Vocabulary saved with {len(vocab)} unique tokens")
    print(f"Average sequence length: {metadata['avg_length']:.2f}")
    print(f"Vocabulary size: {len(vocab)} tokens")

    return processed_sequences

# Example of how to load the data later
def load_processed_data():
    """Load processed data for training"""
    with open(os.path.join(output, 'processed_piano_sequences.pkl'), 'rb') as f:
        sequences = pickle.load(f)

    with open(os.path.join(output, 'vocabulary.pkl'), 'rb') as f:
        vocab_data = pickle.load(f)

    return sequences, vocab_data

if __name__ == "__main__":
    # Install required packages first:
    # pip install mido numpy

    processed_data = preprocess_midi_dataset()

    if processed_data:
        print("\nPreprocessing successful! You can now use this data for RNN training.")
        print("\nNext steps:")
        print("1. Load data using load_processed_data()")
        print("2. Convert sequences to integers using vocabulary")
        print("3. Create training batches for RNN")
