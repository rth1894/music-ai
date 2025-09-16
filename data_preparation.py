import os
import pickle
import numpy as np
import json
from collections import Counter
import random

output_dir = "D:/music-evolution/Data/process1_notes"
prepared_data_dir = "D:/music-evolution/Data/rnn_training_data"
num_best_sequences = 1000
sequence_length = 64  
test_split = 0.2 

def calculate_musical_quality_score(sequence):

    if len(sequence) < 10:
        return 0

    score = 0

    length_score = min(len(sequence) / 200, 1.0) 
    if len(sequence) < 50:
        length_score *= 0.5 
    score += length_score * 20

    unique_tokens = len(set(sequence))
    variety_score = min(unique_tokens / len(sequence), 0.8) 
    score += variety_score * 30

    counter = Counter(sequence)
    most_common_count = counter.most_common(1)[0][1]
    repetition_ratio = most_common_count / len(sequence)
    repetition_score = max(0, 1 - repetition_ratio)
    score += repetition_score * 25
    context_tokens = sum(1 for token in sequence if any(marker in token 
                        for marker in ['START_', 'BEAT_', 'END_']))
    context_score = min(context_tokens / len(sequence) * 10, 1.0)
    score += context_score * 15

    velocity_types = set()
    for token in sequence:
        if '_V' in token:
            velocity = token.split('_V')[1]
            velocity_types.add(velocity)

    dynamic_score = min(len(velocity_types) / 6, 1.0) 
    score += dynamic_score * 10

    return score

def load_and_select_best_data():
    sequences_file = os.path.join(output_dir, 'processed_piano_sequences.pkl')

    try:
        with open(sequences_file, 'rb') as f:
            all_sequences = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {sequences_file}")
        print("Please run the preprocessing script first!")
        return None

    print(f"Loaded {len(all_sequences)} preprocessed sequences")

    sequence_scores = []
    for i, seq_data in enumerate(all_sequences):
        if i % 1000 == 0:
            print(f"Calculating scores... {i}/{len(all_sequences)}")
    
        score = calculate_musical_quality_score(seq_data['sequence'])
        sequence_scores.append((score, i, seq_data))

    sequence_scores.sort(key=lambda x: x[0], reverse=True)
    best_sequences = sequence_scores[:num_best_sequences]

    print(f"Selected top {len(best_sequences)} sequences")
    print(f"Score range: {best_sequences[-1][0]:.2f} to {best_sequences[0][0]:.2f}")

    return [seq_data for score, idx, seq_data in best_sequences]

def create_training_sequences(sequences, vocab_data, seq_length):


    token_to_int = vocab_data['token_to_int']

    X_sequences = []  
    y_sequences = []  

    print(f"Creating training samples from {len(sequences)} sequences...")

    for i, seq_data in enumerate(sequences):
        if i % 100 == 0:
            print(f"Processing sequence {i+1}/{len(sequences)}")
        
        sequence = seq_data['sequence']

        int_sequence = []
        for token in sequence:
            if token in token_to_int:
                int_sequence.append(token_to_int[token])
            else:
                int_sequence.append(0) 

        for j in range(len(int_sequence) - seq_length):
            X_sequences.append(int_sequence[j:j + seq_length])
            y_sequences.append(int_sequence[j + seq_length])

    return np.array(X_sequences), np.array(y_sequences)

def prepare_rnn_training_data():
    os.makedirs(prepared_data_dir, exist_ok=True)
    vocab_file = os.path.join(output_dir, 'vocabulary.pkl')

    try:
        with open(vocab_file, 'rb') as f:
            vocab_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {vocab_file}")
        print("Please run the preprocessing script first!")
        return None

    print(f"Vocabulary size: {vocab_data['vocab_size']}")

    best_sequences = load_and_select_best_data()
    if best_sequences is None:
        return None
    print(f"Creating training sequences with length {sequence_length}...")
    X, y = create_training_sequences(best_sequences, vocab_data, sequence_length)

    print(f"Created {len(X)} training samples")
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")

    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.seed(42)  
    np.random.shuffle(indices)

    test_size = int(num_samples * test_split)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    training_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'vocab_data': vocab_data,
        'sequence_length': sequence_length,
        'num_best_sequences': num_best_sequences
    }

    training_file = os.path.join(prepared_data_dir, 'rnn_training_data.pkl')
    with open(training_file, 'wb') as f:
        pickle.dump(training_data, f)

    best_sequences_info = []
    for seq_data in best_sequences:
        info = {
            'filename': seq_data['filename'],
            'length': seq_data['length'],
            'score': calculate_musical_quality_score(seq_data['sequence'])
        }
        best_sequences_info.append(info)

    best_sequences_file = os.path.join(prepared_data_dir, 'best_sequences_info.json')
    with open(best_sequences_file, 'w') as f:
        json.dump(best_sequences_info, f, indent=2)

    metadata = {
        'total_sequences_used': len(best_sequences),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'sequence_length': sequence_length,
        'vocab_size': vocab_data['vocab_size'],
        'test_split': test_split,
        'avg_sequence_length': np.mean([len(seq['sequence']) for seq in best_sequences]),
        'score_range': {
            'min': min([calculate_musical_quality_score(seq['sequence']) for seq in best_sequences]),
            'max': max([calculate_musical_quality_score(seq['sequence']) for seq in best_sequences])
        }
    }

    metadata_file = os.path.join(prepared_data_dir, 'training_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTraining data saved to: {training_file}")
    print(f"Best sequences info saved to: {best_sequences_file}")
    print(f"Metadata saved to: {metadata_file}")

    return training_data



def create_data_loader_function():
    loader_code = '''
def load_rnn_training_data():
    """Load prepared training data for RNN model"""
    import pickle

    training_file = "D:/music-evolution/Data/rnn_training_data/rnn_training_data.pkl"

    with open(training_file, 'rb') as f:
        data = pickle.load(f)

    return data

def get_data_info():
    """Get information about the prepared data"""
    import json

    metadata_file = "D:/music-evolution/Data/rnn_training_data/training_metadata.json"

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    return metadata

# Example usage:
# data = load_rnn_training_data()
# X_train, y_train = data['X_train'], data['y_train']
# X_test, y_test = data['X_test'], data['y_test']
# vocab_size = data['vocab_data']['vocab_size']
# int_to_token = data['vocab_data']['int_to_token']
# token_to_int = data['vocab_data']['token_to_int']

# info = get_data_info()
# print(f"Vocabulary size: {info['vocab_size']}")
# print(f"Training samples: {info['training_samples']}")
'''

    loader_file = os.path.join(prepared_data_dir, 'data_loader.py')
    with open(loader_file, 'w') as f:
        f.write(loader_code)

    print(f"Data loader function saved to: {loader_file}")

if __name__ == "__main__":
    print("="*60)
    print("PREPARING RNN TRAINING DATA")
    print("="*60)
    print(f"Selecting best {num_best_sequences} sequences from preprocessed data")
    print(f"Sequence length for RNN: {sequence_length}")
    print(f"Train/test split: {int((1-test_split)*100)}%/{int(test_split*100)}%")
    print("="*60)

    training_data = prepare_rnn_training_data()

    if training_data is not None:
        create_data_loader_function()
        
        print("\n" + "="*60)
        print("DATA PREPARATION COMPLETE!")
        print("="*60)
        print(f"Selected {num_best_sequences} best sequences based on:")
        print("Sequence length (50-200 notes preferred)")
        print("Musical variety (unique token ratio)")
        print("Low repetition ratio")
        print("Presence of musical context markers")
        print("Dynamic range variety")
        print("\nCreated training data:")
        print(f"Training samples: {len(training_data['X_train'])}")
        print(f"Test samples: {len(training_data['X_test'])}")
        print(f"Vocabulary size: {training_data['vocab_data']['vocab_size']}")
        print(f"Sequence length: {sequence_length}")
    else:

        print("Please make sure the preprocessing script has been run first.")
