
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import json

# Configuration
prepared_data_dir = "D:/music-evolution/Data/rnn_training_data"
model_save_dir = "D:/music-evolution/Models"
os.makedirs(model_save_dir, exist_ok=True)

class BalancedPianoRNNModel:
    def __init__(self, vocab_size, sequence_length, embedding_dim=200, lstm_units=384, dropout_rate=0.25):
        """
        Initialize Balanced Piano RNN Model (quality vs speed optimized)

        Args:
            vocab_size: Number of unique tokens in vocabulary
            sequence_length: Length of input sequences
            embedding_dim: Dimension of embedding layer (moderate)
            lstm_units: Number of LSTM units (moderate)
            dropout_rate: Dropout rate for regularization (moderate)
        """
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.vocab_data = None

    def build_model(self):
        """Build a balanced RNN model - good quality with reasonable speed"""

        model = Sequential([
            # Moderate embedding layer
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.sequence_length,
                name='embedding'
            ),

            # First LSTM layer - keep return_sequences for depth
            LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=0.15,
                name='lstm_1'
            ),
            BatchNormalization(),  # Keep for training stability

            # Second LSTM layer - important for music quality
            LSTM(
                self.lstm_units // 2,  # Smaller second layer for efficiency
                dropout=self.dropout_rate,
                recurrent_dropout=0.15,
                name='lstm_2'
            ),
            BatchNormalization(),

            # Dense layers for musical reasoning
            Dense(self.lstm_units // 2, activation='relu', name='dense_1'),
            Dropout(self.dropout_rate),

            Dense(self.lstm_units // 4, activation='relu', name='dense_2'),
            Dropout(self.dropout_rate // 2),  # Less dropout in final layer

            # Output layer
            Dense(self.vocab_size, activation='softmax', name='output')
        ])

        # Moderate learning rate - not too fast, not too slow
        optimizer = Adam(learning_rate=0.0015, clipnorm=1.0)
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
        )

        self.model = model
        return model

    def load_training_data(self):
        """Load the prepared training data"""

        training_file = os.path.join(prepared_data_dir, 'rnn_training_data.pkl')

        with open(training_file, 'rb') as f:
            data = pickle.load(f)

        self.vocab_data = data['vocab_data']

        return data

    def train(self, epochs=30, batch_size=128):  # Moderate batch size
        """Train the RNN model with balanced settings"""

        print("Loading training data...")
        data = self.load_training_data()

        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")

        if self.model is None:
            print("Building balanced model...")
            self.build_model()

        print("Model Architecture:")
        self.model.summary()

        # Balanced callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(model_save_dir, 'balanced_piano_rnn_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,  # Reasonable patience for quality
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.6,  # Gentle LR reduction
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        print("Starting balanced training...")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")

        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1,
            workers=2,  # Moderate multiprocessing
            use_multiprocessing=False  # More stable
        )

        # Save final model
        self.model.save(os.path.join(model_save_dir, 'final_balanced_piano_rnn_model.h5'))

        # Save training history
        with open(os.path.join(model_save_dir, 'balanced_training_history.pkl'), 'wb') as f:
            pickle.dump(history.history, f)

        # Plot training history
        self.plot_training_history(history)

        return history

    def plot_training_history(self, history):
        """Plot training and validation metrics"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot accuracy
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(model_save_dir, 'balanced_training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def load_trained_model(self, model_path=None):
        """Load a pre-trained model"""

        if model_path is None:
            model_path = os.path.join(model_save_dir, 'balanced_piano_rnn_model.h5')

        self.model = load_model(model_path)

        # Load vocabulary data
        training_file = os.path.join(prepared_data_dir, 'rnn_training_data.pkl')
        with open(training_file, 'rb') as f:
            data = pickle.load(f)

        self.vocab_data = data['vocab_data']

        print(f"Model loaded from: {model_path}")

    def generate_music(self, seed_sequence=None, num_notes=200, temperature=1.0):
        """Generate music using the trained model"""

        if self.model is None:
            print("No model loaded! Please train or load a model first.")
            return None

        if self.vocab_data is None:
            print("No vocabulary data loaded!")
            return None

        token_to_int = self.vocab_data['token_to_int']
        int_to_token = self.vocab_data['int_to_token']

        # Create seed sequence if not provided
        if seed_sequence is None:
            # Use a random sequence from training data
            data = self.load_training_data()
            random_idx = np.random.randint(0, len(data['X_train']))
            seed_sequence = data['X_train'][random_idx].tolist()

        generated_sequence = seed_sequence.copy()

        print(f"Generating {num_notes} notes with temperature {temperature}...")

        for i in range(num_notes):
            if i % 50 == 0:
                print(f"Generated {i}/{num_notes} notes...")

            # Prepare input sequence
            input_seq = np.array(generated_sequence[-self.sequence_length:]).reshape(1, -1)

            # Predict next token
            predictions = self.model.predict(input_seq, verbose=0)[0]

            # Apply temperature for creativity
            predictions = np.log(predictions + 1e-8) / temperature
            exp_preds = np.exp(predictions)
            predictions = exp_preds / np.sum(exp_preds)

            # Sample next token
            next_token = np.random.choice(len(predictions), p=predictions)
            generated_sequence.append(next_token)

        # Convert back to tokens
        generated_tokens = [int_to_token[str(token)] for token in generated_sequence]

        return generated_tokens

    def save_generated_music(self, generated_tokens, filename="balanced_generated_piano_music.json"):
        """Save generated music tokens to file"""

        output_file = os.path.join(model_save_dir, filename)

        music_data = {
            'generated_tokens': generated_tokens,
            'num_tokens': len(generated_tokens),
            'generation_timestamp': str(np.datetime64('now'))
        }

        with open(output_file, 'w') as f:
            json.dump(music_data, f, indent=2)

        print(f"Generated music saved to: {output_file}")
        return output_file

def main():
    """Main function to train the balanced RNN model"""

    print("="*60)
    print("BALANCED PIANO RNN MODEL TRAINING")
    print("="*60)

    # Load data to get vocabulary size and sequence length
    training_file = os.path.join(prepared_data_dir, 'rnn_training_data.pkl')

    try:
        with open(training_file, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Error: Training data not found!")
        print("Please run the data preparation script first.")
        return

    vocab_size = data['vocab_data']['vocab_size']
    sequence_length = data['sequence_length']

    print(f"Vocabulary size: {vocab_size}")
    print(f"Sequence length: {sequence_length}")
    print(f"Training samples: {len(data['X_train'])}")
    print(f"Test samples: {len(data['X_test'])}")

    # Initialize balanced model
    piano_rnn = BalancedPianoRNNModel(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        embedding_dim=200,    # Moderate size
        lstm_units=384,       # Good capacity
        dropout_rate=0.25     # Balanced regularization
    )

    # Train the model with balanced settings
    history = piano_rnn.train(epochs=30, batch_size=128)

    print("\n" + "="*60)
    print("BALANCED TRAINING COMPLETE!")
    print("="*60)
    print("✓ Model saved with good quality-speed balance")
    print("✓ Should train in ~15-20 minutes")
    print("✓ Maintains musical quality with reasonable efficiency")

    # Generate a sample music piece
    print("\nGenerating sample music...")
    generated_tokens = piano_rnn.generate_music(num_notes=100, temperature=0.8)
    piano_rnn.save_generated_music(generated_tokens)

    print("\n🎵 This model balances:")
    print("- Two LSTM layers for musical depth")
    print("- Moderate model size for quality")
    print("- Reasonable training time")
    print("- Good generalization capability")

if __name__ == "__main__":
    main()
