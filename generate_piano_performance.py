
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import mido
import json
import random
import math

# Configuration
prepared_data_dir = "D:/music-evolution/Data/rnn_training_data"
model_save_dir = "D:/music-evolution/Models"

class PianoPerformanceMIDIGenerator:
    def __init__(self):
        self.model = None
        self.vocab_data = None
        self.sequence_length = None

        # Enhanced duration mappings with performance variations
        self.duration_map = {
            'short': 180,      # Staccato 8th
            'medium': 400,     # Legato quarter  
            'long': 800,       # Sustained half
            'very_long': 1600  # Held whole
        }

        # Extended velocity map for expression
        self.velocity_map = {
            'pp': 25,   # Very soft
            'p': 45,    # Soft
            'mp': 65,   # Medium soft
            'mf': 80,   # Medium
            'f': 95,    # Strong
            'ff': 115   # Very strong
        }

        # Performance chord voicings (more sophisticated)
        self.chord_voicings = {
            'major': [0, 4, 7, 12],           # Major with octave
            'minor': [0, 3, 7, 12],           # Minor with octave  
            'major7': [0, 4, 7, 11, 14],      # Extended major 7th
            'minor7': [0, 3, 7, 10, 15],      # Extended minor 7th
            'major9': [0, 4, 7, 11, 14, 17],  # Major 9th
            'minor9': [0, 3, 7, 10, 14, 17],  # Minor 9th
            'sus2': [0, 2, 7, 12],            # Sus2 with octave
            'sus4': [0, 5, 7, 12],            # Sus4 with octave
            'dim7': [0, 3, 6, 9],             # Diminished 7th
            'aug': [0, 4, 8, 12],             # Augmented with octave
        }

        self.base_pitch = 60  # Middle C
        self.current_tempo_factor = 1.0  # For rubato
        self.phrase_intensity = 0.5      # Current emotional intensity

    def load_trained_model(self, model_path=None):
        """Load the pre-trained RNN model"""
        if model_path is None:
            model_path = os.path.join(model_save_dir, 'balanced_piano_rnn_model.h5')

        try:
            self.model = load_model(model_path)
            print(f"✓ Performance model loaded from: {model_path}")
        except FileNotFoundError:
            print(f"❌ Model file not found: {model_path}")
            return False

        training_file = os.path.join(prepared_data_dir, 'rnn_training_data.pkl')
        try:
            with open(training_file, 'rb') as f:
                data = pickle.load(f)
            self.vocab_data = data['vocab_data']
            self.sequence_length = data['sequence_length']
            print(f"✓ Vocabulary loaded - Size: {self.vocab_data['vocab_size']}")
        except FileNotFoundError:
            print(f"❌ Training data not found: {training_file}")
            return False

        return True

    def apply_rubato(self, base_time, phrase_position):
        """Apply subtle tempo variations (rubato) like a human performer"""
        # Create natural tempo variations
        phrase_factor = math.sin(phrase_position * math.pi) * 0.15  # Speed up/slow down within phrases
        random_variation = (random.random() - 0.5) * 0.1  # Small random variations

        tempo_multiplier = 1.0 + phrase_factor + random_variation
        return int(base_time * tempo_multiplier)

    def apply_dynamic_expression(self, base_velocity, phrase_position, note_importance):
        """Apply dynamic expression like a performing artist"""
        # Phrase shaping (crescendo/diminuendo)
        phrase_curve = math.sin(phrase_position * math.pi) * 20

        # Note importance (melody notes louder)
        importance_boost = note_importance * 15

        # Random micro-dynamics
        micro_dynamics = (random.random() - 0.5) * 10

        # Emotional intensity variations
        intensity_factor = self.phrase_intensity * 20

        final_velocity = base_velocity + phrase_curve + importance_boost + micro_dynamics + intensity_factor
        return max(20, min(127, int(final_velocity)))

    def generate_arpeggiation(self, chord_notes, arp_type="up"):
        """Generate arpeggiated chord patterns"""
        arpeggio_notes = []

        if arp_type == "up":
            note_order = sorted(chord_notes)
        elif arp_type == "down":
            note_order = sorted(chord_notes, reverse=True)
        elif arp_type == "up_down":
            ascending = sorted(chord_notes)
            note_order = ascending + ascending[:-1][::-1]
        else:  # "random"
            note_order = chord_notes.copy()
            random.shuffle(note_order)

        return note_order

    def add_grace_notes(self, main_pitch, start_time, main_velocity):
        """Add grace notes for expressiveness"""
        grace_notes = []

        if random.random() < 0.3:  # 30% chance of grace notes
            # Upper or lower neighbor
            grace_pitch = main_pitch + random.choice([-2, -1, 1, 2])
            grace_pitch = max(21, min(108, grace_pitch))

            grace_notes.append({
                'pitch': grace_pitch,
                'velocity': main_velocity - 20,
                'start_time': start_time - 60,  # Before main note
                'duration': 60
            })

        return grace_notes

    def create_performance_phrases(self, notes):
        """Organize notes into musical phrases with natural breathing"""
        phrases = []
        current_phrase = []
        phrase_length = 0

        for note in notes:
            current_phrase.append(note)
            phrase_length += note['duration']

            # End phrase after 4-8 beats or large interval jump
            if (phrase_length > 1920 or  # About 4 beats
                len(current_phrase) > 12 or
                (len(current_phrase) > 4 and random.random() < 0.3)):

                phrases.append(current_phrase.copy())
                current_phrase = []
                phrase_length = 0

        if current_phrase:
            phrases.append(current_phrase)

        return phrases

    def add_pedaling_effects(self, notes):
        """Simulate sustain pedal usage"""
        pedaled_notes = []

        for i, note in enumerate(notes):
            # Extend duration slightly for pedal resonance on important notes
            if (i % 4 == 0 or  # Strong beats
                note['velocity'] > 90 or  # Loud notes
                random.random() < 0.4):  # Some random notes

                # Create overlapping resonance
                resonance_note = note.copy()
                resonance_note['velocity'] = max(15, note['velocity'] - 40)
                resonance_note['start_time'] = note['start_time'] + note['duration'] - 100
                resonance_note['duration'] = min(480, note['duration'])

                pedaled_notes.append(resonance_note)

        return pedaled_notes

    def tokens_to_performance_notes(self, tokens):
        """Convert tokens to expressive performance with artist-like qualities"""
        melody_notes = []
        current_pitch = self.base_pitch
        current_time = 0
        phrase_position = 0.0

        print("🎭 Converting tokens to expressive performance...")

        for i, token in enumerate(tokens):
            interval, duration_key, velocity_key, is_chord = self.parse_token_for_performance(token)

            # Update pitch and phrase position
            current_pitch += interval
            current_pitch = max(21, min(108, current_pitch))
            phrase_position = (i % 32) / 32.0  # 32-note phrases

            # Get base values
            base_duration = self.duration_map.get(duration_key, 480)
            base_velocity = self.velocity_map.get(velocity_key, 70)

            # Apply performance techniques
            performance_duration = self.apply_rubato(base_duration, phrase_position)
            note_importance = 1.0 if not is_chord else 0.7
            performance_velocity = self.apply_dynamic_expression(
                base_velocity, phrase_position, note_importance
            )

            if is_chord and random.random() < 0.6:
                # Generate expressive chord
                chord_type = random.choice(['major', 'minor', 'major7', 'minor7', 'major9', 'sus2', 'sus4'])
                chord_notes = self.generate_performance_chord(current_pitch, chord_type)

                # Sometimes arpeggiate chords
                if random.random() < 0.4:
                    arp_type = random.choice(["up", "down", "up_down"])
                    arpeggiated = self.generate_arpeggiation(chord_notes, arp_type)

                    for j, arp_pitch in enumerate(arpeggiated):
                        arp_time = current_time + (j * 80)  # Stagger timing
                        arp_velocity = performance_velocity - (j * 3)

                        melody_notes.append({
                            'pitch': arp_pitch,
                            'velocity': max(30, arp_velocity),
                            'start_time': arp_time,
                            'duration': performance_duration // 2
                        })
                else:
                    # Block chord with expression
                    for j, chord_pitch in enumerate(chord_notes):
                        chord_velocity = performance_velocity - (j * 2)
                        chord_time = current_time + (j * 15)  # Slight roll

                        melody_notes.append({
                            'pitch': chord_pitch,
                            'velocity': max(30, chord_velocity),
                            'start_time': chord_time,
                            'duration': performance_duration
                        })

                        # Add grace notes occasionally
                        grace_notes = self.add_grace_notes(chord_pitch, chord_time, chord_velocity)
                        melody_notes.extend(grace_notes)
            else:
                # Single melodic note with expression
                main_note = {
                    'pitch': current_pitch,
                    'velocity': performance_velocity,
                    'start_time': current_time,
                    'duration': performance_duration
                }
                melody_notes.append(main_note)

                # Add grace notes
                grace_notes = self.add_grace_notes(current_pitch, current_time, performance_velocity)
                melody_notes.extend(grace_notes)

            current_time += performance_duration

            # Update phrase intensity
            self.phrase_intensity += random.choice([-0.02, -0.01, 0.01, 0.02])
            self.phrase_intensity = max(0.2, min(0.8, self.phrase_intensity))

            if i % 50 == 0:
                print(f"   Processed {i}/{len(tokens)} tokens...")

        # Add sophisticated bass and harmony
        print("🎵 Adding expressive bass line...")
        bass_notes = self.add_performance_bass(melody_notes)

        print("🎶 Adding sophisticated harmony...")
        harmony_notes = self.add_performance_harmony(melody_notes)

        # Add pedaling effects
        print("🎹 Adding sustain pedal effects...")
        pedal_effects = self.add_pedaling_effects(melody_notes)

        # Combine all notes
        all_notes = melody_notes + bass_notes + harmony_notes + pedal_effects
        all_notes.sort(key=lambda x: x['start_time'])

        print(f"✓ Created expressive performance with {len(all_notes)} total notes")
        return all_notes

    def parse_token_for_performance(self, token):
        """Parse token with performance considerations"""
        try:
            # Handle special markers
            is_strong_beat = token.startswith('BEAT_') or token.startswith('START_')

            if token.startswith('START_') or token.startswith('BEAT_') or token.startswith('END_'):
                parts = token.split('_', 1)
                if len(parts) > 1:
                    token = parts[1]
                else:
                    return 0, 'medium', 'mf', is_strong_beat

            if '_D' in token and '_V' in token:
                parts = token.split('_')
                interval = int(parts[0][1:]) if parts[0].startswith('I') else 0
                duration = parts[1][1:] if parts[1].startswith('D') else 'medium'
                velocity = parts[2][1:] if parts[2].startswith('V') else 'mf'

                # Performance decision for chords
                is_chord = (abs(interval) > 5 or is_strong_beat or 
                           velocity in ['f', 'ff'] or random.random() < 0.25)

                return interval, duration, velocity, is_chord
            else:
                return 0, 'medium', 'mf', False

        except (ValueError, IndexError):
            return 0, 'medium', 'mf', False

    def generate_performance_chord(self, root_pitch, chord_type):
        """Generate performance-ready chord voicings"""
        intervals = self.chord_voicings.get(chord_type, self.chord_voicings['major'])
        chord_notes = []

        for interval in intervals:
            note_pitch = root_pitch + interval
            if 21 <= note_pitch <= 108:
                chord_notes.append(note_pitch)

        return chord_notes[:4]  # Limit to 4 notes for playability

    def add_performance_bass(self, melody_notes):
        """Add sophisticated bass line like a concert pianist"""
        bass_notes = []
        last_bass_time = 0

        for i, note in enumerate(melody_notes[::4]):  # Every 4th melody note
            if note['start_time'] >= last_bass_time + 960:  # Don't overlap too much
                # Choose bass pattern
                bass_pattern = random.choice(['root', 'walking', 'alberti'])

                if bass_pattern == 'walking':
                    # Walking bass line
                    for j in range(3):
                        bass_pitch = note['pitch'] - 12 - j * 2
                        bass_pitch = max(21, min(50, bass_pitch))

                        bass_notes.append({
                            'pitch': bass_pitch,
                            'velocity': note['velocity'] - 25,
                            'start_time': note['start_time'] + (j * 320),
                            'duration': 300
                        })
                elif bass_pattern == 'alberti':
                    # Alberti bass pattern
                    pattern = [0, 7, 3, 7]  # Root, fifth, third, fifth
                    for j, interval in enumerate(pattern):
                        bass_pitch = note['pitch'] - 12 + interval
                        bass_pitch = max(21, min(60, bass_pitch))

                        bass_notes.append({
                            'pitch': bass_pitch,
                            'velocity': note['velocity'] - 30,
                            'start_time': note['start_time'] + (j * 120),
                            'duration': 100
                        })
                else:  # root
                    bass_pitch = note['pitch'] - 12
                    bass_pitch = max(21, min(50, bass_pitch))

                    bass_notes.append({
                        'pitch': bass_pitch,
                        'velocity': note['velocity'] - 20,
                        'start_time': note['start_time'],
                        'duration': note['duration'] * 2
                    })

                last_bass_time = note['start_time']

        return bass_notes

    def add_performance_harmony(self, melody_notes):
        """Add sophisticated harmony like a professional pianist"""
        harmony_notes = []

        for i, note in enumerate(melody_notes[::2]):  # Every other note
            if random.random() < 0.4:  # 40% of notes get harmony
                # Choose harmony type
                harmony_type = random.choice(['thirds', 'sixths', 'octaves', 'fourths'])

                if harmony_type == 'thirds':
                    harmony_pitch = note['pitch'] - random.choice([3, 4])  # Major/minor third below
                elif harmony_type == 'sixths':
                    harmony_pitch = note['pitch'] + random.choice([8, 9])  # Major/minor sixth above
                elif harmony_type == 'octaves':
                    harmony_pitch = note['pitch'] + 12  # Octave above
                else:  # fourths
                    harmony_pitch = note['pitch'] + 5  # Perfect fourth above

                if 21 <= harmony_pitch <= 108:
                    harmony_notes.append({
                        'pitch': harmony_pitch,
                        'velocity': note['velocity'] - 20,
                        'start_time': note['start_time'] + random.randint(0, 50),
                        'duration': note['duration']
                    })

        return harmony_notes

    def create_midi_file(self, notes, output_path, tempo_bpm=120):
        """Create expressive MIDI file"""
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Set initial tempo
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_bpm)))

        # Set grand piano
        track.append(mido.Message('program_change', program=0))

        # Add expression controller for dynamics
        track.append(mido.Message('control_change', control=11, value=80))

        # Convert to MIDI events
        midi_events = []

        for note in notes:
            midi_events.append({
                'time': note['start_time'],
                'type': 'note_on',
                'note': note['pitch'],
                'velocity': note['velocity']
            })

            midi_events.append({
                'time': note['start_time'] + note['duration'],
                'type': 'note_off',
                'note': note['pitch'],
                'velocity': 0
            })

        midi_events.sort(key=lambda x: (x['time'], x['type']))

        # Add to track
        last_time = 0
        for event in midi_events:
            delta_time = max(0, event['time'] - last_time)

            if event['type'] == 'note_on':
                track.append(mido.Message('note_on', 
                                        note=event['note'], 
                                        velocity=event['velocity'], 
                                        time=delta_time))
            else:
                track.append(mido.Message('note_off', 
                                        note=event['note'], 
                                        velocity=0, 
                                        time=delta_time))

            last_time = event['time']

        mid.save(output_path)
        print(f"🎭 Expressive performance MIDI saved to: {output_path}")
        return output_path

    # Include generate_music and other methods from previous version...
    def generate_music(self, seed_sequence=None, num_notes=200, temperature=1.0, seed_type="random"):
        """Generate music tokens"""
        if self.model is None or self.vocab_data is None:
            print("❌ Model not loaded!")
            return None

        token_to_int = self.vocab_data['token_to_int']
        int_to_token = self.vocab_data['int_to_token']

        if seed_sequence is None:
            training_file = os.path.join(prepared_data_dir, 'rnn_training_data.pkl')
            with open(training_file, 'rb') as f:
                data = pickle.load(f)

            random_idx = np.random.randint(0, len(data['X_train']))
            seed_sequence = data['X_train'][random_idx].tolist()
            print(f"🎵 Using seed sequence (index: {random_idx})")

        generated_sequence = seed_sequence.copy()
        print(f"🎼 Generating {num_notes} notes for performance...")

        for i in range(num_notes):
            if i % 50 == 0:
                print(f"   Progress: {i}/{num_notes} notes...")

            input_seq = np.array(generated_sequence[-self.sequence_length:]).reshape(1, -1)
            predictions = self.model.predict(input_seq, verbose=0)[0]

            predictions = np.log(predictions + 1e-8) / temperature
            exp_preds = np.exp(predictions)
            predictions = exp_preds / np.sum(exp_preds)

            next_token = np.random.choice(len(predictions), p=predictions)
            generated_sequence.append(next_token)

        generated_tokens = []
        for token_int in generated_sequence:
            if str(token_int) in int_to_token:
                generated_tokens.append(int_to_token[str(token_int)])
            else:
                generated_tokens.append("I0_Dmedium_Vmf")

        return generated_tokens

    def generate_performance_midi(self, filename=None, num_notes=250, temperature=0.9, tempo_bpm=120):
        """Generate a full expressive piano performance"""
        if filename is None:
            filename = f"piano_performance_{np.random.randint(1000, 9999)}.mid"

        output_path = os.path.join(model_save_dir, filename)

        tokens = self.generate_music(num_notes=num_notes, temperature=temperature)
        if tokens is None:
            return None

        print("🎭 Creating expressive piano performance...")
        notes = self.tokens_to_performance_notes(tokens)

        print("🎵 Rendering performance MIDI...")
        midi_path = self.create_midi_file(notes, output_path, tempo_bpm)

        return midi_path

def main():
    """Generate expressive piano performances"""
    print("="*70)
    print("🎭 EXPRESSIVE PIANO PERFORMANCE GENERATOR")
    print("🎹 Creates music like a piano artist performing live")
    print("="*70)

    generator = PianoPerformanceMIDIGenerator()

    if not generator.load_trained_model():
        print("❌ Failed to load model. Exiting...")
        return

    print("\n🎼 GENERATING EXPRESSIVE PERFORMANCES...")

    # Generate different performance styles
    performances = [
        ("classical_performance.mid", 280, 0.7, 100, "Classical Style"),
        ("romantic_performance.mid", 300, 1.0, 90, "Romantic Style"), 
        ("jazz_performance.mid", 250, 1.2, 120, "Jazz Style"),
        ("contemporary_performance.mid", 320, 0.9, 110, "Contemporary Style")
    ]

    generated_files = []

    for filename, num_notes, temp, tempo, style in performances:
        print(f"\n🎭 Creating {style} performance...")
        midi_path = generator.generate_performance_midi(
            filename=filename,
            num_notes=num_notes,
            temperature=temp,
            tempo_bpm=tempo
        )
        if midi_path:
            generated_files.append(filename)

    print("\n" + "="*70)
    print("✅ EXPRESSIVE PERFORMANCES COMPLETE!")
    print("="*70)
    print("🎵 Generated performance files:")
    for file in generated_files:
        print(f"   - {file}")

    print("\n🎭 Performance features included:")
    print("   ✓ Dynamic expression and phrasing")
    print("   ✓ Rubato (tempo variations)")
    print("   ✓ Grace notes and ornamentation")
    print("   ✓ Sophisticated chord voicings")
    print("   ✓ Arpeggiated passages") 
    print("   ✓ Expressive bass lines")
    print("   ✓ Professional harmony")
    print("   ✓ Sustain pedal effects")
    print("   ✓ Natural breathing between phrases")

    print("\n🎼 Perfect for:")
    print("   - Concert hall performances")
    print("   - Movie soundtracks") 
    print("   - Emotional background music")
    print("   - Professional piano recordings")

if __name__ == "__main__":
    main()
