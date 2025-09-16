import random
from mido import Message, MidiFile, MidiTrack

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

NUM_NOTES = 50
MIN_NOTE = 60
MAX_NOTE = 127
MIN_DURATION = 100
MAX_DURATION = 400

for i in range(NUM_NOTES):
    note = random.randint(MIN_NOTE, MAX_NOTE)
    velocity = random.randint(50, 100)
    duration = random.randint(MIN_DURATION, MAX_DURATION)

    track.append(Message('note_on', note=note, velocity=velocity, time=0))
    track.append(Message('note_off', note=note, velocity=velocity, time=duration))

mid.save('mid/random_notes.mid')
