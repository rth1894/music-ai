# fluidsynth -i /usr/share/soundfonts/FluidR3_GM.sf2 best_melody.mid
import random
from mido import Message, MidiFile, MidiTrack

POPULATION_SIZE = 20
CHROMOSOME_LENGTH = 50
NUM_GENERATIONS = 30
MUTATION_RATE = 0.1
NOTE_RANGE = (60, 120)

def fitness(chromosome):
    score = 0
    for i in range(1, len(chromosome)):
        interval = abs(chromosome[i] - chromosome[i-1])
        score += max(0, 12 - interval)
    return score

def crossover(parent1, parent2):
    point = random.randint(1, CHROMOSOME_LENGTH - 2)
    child = parent1[:point] + parent2[point:]
    return child

def mutate(chromosome):
    mutated = []
    for note in chromosome:
        if random.random() > MUTATION_RATE:
            mutated.append(note)
        else:
            mutated.append(random.randint(*NOTE_RANGE))
    return mutated

def generate_chromosome():
    chromosome = []
    for _ in range(CHROMOSOME_LENGTH):
        note = random.randint(*NOTE_RANGE)
        chromosome.append(note)
    return chromosome

population = []
for i in range(POPULATION_SIZE):
    chromosome = generate_chromosome()
    population.append(chromosome)

for generation in range(NUM_GENERATIONS):
    scored = [(fitness(c), c) for c in population]
    scored.sort(reverse=True)
    population = []
    for i, c in scored:
        population.append(c)

    print(f"Generation {generation + 1}: Best fitness = {scored[0][0]}")

    new_population = population[:2]

    while len(new_population) < POPULATION_SIZE:
        parent1, parent2 = random.choices(population[:10], k=2)
        child = mutate(crossover(parent1, parent2))
        new_population.append(child)

    population = new_population

best = population[0]
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

for note in best:
    track.append(Message('note_on', note=note, velocity=64, time=0))
    track.append(Message('note_off', note=note, velocity=64, time=480))

mid.save('mid/best.mid')
print("best saved to best.mid")
