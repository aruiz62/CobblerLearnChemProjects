
import random
import string
import matplotlib.pyplot as plt
from pathlib import Path

# STEP 1: Set up the target phrase
target = "METHINKS IT IS LIKE A WEASEL"
characters = string.ascii_uppercase + " "

# Create or locate the Weasel folder
if "__file__" in globals():
    folder = Path(__file__).resolve().parent
else:
    folder = Path.cwd() / "Weasel"

folder.mkdir(parents=True, exist_ok=True)


# STEP 2: Fitness function
# Gives 1 point for each matching character
def fitness(candidate):
    score = 0

    for a, b in zip(candidate, target):
        if a == b:
            score += 1

    return score


# STEP 3: Mutation function
# Each character has a 5% chance of changing
def mutate(parent):
    new_string = ""

    for letter in parent:
        if random.random() < 0.05:
            new_string += random.choice(characters)
        else:
            new_string += letter

    return new_string


# STEP 4: Evolution loop

# Start with a completely random phrase
best_parent = "".join(
    random.choice(characters)
    for _ in range(len(target))
)

generation = 0
best_score = fitness(best_parent)

# Store results for the graph
generations = [0]
fitness_scores = [best_score]

# Store results for the text file
history = []

first_result = f"Generation 0 | Score: {best_score}/28 | {best_parent}"
print(first_result)
history.append(first_result)

# Keep evolving until the phrase is perfect
while best_score < len(target) and generation < 10000:

    generation += 1

    # Create 100 mutated offspring
    offspring = [
        mutate(best_parent)
        for _ in range(100)
    ]

    # Select the offspring with the highest fitness
    best_child = max(offspring, key=fitness)
    child_score = fitness(best_child)

    # Replace parent only if offspring is better
    if child_score > best_score:
        best_parent = best_child
        best_score = child_score

    # Save fitness for graph
    generations.append(generation)
    fitness_scores.append(best_score)

    # Print each generation
    result = (
        f"Generation {generation} | "
        f"Score: {best_score}/28 | "
        f"{best_parent}"
    )

    print(result)
    history.append(result)


# STEP 5: Final results
print("\nEVOLUTION COMPLETE")
print("Final phrase:", best_parent)
print("Final fitness:", best_score, "/28")
print("Total generations:", generation)


# STEP 6: Save all generations to a text file
text_file = folder / "weasel_generations.txt"

with open(text_file, "w") as file:
    for line in history:
        file.write(line + "\n")

print("Generations saved to:", text_file)


# STEP 6: Create fitness graph
plt.figure(figsize=(10, 6))

plt.plot(
    generations,
    fitness_scores,
    color="blue",
    linewidth=2
)

plt.title("Weasel Program: Fitness Over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness Score")
plt.ylim(0, 29)
plt.grid(True, alpha=0.3)

# Save graph automatically
graph_file = folder / "weasel_fitness.png"

plt.savefig(graph_file, dpi=300, bbox_inches="tight")
plt.show()

print("Graph saved to:", graph_file)
