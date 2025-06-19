# main.py
from dataloader.arc_data_loader import ARCDataLoader

loader = ARCDataLoader(
    challenges_path="data/arc-agi_training_challenges.json",
    solutions_path="data/arc-agi_training_solutions.json"
)
loader.load()

task = loader.get_task("00576224")
print("Train Pairs:")
for i, (inp, out) in enumerate(task.get_train_io()):
    print(f"\nPair {i+1}:")
    print("Input:", inp)
    print("Output:", out)

print("\nTest Inputs:")
for test_input in task.get_test_inputs():
    print(test_input)

print("\nExpected Outputs:")
for solution in task.get_test_outputs():
    print(solution)
