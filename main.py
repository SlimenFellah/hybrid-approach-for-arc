# main.py
from dataloader.arc_data_loader import ARCDataLoader
from arc_solver import ARCSolver

loader = ARCDataLoader(
    challenges_path="data/arc-agi_training_challenges.json",
    solutions_path="data/arc-agi_training_solutions.json"
)
loader.load()

# task = loader.get_task("00576224")

# print("Train Pairs:")
# for i, (inp, out) in enumerate(task.get_train_io()):
#     print(f"\nPair {i+1}:")
#     print("Input:", inp)
#     print("Output:", out)

# print("\nTest Inputs:")
# for test_input in task.get_test_inputs():
#     print(test_input)

# print("\nExpected Outputs:")
# for solution in task.get_test_outputs():
#     print(solution)

# task = loader.get_task("00576224")

# # Visualize training data
# task.visualize_train_pairs()

# # Visualize test input and solution (if available)
# task.visualize_test_cases()

# Solve a specific task
solver = ARCSolver(loader)
preds = solver.solve_task_00576224()

# Show predictions visually
task = loader.get_task("00576224")
for i, pred in enumerate(preds):
    print(f"\nTest {i+1} Prediction:")
    task.visualize_grid(pred, title="Predicted Output")

    if i < len(task.get_test_outputs()):
        task.visualize_grid(task.get_test_outputs()[i], title="Expected Output")