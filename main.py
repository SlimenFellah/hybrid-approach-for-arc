# main.py
from dataloader.arc_data_loader import ARCDataLoader
from arc_solver import ARCSolver
from evaluation.evaluation import evaluate_task, print_evaluation_summary

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

# Show predictions visually
task_id = "00576224"
task = loader.get_task(task_id)

predictions = solver.solve_task_00576224()
correct, total = evaluate_task(task, predictions)

print_evaluation_summary(task_id, correct, total)

for i, pred in enumerate(predictions):
    task.visualize_grid(pred, title=f"Predicted Output {i+1}")

    if i < len(task.get_test_outputs()):
        task.visualize_grid(task.get_test_outputs()[i], title=f"Expected Output {i+1}")