import torch
import numpy as np
from data.data_loader import DataLoader
from models.neural.grid_encoder import GridEncoder
from models.neural.pattern_recognizer import PatternRecognizer
from models.symbolic.rule_engine import RuleEngine
from models.hybrid.controller import MetaController
from models.hybrid.integrator import Integrator
from data.data_processor import DataProcessor
from utils.metrics import accuracy_score, top_k_error_rate
from utils.visualization import plot_grid, compare_grids
# Load data
data_loader = DataLoader('dataset')
training_data = data_loader.load_data('training')

training_data = {k: training_data[k] for i, k in enumerate(training_data) if i < 5}  # Load only 5 tasks for now

# Initialize models
encoder = GridEncoder()
recognizer = PatternRecognizer()
rule_engine = RuleEngine()
meta_controller = MetaController(recognizer, rule_engine)
integrator = Integrator(encoder, recognizer, rule_engine)

for task_id, task in training_data.items():
    input_grid = torch.tensor(task['train'][0]['input']).unsqueeze(0).unsqueeze(0).float()
    true_output = np.array(task['train'][0]['output'])

    # Process the input grid using the hybrid integrator
    predicted_output = integrator.process(input_grid)

    # Visualize the input grid
    plot_grid(task['train'][0]['input'], title=f'{task_id} - Input Grid')

    # Visualize and compare the true and predicted outputs
    compare_grids(true_output, predicted_output, task_id=task_id)

    # Evaluate accuracy
    acc = accuracy_score(true_output, predicted_output)
    print(f'Accuracy for {task_id}: {acc:.2%}')

    # Example of a batch evaluation for top-k error rate (if applicable)
    # true_outputs = [np.array(sample['output']) for sample in task['train']]
    # predicted_outputs = [[predicted_output]]  # Assuming only one prediction per sample for now
    # top_k_error = top_k_error_rate(true_outputs, predicted_outputs)
    # print(f'Top-3 Error Rate for {task_id}: {top_k_error:.2%}')