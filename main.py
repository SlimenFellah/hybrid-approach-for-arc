# import torch
# import numpy as np
# from data.data_loader import DataLoader
# from models.neural.grid_encoder import GridEncoder
# from models.neural.pattern_recognizer import PatternRecognizer
# from models.symbolic.rule_engine import RuleEngine
# from models.hybrid.controller import MetaController
# from models.hybrid.integrator import Integrator
# from data.old.data_processor import DataProcessor
# from utils.metrics import accuracy_score, top_k_error_rate
# from utils.visualization import plot_grid, compare_grids
# # Load data
# data_loader = DataLoader('dataset')
# training_data = data_loader.load_data('training')

# training_data = {k: training_data[k] for i, k in enumerate(training_data) if i < 5}  # Load only 5 tasks for now

# # Initialize models
# encoder = GridEncoder()
# recognizer = PatternRecognizer()
# rule_engine = RuleEngine()
# meta_controller = MetaController(recognizer, rule_engine)
# integrator = Integrator(encoder, recognizer, rule_engine)

# for task_id, task in training_data.items():
#     input_grid = torch.tensor(task['train'][0]['input']).unsqueeze(0).unsqueeze(0).float()
#     true_output = np.array(task['train'][0]['output'])

#     # Process the input grid using the hybrid integrator
#     predicted_output = integrator.process(input_grid)

#     # Visualize the input grid
#     plot_grid(task['train'][0]['input'], title=f'{task_id} - Input Grid')

#     # Visualize and compare the true and predicted outputs
#     compare_grids(true_output, predicted_output, task_id=task_id)

#     # Evaluate accuracy
#     acc = accuracy_score(true_output, predicted_output)
#     print(f'Accuracy for {task_id}: {acc:.2%}')

#     # Example of a batch evaluation for top-k error rate (if applicable)
#     # true_outputs = [np.array(sample['output']) for sample in task['train']]
#     # predicted_outputs = [[predicted_output]]  # Assuming only one prediction per sample for now
#     # top_k_error = top_k_error_rate(true_outputs, predicted_outputs)
#     # print(f'Top-3 Error Rate for {task_id}: {top_k_error:.2%}')


import argparse
from pathlib import Path
from src.config import Config
from src.data_handling.data_loader import ARCDataLoader
from src.meta_learning.controller import ARCController
from src.visualization.grid_viz import GridVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='ARC Solver')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'test'], 
                       default='train', help='Operation mode')
    parser.add_argument('--visualize', action='store_true', 
                       help='Visualize solutions')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize components
    data_loader = ARCDataLoader(
        Config.TRAINING_DIR if args.mode == 'train'
        else Config.EVALUATION_DIR if args.mode == 'evaluate'
        else Config.TEST_DIR
    )
    controller = ARCController()
    visualizer = GridVisualizer()
    
    # Load tasks
    tasks = data_loader.load_all_tasks()
    
    results = []
    for task_id, task in enumerate(tasks):
        print(f"Processing task {task_id}...")
        
        # Get train/test pairs
        train_pairs, test_pairs = data_loader.get_train_test_pairs(task)
        
        # Process each test input
        for test_idx, (test_input, expected_output) in enumerate(test_pairs):
            # Generate solution
            solution = controller.solve_task(train_pairs, test_input)
            
            # Visualize if requested
            if args.visualize:
                visualizer.visualize_sample(
                    test_input, 
                    expected_output if expected_output is not None else solution,
                    solution
                )
            
            # Store results
            results.append({
                'task_id': task_id,
                'test_id': test_idx,
                'solution': solution,
                'expected': expected_output
            })
    
    # Save results if needed
    if args.mode == 'test':
        save_submission(results)

def save_submission(results):
    """Save results in competition submission format"""
    import pandas as pd
    
    submissions = []
    for result in results:
        task_id = f"{result['task_id']:06d}"
        prediction = result['solution'].tolist()
        submissions.append({
            'output_id': f'{task_id}_{result["test_id"]}',
            'output': prediction
        })
    
    df = pd.DataFrame(submissions)
    df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()