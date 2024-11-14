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


# import argparse
# from pathlib import Path
# from config import Config
# from data.data_loader import ARCDataLoader
# from meta_learning.controller import ARCController
# from visualization.grid_viz import GridVisualizer

# def parse_args():
#     parser = argparse.ArgumentParser(description='ARC Solver')
#     parser.add_argument('--mode', choices=['train', 'evaluate', 'test'], 
#                        default='train', help='Operation mode')
#     parser.add_argument('--visualize', action='store_true', 
#                        help='Visualize solutions')
#     return parser.parse_args()

# def main():
#     args = parse_args()
    
#     # Initialize components
#     data_loader = ARCDataLoader(
#         Config.TRAINING_DIR if args.mode == 'train'
#         else Config.EVALUATION_DIR if args.mode == 'evaluate'
#         else Config.TEST_DIR
#     )
#     controller = ARCController()
#     visualizer = GridVisualizer()
    
#     # Load tasks
#     tasks = data_loader.load_all_tasks()
    
#     results = []
#     for task_id, task in enumerate(tasks):
#         print(f"Processing task {task_id}...")
        
#         # Get train/test pairs
#         train_pairs, test_pairs = data_loader.get_train_test_pairs(task)
        
#         # Process each test input
#         for test_idx, (test_input, expected_output) in enumerate(test_pairs):
#             # Generate solution
#             solution = controller.solve_task(train_pairs, test_input)
            
#             # Visualize if requested
#             if args.visualize:
#                 visualizer.visualize_sample(
#                     test_input, 
#                     expected_output if expected_output is not None else solution,
#                     solution
#                 )
            
#             # Store results
#             results.append({
#                 'task_id': task_id,
#                 'test_id': test_idx,
#                 'solution': solution,
#                 'expected': expected_output
#             })
    
#     # Save results if needed
#     if args.mode == 'test':
#         save_submission(results)

# def save_submission(results):
#     """Save results in competition submission format"""
#     import pandas as pd
    
#     submissions = []
#     for result in results:
#         task_id = f"{result['task_id']:06d}"
#         prediction = result['solution'].tolist()
#         submissions.append({
#             'output_id': f'{task_id}_{result["test_id"]}',
#             'output': prediction
#         })
    
#     df = pd.DataFrame(submissions)
#     df.to_csv('submission.csv', index=False)

# if __name__ == '__main__':
#     main()


import os
from pathlib import Path
from data.data_loader import ARCDataLoader
from data.data_preprocessor import ARCPreprocessor
from feature_extraction.neural_features import CNNFeatureExtractor
from feature_extraction.symbolic_features import SymbolicFeatureExtractor
from feature_extraction.statistical_features import StatisticalFeatureExtractor
from pattern_recognition.deep_learning import PatternRecognitionModel
from pattern_recognition.rule_based import RuleBasedPatternRecognizer
from pattern_recognition.heuristics import HeuristicEngine
from meta_learning.controller import ARCController
# from solution_generation.program_synthesis import ProgramSynthesis
from solution_generation.program_synthesis import ProgramSynthesizer
from solution_generation.neural_decoder import NeuralDecoder
from solution_generation.transformation_library import TransformationLibrary
from visualization.grid_viz import GridVisualizer
from config import Config

def main():
    # Load and preprocess data
    data_loader = ARCDataLoader(Config.TRAINING_DIR)
    training_tasks = data_loader.load_all_tasks()
    
    data_loader = ARCDataLoader(Config.EVALUATION_DIR)
    evaluation_tasks = data_loader.load_all_tasks()
    
    data_loader = ARCDataLoader(Config.TEST_DIR)
    test_tasks = data_loader.load_all_tasks()

    # Feature extraction
    neural_extractor = CNNFeatureExtractor()
    symbolic_extractor = SymbolicFeatureExtractor()
    statistical_extractor = StatisticalFeatureExtractor()

    # Pattern recognition models
    deep_learning_model = PatternRecognitionModel()
    rule_based_model = RuleBasedPatternRecognizer()
    heuristic_model = HeuristicEngine()

    # Meta-learning controller
    meta_controller = ARCController()

    # Solution generation components
    program_synthesis = ProgramSynthesizer()
    neural_decoder = NeuralDecoder()
    transformation_library = TransformationLibrary()

    # Iterate through training tasks
    for task in training_tasks:
        train_pairs, test_pairs = data_loader.get_train_test_pairs(task)

        # Extract features
        neural_features = neural_extractor.extract_features(train_pairs)
        symbolic_features = symbolic_extractor.extract_features(train_pairs)
        statistical_features = statistical_extractor.extract_features(train_pairs)

        # Recognize patterns
        deep_learning_output = deep_learning_model.predict(neural_features)
        rule_based_output = rule_based_model.predict(symbolic_features)
        heuristic_output = heuristic_model.predict(statistical_features)

        # Meta-learning
        meta_controller.learn_from_task(task, deep_learning_output, rule_based_output, heuristic_output)

        # Generate solutions
        program_synthesis.generate_solutions(task, train_pairs)
        neural_decoder.generate_solutions(task, train_pairs)
        transformation_library.generate_solutions(task, train_pairs)

    # Evaluate on test tasks
    for task in test_tasks:
        train_pairs, test_pairs = data_loader.get_train_test_pairs(task)

        # Extract features
        neural_features = neural_extractor.extract_features(train_pairs)
        symbolic_features = symbolic_extractor.extract_features(train_pairs)
        statistical_features = statistical_extractor.extract_features(train_pairs)

        # Recognize patterns
        deep_learning_output = deep_learning_model.predict(neural_features)
        rule_based_output = rule_based_model.predict(symbolic_features)
        heuristic_output = heuristic_model.predict(statistical_features)

        # Meta-learning
        solution_options = meta_controller.select_solution_strategy(task, deep_learning_output, rule_based_output, heuristic_output)

        # Generate solutions
        best_solution = None
        for solution in solution_options:
            if program_synthesis.validate_solution(solution, test_pairs):
                best_solution = solution
                break
        if best_solution is None:
            best_solution = neural_decoder.generate_solution(test_pairs[0][0])
        if best_solution is None:
            best_solution = transformation_library.generate_solution(test_pairs[0][0])

        # Visualize results
        GridVisualizer.visualize_sample(test_pairs[0][0], test_pairs[0][1], best_solution)

if __name__ == "__main__":
    main()