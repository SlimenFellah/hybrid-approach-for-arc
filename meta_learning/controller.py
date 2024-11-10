import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from src.config import Config
from src.data_handling.data_loader import ARCDataLoader
from src.feature_extraction.neural_features import NeuralFeatureExtractor
from src.feature_extraction.symbolic_features import SymbolicFeatureExtractor
from src.feature_extraction.statistical_features import StatisticalFeatureExtractor
from src.pattern_recognition.deep_learning import DeepLearningModel
from src.pattern_recognition.rule_based import RuleBasedSystem
from src.pattern_recognition.heuristics import HeuristicEngine
from src.solution_generation.program_synthesis import ProgramSynthesizer
from src.solution_generation.neural_decoder import NeuralDecoder
from src.visualization.grid_viz import GridVisualizer

class ARCController:
    def __init__(self):
        # Initialize components
        self.neural_extractor = NeuralFeatureExtractor()
        self.symbolic_extractor = SymbolicFeatureExtractor()
        self.statistical_extractor = StatisticalFeatureExtractor()
        
        self.deep_learning = DeepLearningModel()
        self.rule_system = RuleBasedSystem()
        self.heuristic_engine = HeuristicEngine()
        
        self.program_synthesizer = ProgramSynthesizer()
        self.neural_decoder = NeuralDecoder()
        
        self.visualizer = GridVisualizer()
        
        # Initialize knowledge base
        self.knowledge_base = {
            'successful_patterns': [],
            'transformation_history': [],
            'meta_strategies': {}
        }
    
    def analyze_task(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Analyze training pairs to determine task characteristics"""
        analysis = {
            'neural_features': [],
            'symbolic_features': [],
            'statistical_features': [],
            'identified_patterns': [],
            'potential_transformations': []
        }
        
        for input_grid, output_grid in train_pairs:
            # Extract features
            neural_feats = self.neural_extractor.extract_features(input_grid)
            symbolic_feats = self.symbolic_extractor.extract_features(input_grid, output_grid)
            statistical_feats = self.statistical_extractor.extract_features(input_grid)
            
            analysis['neural_features'].append(neural_feats)
            analysis['symbolic_features'].append(symbolic_feats)
            analysis['statistical_features'].append(statistical_feats)
            
            # Identify patterns
            deep_patterns = self.deep_learning.identify_patterns(input_grid, output_grid)
            rule_patterns = self.rule_system.identify_patterns(input_grid, output_grid)
            heuristic_patterns = self.heuristic_engine.identify_patterns(input_grid, output_grid)
            
            analysis['identified_patterns'].extend([
                deep_patterns, rule_patterns, heuristic_patterns
            ])
            
            # Generate potential transformations
            transformations = self.generate_transformations(input_grid, output_grid)
            analysis['potential_transformations'].extend(transformations)
        
        return analysis
    
    def generate_transformations(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Dict]:
        """Generate potential transformations between input and output"""
        transformations = []
        
        # Try program synthesis
        program = self.program_synthesizer.synthesize(input_grid, output_grid)
        if program:
            transformations.append({
                'type': 'program',
                'transformation': program,
                'confidence': self.program_synthesizer.evaluate_confidence(program)
            })
        
        # Try neural decoder
        neural_transform = self.neural_decoder.decode_transformation(input_grid, output_grid)
        if neural_transform:
            transformations.append({
                'type': 'neural',
                'transformation': neural_transform,
                'confidence': self.neural_decoder.evaluate_confidence(neural_transform)
            })
        
        return transformations
    
    def solve_task(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]], 
                  test_input: np.ndarray) -> np.ndarray:
        """Solve a task given training pairs and test input"""
        # Analyze task
        analysis = self.analyze_task(train_pairs)
        
        # Select best strategy based on analysis
        strategy = self.select_strategy(analysis)
        
        # Generate solution
        if strategy['type'] == 'program':
            solution = self.program_synthesizer.apply_program(
                strategy['transformation'], test_input)
        elif strategy['type'] == 'neural':
            solution = self.neural_decoder.apply_transformation(
                strategy['transformation'], test_input)
        else:
            # Fallback to rule-based approach
            solution = self.rule_system.apply_rules(test_input, analysis['identified_patterns'])
        
        # Validate solution
        if self.validate_solution(solution):
            return solution
        else:
            # Try alternative approach if validation fails
            return self.generate_fallback_solution(test_input, analysis)
    
    def select_strategy(self, analysis: Dict) -> Dict:
        """Select best solution strategy based on task analysis"""
        # Calculate confidence scores for different approaches
        program_confidence = max([t['confidence'] for t in analysis['potential_transformations'] 
                                if t['type'] == 'program'], default=0)
        neural_confidence = max([t['confidence'] for t in analysis['potential_transformations']
                               if t['type'] == 'neural'], default=0)
        
        # Select strategy with highest confidence
        if program_confidence > neural_confidence:
            return next(t for t in analysis['potential_transformations'] 
                       if t['type'] == 'program' and t['confidence'] == program_confidence)
        else:
            return next(t for t in analysis['potential_transformations']
                       if t['type'] == 'neural' and t['confidence'] == neural_confidence)
    
    def validate_solution(self, solution: np.ndarray) -> bool:
        """Validate generated solution"""
        # Check basic constraints
        if solution is None or solution.size == 0:
            return False
        
        if solution.shape[0] > Config.MAX_GRID_SIZE or solution.shape[1] > Config.MAX_GRID_SIZE:
            return False
        
        # Check value constraints
        if np.any(solution < 0) or np.any(solution > 9):
            return False
        
        return True
    
    def generate_fallback_solution(self, test_input: np.ndarray, 
                                analysis: Dict) -> np.ndarray:
        """Generate fallback solution when primary approach fails"""
        # Try rule-based approach
        rule_solution = self.rule_system.apply_rules(test_input, analysis['identified_patterns'])
        if self.validate_solution(rule_solution):
            return rule_solution
        
        # Try heuristic approach
        heuristic_solution = self.heuristic_engine.generate_solution(test_input)
        if self.validate_solution(heuristic_solution):
            return heuristic_solution
        
        # Last resort: return modified input
        return test_input.copy()
