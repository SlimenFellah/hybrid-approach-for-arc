from dataloader.arc_data_loader import ARCDataLoader
from symbolic_rules.rule_library import repeat_tile_flip_pattern


class ARCSolver:
    def __init__(self, data_loader: ARCDataLoader):
        self.data_loader = data_loader

    def solve_task_00576224(self):
        task = self.data_loader.get_task("00576224")
        predictions = []

        for test_input in task.get_test_inputs():
            predicted = repeat_tile_flip_pattern(test_input)
            predictions.append(predicted)

        return predictions
