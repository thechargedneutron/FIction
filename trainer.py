from transformers import Trainer
from typing import Optional
import pickle
import os


class CustomTrainer(Trainer):
    def evaluation_loop(self, *args, **kwargs):
        # Do the regular evaluation
        output = super().evaluation_loop(*args, **kwargs)

        # Define custom variables that will be used in the save function
        self.custom_current_indexes = output.metrics['eval_indexes']
        self.custom_current_predictions = output.predictions
        self.custom_current_labels = output.label_ids
        del output.metrics['eval_indexes'] # We cannot have np array in metrics

        return output

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # Do the regular save operation
        super()._save(output_dir, state_dict)

        # Also save the custom variables
        with open(os.path.join(output_dir, 'indexes.pkl'), 'wb') as f:
            pickle.dump(self.custom_current_indexes, f)
        with open(os.path.join(output_dir, 'predictions.pkl'), 'wb') as f:
            pickle.dump(self.custom_current_predictions, f)
        with open(os.path.join(output_dir, 'labels.pkl'), 'wb') as f:
            pickle.dump(self.custom_current_labels, f)