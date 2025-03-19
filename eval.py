from transformers import AutoModel

from main import CustomTransformerModel, CustomConfig, ComputeMetricsWrapper, torch_default_data_collator
from loader import InteractionDataset
from transformers import (
    Trainer,
    TrainingArguments,
)

checkpoint_path = '/path/to/run/results/run_name_lr_5e-6_ft_180_obst_30/checkpoint-6120'

model = CustomTransformerModel.from_pretrained(checkpoint_path)
config = CustomConfig.from_pretrained(checkpoint_path)

save_vertices = False # Setting it incorrectly will not give wrong result but possibly slow down the inference
if save_vertices:
    # Set set_return_smpl_vertices for pose analysis
    model.set_return_smpl_vertices()

eval_dataset = InteractionDataset('test', observation_time=config.observation_time, anticipation_time=config.anticipation_time, future_time=config.future_time, scenarios=config.scenarios)
compute_metrics_obj = ComputeMetricsWrapper(eval_dataset, config.training_task, save_vertices=save_vertices)

# Define the training arguments (used for evaluation)
training_args = TrainingArguments(
    output_dir="./results/eval",
    per_device_eval_batch_size=2,
    do_train=False,
    do_eval=True,
    logging_steps=10,
    evaluation_strategy="no",
    report_to="none",  # Avoid unnecessary logging
    include_inputs_for_metrics=True,
)

# Instantiate the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics_obj.compute_metrics,
    data_collator=torch_default_data_collator,
)

# Evaluate the model
eval_result = trainer.evaluate()
if save_vertices:
    import pickle
    with open('viz/test_vertices.pkl', 'wb') as f:
        pickle.dump(eval_result, f)

print("*"*100)
print(eval_result)