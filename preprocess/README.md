# Dataset preparation

## Extract WHAM poses

Use the attached WHAM codebase to extract the SMPL poses for people in the Ego-Exo4D dataset. We also provide a SLURM script to parallelize the process. Finally, use the `find_actor_from_outputs_largest_area.py` script to find the actor for each video. This code produces `{take_name}_largest_area.pkl` output files that are used later in the dataset creation pipeline.

## Extract object labels and 3D positions

Use the Detic codebase to extract object labels from the LVIS vocabulary. The output is `{frame_idx:06}.pkl` for every frame in a video. Use the `map_3d_to_detic_objects.py` script to find the mapping between 3D locations and 2D frame pixels, and thus, 3D to Detic objects. Finally, run `create_spatial_cubes_from_mappings.py` that produces `{take_name}_object_bbs_obb.pkl` for every take.

## Obtaining interaction instances from narrations

We use Llama3 and the provided narrations to find the interaction instances. In the InteractionFromNarration folder, run `llama3.py` to use Llama to find the interaction instances, followed by `parse_outputs.py` to obtain the `compiled_interactions` output. The output is also provided in case you cannot run it yourself.

## Putting it all together

Finally, run `PuttingEverythingIn3D/main.py` after setting the output paths correctly from all the previous steps. This results in the final dataset that we use in the training and testing.
