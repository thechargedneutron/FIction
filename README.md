# FIction: 4D Future Interaction Prediction - CVPR 2025

Code implementation of CVPR 2025 paper 'FIction: 4D Future Interaction Prediction from Video'.

[![arXiv](https://img.shields.io/badge/arXiv-2412.00932-00ff00.svg)](https://arxiv.org/pdf/2412.00932.pdf)  [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://vision.cs.utexas.edu/projects/FIction/)

![Teaser](teaser.png)

## Data preparation

Please follow [the instructions](preprocess/README.md) to create the dataset. We only use Ego-Exo4D videos and annotations to create the training and testing data.

## Training and inference

After preparing the code and setting the paths correctly, simply run

```
bash mover_trainer.sh <job_name>
```

This code copies the repo to a new locations and runs the code using SLURM. Making a copy is useful when your code does not run immediately.

For evaluation, set the path in `eval.py` and run

```
python eval.py
```

## Citation

If you use the code or the method, please cite the following paper:

```bibtek
@misc{ashutosh2024fiction,
      title={FIction: 4D Future Interaction Prediction from Video}, 
      author={Kumar Ashutosh and Georgios Pavlakos and Kristen Grauman},
      year={2024},
      eprint={2412.00932},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.00932}, 
}
```
