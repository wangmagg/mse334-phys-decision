# Physician-Mediated Interference in Experimental Evaluations of Clinical Decision Support Tools
Simulations of experimental evaluations of support tools where learning and habituation effects drive physician-mediated interference.

## Setup
__Installation__ <br />
To clone this repository, run the following <br />
```
git clone https://github.com/wangmagg/phys-deicsion.git
cd phys-decision
```

__Python dependencies__ <br />
Python dependencies are specified in `env/requirements.txt`. To set up and activate a Conda environment with the requisite dependicies, run the following <br />
```
conda env create -f env/environment.yml
conda activate phys-decision-env
```

## File Structure
    .
    ├── scripts                 # Bash scripts for running simulations
    ├── src                     
      ├── decisions             # Functions for computing decision probabilities and making decisions.
      ├── estimands             # Functions for calculating ground truth effect estimands and estimating effects
      ├── main                  # Experiment simulations
    ├── visualization.ipynb     # Notebook for producing figures in write-up
    └── README.md

## Usage
### Running Simulations
For convenience, we provide bash scripts for running the simulated experiments. Each script corresponds to one behavioral adaptation type (static, linear, exponential, step). To execute these scripts, run the following:
```
bash scripts/static.sh
bash scripts/linear.sh
bash scripts/run_exponential.sh
bash scripts/run_step.sh
```

Running these scripts creates a ```res``` output directory to store results from the simulations. The subdirectories correspond to different settings of physician and tool decision-making qualities, and the subsubdirectories correspond to different adaptation types.

### Visualizations
Code for generating the plots in the write-up can be found in the ```visualization.ipynb``` notebook.


