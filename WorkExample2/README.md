# Reward Learning with Intractable Normalizing Functions
This is a repository for our paper, ["Reward Learning with Intractable Normalizing Functions"](https://collab.me.vt.edu/pdfs/josh_ral23.pdf). We include:
- Implementation of a basic environment showcasing all normalizers mentioned in the paper for test and comparison of normalizer performance under different sampling conditions.

## Working Example 2 - 2D Goal Navigation
Hello, this is a simulation made to render a simplified state-action version of Inference Sampling methods as opposed to the trajectory-based methods used within the paper's simulations. The environment rendered is a simple 2D xy space with two goals the noisily rational human model seeks to navigate between, where the agent is given a random starting space in the environment and given limited action space to move within for future state estimation.
## Requirements
Requirements are listed in requirements.txt:
- python3
- numpy $\ge$ 1.24.2

Requirements can be installed using pip:

    pip install -r requirements.txt
## Instructions
To run a demonstration of the normalizer runs, run `python main.py` or run 'python main_alt.py' for a Q-Value function variation of the simulation.

The initial settings define an initial 1000-run test summing the total error for each normalizer approximation of theta.

You can also provide arguments to adjust the run parameters of the code:

--runs: changes the number of runs that the errors are summed over. Default is 1000

--outer: changes the number of outer sample loops used to sample for different beliefs for an ideal human action. Default is 1000

--inner: changes the number of inner sample loops are used to find the normalizers for each approach. Default is 50
