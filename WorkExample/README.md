# Reward Learning with Intractable Normalizing Functions
This is a repository for our paper, ["Reward Learning with Intractable Normalizing Functions"](https://collab.me.vt.edu/pdfs/josh_ral23.pdf). We include:
- Implementation of a basic environment showcasing all normalizers mentioned in the paper for test and comparison of normalizer performance under different sampling conditions. 


## Working Environment 1
Hello, this is a simulation made to isolate the mathematical basis for the sampling methods to measure their time requirements outside of the use of internal factors like pybullet environment sampling used in the practical Panda_Env simulations. The environment rendered just represents a belief of where a marker should be in between two goals (1 and 2) where the user has a belief of how much to value the distance from each goal relative to the marker where the agent has to take an action to represent their belief.
## Requirements
Requirements are listed in requirements.txt:
- python3
- numpy $\ge$ 1.24.2

Requirements can be installed using pip:

    pip install -r requirements.txt
## Instructions
To run a demonstration of the normalizer runs, run `python main.py`. The initial settings define an initial 1000-run test summing the total error for each normalizer approximation of theta.

You can also provide arguments to adjust the run parameters of the code:

--runs: changes the number of runs that the errors are summed over. Default is 1000

--outer: changes the number of outer sample loops are used to sample for different beliefs for an ideal human action. Default is 1000

--inner: changes the number of inner sample loops are used to find the normalizers for each approach. Default is 50
