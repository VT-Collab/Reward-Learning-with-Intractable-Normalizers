# Reward Learning with Intractable Normalizing Functions
This is a repository for our paper, ["Reward Learning with Intractable Normalizing Functions"](https://collab.me.vt.edu/pdfs/josh_ral23.pdf). We include:
- Implementation of a Panda Robot environment showcasing all normalizers mentioned in the paper for test and comparison of normalizer performance under different sampling conditions 
  in a comparable setting to the User Study Setting.
  
## Panda Sim Environment
This is a simulation made to isolate the mathematical basis for the sampling methods to measure their time requirements outside of the use of internal factors like pybullet environment sampling used in the practical Panda_Env simulations. Within this is the 3 different simulations

## Requirements
Requirements are listed in requirements.txt:
- python3
- numpy $\ge$ 1.24.2
- pybullet $\ge$ 3.5.2
- scipy $\ge$ 1.8.0

Requirements can be installed using pip:

    pip install -r requirements.txt

## Instructions - Panda_Sims
To run a demonstration of a set of the normalizer runs for each individual simulation, run `python main.py`. The initial settings define an initial 100-run test summing the total error for each normalizer approximation of theta. If you want a graphical representation of the error and regret metrics afterward, run 'python plotter.py'.

You can also provide arguments to adjust the run parameters of the code:

--runs: changes the number of runs that the errors are summed over. Default is 100

--outer: changes the number of outer sample loops are used to sample for different beliefs for an ideal human action. Default is 50

--inner: changes the number of inner sample loops are used to find the normalizers for each approach. Default is 25

## Basic Results
Here are an example set of results for running each of the simulations with default parameters. As you all can see, for the push simulation, the error is all fairly low given the simulation's low dimensional requirements allowing all to achieve fairly applicable results. However, as you expand the feature dimensions and action space as you would for the close and pour simulations, the results for the Ignore method quickly become unreliable despite increased sampling loops where the other methods continue to scale. 

![ErrorSim1](https://github.com/VT-Collab/Reward-Learning-with-Intractable-Normalizers/assets/112197239/3dfa46e8-0e6a-4b90-ac14-39adbd1d4bf4)
![RegretSim1](https://github.com/VT-Collab/Reward-Learning-with-Intractable-Normalizers/assets/112197239/70e14886-dd9a-4f74-a0d3-779beb47415b)

![ErrorSim2](https://github.com/VT-Collab/Reward-Learning-with-Intractable-Normalizers/assets/112197239/00249676-bdaf-4c6f-a209-840a11f1c5d7)
![RegretSim2](https://github.com/VT-Collab/Reward-Learning-with-Intractable-Normalizers/assets/112197239/e1314f0b-719a-4ea9-aa23-8d3caaa3a835)

![ErrorSim3](https://github.com/VT-Collab/Reward-Learning-with-Intractable-Normalizers/assets/112197239/7344be44-7c0c-4fdc-8619-123c97cfd8e6)
![RegretSim3](https://github.com/VT-Collab/Reward-Learning-with-Intractable-Normalizers/assets/112197239/d151e1b0-458c-454a-90b3-2f738d894473)

