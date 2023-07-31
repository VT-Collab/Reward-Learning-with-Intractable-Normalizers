# Reward Learning with Intractable Normalizing Functions
This is a repository for our paper, ["Reward Learning with Intractable Normalizing Functions"](https://collab.me.vt.edu/pdfs/josh_ral23.pdf). We include:
- Implementation of a basic environment showcasing all normalizers mentioned in the paper for test and comparison of normalizer performance under different sampling conditions.
- Implementation of a Panda Robot environment showcasing all normalizers mentioned in the paper for test and comparison of normalizer performance under different sampling conditions 
  in a comparable setting to User Study Setting.
  

## Requirements
Requirements are listed in requirements.txt:
- python3
- numpy $\ge$ 1.24.2
- pybullet $\ge$ 3.5.2
- scipy $\ge$ 1.8.0

Requirements can be installed using pip:

    pip install -r requirements.txt
## Instructions - Toy Sim
To run a demonstration of the normalizer runs, run `python main.py`. The initial settings define an initial 1000 run test summing the total error for each normalizer approximation of theta.

You can also provide arguments to adjust the run parameters of the code:

--runs: changes the number of runs that the errors are summed over. Default is 1000

--outer: changes the number of outer sample loops are used to sample for different beliefs for an ideal human action. Default is 1000

--inner: changes the number of inner sample loops are used to find the normalizers for each approach. Default is 50

## Instructions - Panda_Sims
To run a demonstration of the normalizer runs, run `python main.py`. The initial settings define an initial 100 run test summing the total error for each normalizer approximation of theta. If you want a graphical representation of the error and regret metrics afterward, run 'python plotter.py'.

You can also provide arguments to adjust the run parameters of the code:

--runs: changes the number of runs that the errors are summed over. Default is 100

--outer: changes the number of outer sample loops are used to sample for different beliefs for an ideal human action. Default is 50

--inner: changes the number of inner sample loops are used to find the normalizers for each approach. Default is 10
