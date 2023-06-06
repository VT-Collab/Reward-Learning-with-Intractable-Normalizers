# Reward Learning with Intractable Normalizing Functions
This is a repository for our paper, ["Reward Learning with Intractable Normalizing Functions"](https://collab.me.vt.edu/pdfs/josh_ral23.pdf). We include:
- Implementation of basic enviroment implementation of all normalizers mentioned in the paper for test and comparison of normalizer performance under different sampling conditions.

## Requirements
Requirements are listed in requirements.txt:
- python3
- numpy $\ge$ 1.24.2
Requirements can be installed using pip:
    pip install -r requirements.txt
    
## Instructions
To run a demonstration of the normalizer runs, run `python main.py`. The initial settings define an initial 1000 run test summing the total error for each normalizer approximation of theta.
