# Differentially-Private-Online-Transfer-Learning-for-Reinforcement-Learning-in-Partially-Observable-E
Demo code for the paper "Differentially Private Online Transfer Learning for Reinforcement Learning in Partially Observable Environment"

### Required Dependencies:
- NVIDIA GPU
- Python 3.6, numpy 1.15.2, pyglet 1.3.2

### Notes:
#### 1. Introduction:
#### There are 3 experiments included in the demo code.
- Ex1 is the environment that robots collecting the rubbish at static positions.
- Ex2 is the environment that has a probability of generating new rubbish at each step.
- Ex3 is the environment that robots rescuing victims who have a probability of moving to a new position at each step.
#### In each experiment file directory, there are three enviroments.
- env1.py is used to test the performance of Reinforcement Learning Method.
- env2.py is used to test the performance of Differetially Private Online Transfer Learning Method.
- env3.py is used to test the performance of Differetially Private Online Transfer Learning Method with Malicious Agent.
#### To test the performance of Random Method in all the three experiments, you need to open env1.py in the three experiment folders and then toggle comment for the following line:
- "file = open('Ex#Random.txt','w')"
- "filename = "ExWithoutTL(" + currentDT.strftime("%H-%M-%S %Y-%m-%d") + ").txt""
- "file = open(filename,'w')"
- "env.step(env.sample_action())"
- "env.step(env.algorithm_one())"
#### 2. Adjustable variables:
- Window size: WINDOW_WIDTH, WINDOW_HEIGHT
- Number of objects: RUBBISH_NUM (VICTIM_NUM in Ex3), BLOCK_NUM, BOT_NUM
- Barrier Positions: BLOCK_POSITION
