cs181-practical-4
=================

WARMUP:
To run the policy iteration for the warm-up, inside the "Warmup" directory type

python main.py

This will print the optimal policies and the values corresponding to those
policies. This will also show a plot of the values versus state. Gamma can be
changed in the main.py file.

SWINGYMONKEY:
To run the different reinforcement learning techniques, inside the "Monkey" directory type

python model_free.py (for model-free or "Q-learning" technique)

OR

python model_based.py (for model-based learning technique)

OR

python td_value.py (for temporal difference learning technique)

This will run the SwingyMonkey simulation with the given reinforcement learning techniques. 
The terminal window will print the current iteration, current score, highest score, and average score.
