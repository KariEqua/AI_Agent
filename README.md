# AI_Agent
Agents for game Connect Four

Connect X is one of the Kaggle Competitions


Rules of the game:

We have a vertical grid board with 6 rows and 7 columns.
1. Players take turns dropping one of their colored discs into any chosen column from the top.
2. The disc will fall to the lowest available space within the column.
3. Players aim to create a line of four of their own colored discs horizontally, vertically, or diagonally.
4. If the grid fills up without any player achieving four in a row, the game ends in a draw.


Included files:

ConnectFourGym - defines a gym environment for the game, inherits from the gym.Env class

runner - class used to train and evaluate 

agentq - defines class AgentAI, which represents an AI agent trained using the Q-learning algorithm

v5 - defines function which represents strategy using agentq

v2x - defines function which represents strategy using heuristics

left - defines function which represents strategy choosing leftmost column

main - plays game for two choosen agents and creates statistics

q_values - pickle file for q values generated during training

output - html file which shows a game (training)


imported libraries:
random

numpy

pandas

pickle

gym

gym -> spaces

kaggle_environment -> evaluate, make
