from kaggle_environments import evaluate, make
from v2x import *
import numpy as np
from v5 import *


def main():
    agent1 = v2x       # k
    agent2 = v5  # kaczka
    env = make("connectx", debug=True)
    env.run([agent1, agent2])
    html = env.render(mode="html")
    print(env.state[-1].reward)
    with open("output.html", "w") as file:
        file.write(html)
    get_win_percentages(agent1, agent2,100)


def get_win_percentages(agent1, agent2, n_rounds=100):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)
    # Agent 2 goes first (roughly) half the time
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]
    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))
    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))
    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))
    print("Number of Tied Games:", outcomes.count([0, 0]))
    print("Number of Agent 1 Win:", outcomes.count([1,-1]))
    print("Number of Agent 2 Win:", outcomes.count([-1,1]))


if __name__ == '__main__':
    main()
