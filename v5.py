from agentq import AgentAI
import numpy as np


def v5(obs, config):
    grid = np.asarray(obs.board).reshape(1, config.rows, config.columns)
    actions = [c for c in range(config.columns) if grid[0][0][c] == 0]
    agent = AgentAI(actions)
    action = agent.choose_action(grid)
    return action
