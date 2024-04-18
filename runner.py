from ConnectFourGym import ConnectFourGym
from agentq import AgentAI
from gym import spaces
from v2x import v2x


class Runner:
    def __init__(self, opponent):
        self.env = ConnectFourGym(opponent)
        actions = [c for c in range(self.env.columns)]
        self.agent = AgentAI(actions)

    def train(self, n: int):
        """
        Train an AI by playing `n` games against 'opponent'.
        """
        for i in range(n):
            print('game: ', i)
            obs = self.env.reset()
            move_list = []
            done = False
            while not done:
                self.agent.actions = [c for c in range(self.env.columns) if obs[0][0][c] == 0]
                action = self.agent.choose_action(obs)
                old_obs = obs
                obs, reward, done, truncated = self.env.step(action)
                move_list.append((old_obs, action, reward))
            reward_factor = 0
            move_number = len(move_list)
            for _ in range(1, move_number + 1, 1):
                old_obs, action, reward = move_list.pop()
                reward += reward_factor
                self.agent.update(old_obs, action, obs, reward)
                reward_factor = reward * len(move_list) / move_number
                obs = old_obs
        self.agent.save_q_values()

        # print(self.env.ks_env.render(mode="ansi"))
        html = self.env.ks_env.render(mode="html")
        with open("game.html", "w") as file:
            file.write(html)
        return self

    def play(self, n: int):
        """
        Play 'opponent' game against the AI agent.
        Prints statistics for agents.
        """
        agent1_wins = 0
        agent2_wins = 0
        agent1_invalid_move = 0
        tie = 0
        for i in range(n):
            obs = self.env.reset()
            done = False
            while not done:
                self.agent.actions = [c for c in range(self.env.columns) if obs[0][0][c] == 0]
                action = self.agent.choose_action(obs)
                obs, reward, done, truncated = self.env.step(action)
            if reward == 1:
                agent1_wins += 1
            elif reward == -1:
                agent2_wins += 1
            elif reward == -10:
                agent1_invalid_move += 1
            else:
                tie += 1
        print('Agent1 won: ', agent1_wins, ' ', 100 * agent1_wins/n, '%')
        print('Agent1 invalid moves: ', agent1_invalid_move,  ' ', 100 * agent1_invalid_move/n, '%')
        print('Agent2 won :', agent2_wins, ' ', 100 * agent2_wins/n, '%')
        print('Tie: ', tie, ' ', 100 * tie/n, '%')

        return self


if __name__ == '__main__':
    # Runner('random').train(2000)
    # Runner(v2x).train(1000)
    # Runner('random').train(10000)
    # Runner(v2x).train(1000)
    # Runner('random').train(100)
    Runner('random').play(100)
