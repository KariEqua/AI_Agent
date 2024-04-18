# Selects middle column
def agent_middle(obs, config):
    return config.columns//2


# Selects leftmost valid column
def agent_leftmost(obs, config):
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return valid_moves[0]