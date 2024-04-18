def v2x(obs, config):
    import random
    import numpy as np

    # Get valid moves for a grid
    def get_valid_moves(grid, config):
        return [c for c in range(config.columns) if grid[0][c] == 0]

    # Calculates score if agent drops piece in selected column
    def score_move(grid, col, mark, config):
        next_grid = drop_piece(grid, col, mark, config)
        score = get_heuristic(next_grid, mark, config)
        return score

    # Helper function for score_move: gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows - 1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid

    # Helper function for score_move: calculates value of heuristic for grid
    def calculate_heuristic(grid, mark, config):
        num_twos = count_windows(grid, 2, mark, config)
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_twos_opp = count_windows(grid, 2, mark % 2 + 1, config)
        num_threes_opp = count_windows(grid, 3, mark % 2 + 1, config)
        score = 1e15 * num_fours + 1e2 * num_threes + 1 * num_twos + -1 * num_twos_opp + -2e2 * num_threes_opp
        return score

    def calculate_opp_heuristic(grid, mark, config):
        num_twos = count_windows(grid, 2, mark, config)
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_threes_opp = count_windows(grid, 3, mark % 2 + 1, config)
        score = 1e9 * num_fours + 1e2 * num_threes + 1 * num_twos + -1e1 * num_threes_opp
        return score

    def get_heuristic(grid, mark, config):
        valid_moves = get_valid_moves(grid, config)
        if not valid_moves:
            opponent_score = 0
        else:
            opponent_scores = dict()
            for c in valid_moves:
                next_grid = drop_piece(grid, c, mark % 2 + 1, config)
                opponent_scores[c] = calculate_opp_heuristic(next_grid, mark % 2 + 1, config)
            opponent_score = max(opponent_scores.values())
        return calculate_heuristic(grid, mark, config) - opponent_score

    # Helper function for get_heuristic: checks if window satisfies heuristic conditions
    def check_window(window, num_discs, piece, config):
        return window.count(piece) == num_discs and window.count(0) == config.inarow - num_discs

    # Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
    def count_windows(grid, num_discs, piece, config):
        num_windows = 0
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[row, col:col + config.inarow])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # vertical
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns):
                window = list(grid[row:row + config.inarow, col])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # positive diagonal
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # negative diagonal
        for row in range(config.inarow - 1, config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        return num_windows

    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # Get list of valid moves
    valid_moves = get_valid_moves(grid, config)
    # Use the heuristic to assign a score to each possible board in the next turn
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config) for col in valid_moves]))
    for key in scores.keys():
        if key == config.columns//2:
            scores[key] += 1e1
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)

