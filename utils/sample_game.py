import numpy as np
import copy

from utils.game_config import get_game

def sample_game(args, default_game=None, min_evader_pth_len=0):
    if default_game is None:
        while True:
            game = get_game(args)
            # check whether the game is valid, e.g., there is at least one path that the attacker can evade
            path_length = np.array([len(i[0]) if len(i) > 0 else 0 for i in game.attacker_path.values()])
            if sum(path_length > 0) > 0 and min(path_length[path_length > 0]) >= min_evader_pth_len:
                break
        new_game = True
    else:
        game = get_game(args)
        # check whether the game is valid, e.g., there is at least one path that the attacker can evade
        path_length = np.array([len(i[0]) if len(i) > 0 else 0 for i in game.attacker_path.values()])
        if sum(path_length > 0) > 0 and min(path_length[path_length > 0]) >= min_evader_pth_len:
            new_game = True
        else:
            new_game = False
            game = copy.deepcopy(default_game)  # use previous valid game if current generated game is invalid
    return game, new_game