import torch.nn as nn
import torch
import numpy as np
import random
import networkx as nx


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, max_encode=10000):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / max_encode ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_1d_sin_pos_embed_from_grid(embed_dim, pos, max_encode=10000):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim, dtype=np.float32)
    omega /= embed_dim
    omega = 1.0 / max_encode ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
    # print(out)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    # emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb_cos

def obs_query(obs, max_time_horizon, t):
    num_agent = obs.shape[0]
    t_one_hot = np.zeros((num_agent, max_time_horizon))
    t_one_hot[:, t] = 1
    return np.concatenate([t_one_hot, np.eye(num_agent)[np.arange(num_agent)]], axis=1)

def get_demonstration(obs, game, exit_node):
    num_agent = obs.shape[0]
    pths = [nx.shortest_path(game._graph.graph, source=obs[i][1], target=exit_node) for i in range(num_agent)]
    act_probs = [np.zeros(game.defender_mix_action) for _ in range(num_agent)]
    for i in range(num_agent):
        if len(pths[i]) == 1:   # stay
            act_probs[i][0] = 1.0
        else:
            curr_node, next_node = pths[i][0], pths[i][1]
            if curr_node - next_node > 1:   # down
                act_probs[i][1] = 1.0
            elif curr_node - next_node < -1:    # up
                act_probs[i][2] = 1.0
            elif curr_node - next_node == 1:    # left
                act_probs[i][3] = 1.0
            elif curr_node - next_node == -1:   # right
                act_probs[i][4] = 1.0
    return np.array(act_probs)