import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.utils import expand_as_pair
from utils.graph_learning_utils import create_activation, create_norm

class GCN(nn.Module):
    def __init__(self, in_dim, num_hidden, out_dim, num_layers, dropout, activation='prelu', norm='layernorm', encoding=False):
        super(GCN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.dropout = dropout

        last_activation = create_activation(activation) if encoding else None
        last_norm = create_norm(norm) if encoding else None

        if num_layers == 1:
            self.gcn_layers.append(GraphConv(in_dim, out_dim, norm=last_norm, activation=last_activation))
        else:
            # input projection (no residual)
            self.gcn_layers.append(GraphConv(in_dim, num_hidden, norm=create_norm(norm), activation=create_activation(activation)))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gcn_layers.append(GraphConv(num_hidden, num_hidden, norm=create_norm(norm), activation=create_activation(activation)))
            # output projection
            self.gcn_layers.append(GraphConv(num_hidden, out_dim, norm=last_norm, activation=last_activation))


        self.norms = None
        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcn_layers[l](g, h)
            if self.norms is not None and l != self.num_layers - 1:
                h = self.norms[l](h)
            hidden_list.append(h)
        # output projection
        if self.norms is not None and len(self.norms) == self.num_layers:
            h = self.norms[-1](h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


def message_func(edges):
    return {'m': edges.src['h']}

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, norm=None, activation=None):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim
        self.fc = nn.Linear(in_dim, out_dim)
        self.register_buffer('res_fc', None)
        self.norm = norm
        if norm is not None:
            self.norm = norm(out_dim)
        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, graph, feat):
        with graph.local_scope():
            # aggregate_fn = dgl.function.copy_src('h', 'm')
            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            degs = graph.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat_src = feat_src * norm
            graph.srcdata['h'] = feat_src
            graph.update_all(message_func, fn.sum(msg='m', out='h'))
            # graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            rst = self.fc(rst)
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)
            if self.norm is not None:
                rst = self.norm(rst)
            if self._activation is not None:
                rst = self._activation(rst)
            return rst