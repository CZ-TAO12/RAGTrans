from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax

class Hyperedge(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'add')
        self.heads = 1
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None) -> Tensor:

        self.out_channels = x.size(-1)

        num_nodes, num_edges = x.size(0), 0
        alpha = None

        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))

        out = out.view(-1, self.heads * self.out_channels)

        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:

        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out

class HypergraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads= 1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels//heads

        self.hyperedge_func = Hyperedge()

        # attention
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = Linear(in_channels, heads * self.out_channels, bias=False,
                            weight_initializer='glorot')
        self.lin2 = Linear(in_channels, heads * self.out_channels, bias=False,
                            weight_initializer='glorot')
        
        self.att = Parameter(torch.Tensor(1, heads,self.out_channels))
        self.att2 = Parameter(torch.Tensor(1, heads, self.out_channels))

        # FFN
        self.FFN_1 = Linear(out_channels, out_channels, bias=False, weight_initializer='glorot')
        self.FFN_2 = Linear(out_channels, out_channels, bias=False, weight_initializer='glorot')
        self.FFN_3 = Linear(out_channels, out_channels, bias=False, weight_initializer='glorot')
        self.FFN_4 = Linear(out_channels, out_channels, bias=False, weight_initializer='glorot')
        self.layer_norm = nn.LayerNorm(out_channels)
        

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * self.out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin2.reset_parameters()
        self.FFN_1.reset_parameters()
        self.FFN_2.reset_parameters()
        self.FFN_3.reset_parameters()
        self.FFN_4.reset_parameters()
        
        glorot(self.att)
        glorot(self.att2)
        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor, x2: Tensor =None,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:

        num_nodes, num_edges = x.size(0), 0

        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.layer_norm(x)
        residual  = x
        x = self.lin(x)
        
        hyperedge_attr = self.hyperedge_func(x, hyperedge_index)
        hyperedge_attr = self.layer_norm(hyperedge_attr)
        # hyperedge_attr = self.lin2(hyperedge_attr)

        alpha = None

        if x2 is not None:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            x2 = x2.view(-1, self.heads, self.out_channels)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads, self.out_channels)

            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            x2_i = x2[hyperedge_index[0]]


            alpha = x_i.mul(x_j)
            alpha2 = x2_i.mul(x_j)
            alpha_mix = torch.cat([alpha, alpha2], dim = 0)

            alpha_mix = (alpha_mix * self.att).sum(dim=-1)
            alpha_mix = F.leaky_relu(alpha_mix, self.negative_slope)
            h_edge_index = torch.cat([hyperedge_index[0],hyperedge_index[0]], 0)
            alpha_mix = softmax(alpha_mix, h_edge_index, num_nodes=x.size(0))
            alpha_mix = F.dropout(alpha_mix, p=self.dropout, training=self.training)
            alpha_mix, alpha_mix2 = torch.chunk(alpha_mix, 2, dim=0)

        else:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            # hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                     self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

       
        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        if x2 is not None:

            out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha_mix, size=(num_nodes, num_edges))
            out2 = self.propagate(hyperedge_index, x=x2, norm=B, alpha=alpha_mix2, size=(num_nodes, num_edges))

            x = x.view(-1, self.heads, self.out_channels)

            hyperedge_attr = self.layer_norm(out.view(-1, self.heads*self.out_channels))
            hyperedge_attr2 = self.layer_norm(out2.view(-1, self.heads*self.out_channels))

            hyperedge_attr = hyperedge_attr.view(-1, self.heads, self.out_channels)
            hyperedge_attr2 = hyperedge_attr2.view(-1, self.heads, self.out_channels)

            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            x2_j = hyperedge_attr2[hyperedge_index[1]]

            alpha = x_i.mul(x_j)
            alpha2 = x_i.mul(x2_j)
            alpha_mix = torch.cat([alpha, alpha2], dim = 0)

            alpha_mix = (alpha_mix * self.att2).sum(dim=-1)
            alpha_mix = F.leaky_relu(alpha_mix, self.negative_slope)
            h_edge_index = torch.cat([hyperedge_index[0],hyperedge_index[0]], 0)
            alpha_mix = softmax(alpha_mix, h_edge_index, num_nodes=x.size(0))
            alpha_mix = F.dropout(alpha_mix, p=self.dropout, training=self.training)
            alpha_mix, alpha_mix2 = torch.chunk(alpha_mix, 2, dim=0)


            out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D, alpha=alpha_mix, size=(num_edges, num_nodes))
            out2 = self.propagate(hyperedge_index.flip([0]), x=out2, norm=D, alpha=alpha_mix2, size=(num_edges, num_nodes))

        else:
            out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha, size=(num_nodes, num_edges))
            out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D, alpha=alpha, size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
            out2 = out2.view(-1, self.heads * self.out_channels)

        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        out = self.FFN_2(F.relu(self.FFN_1(out) + self.FFN_3(out2)))
        out = self.layer_norm(residual + out)
        
        return out



    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out


