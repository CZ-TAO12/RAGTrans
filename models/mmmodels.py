import math
import pickle

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from models.HGAT import HypergraphConv
from models.TransformerBlock import *

'''To GPU'''
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

'''To CPU'''
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

''''Load embedding from pickle file'''
def load_embedding(format, data_name):
    if format == 'text':
        path = 'data/'+data_name+'/text_embedding_list.pickle'
        text_embedding_list = pickle.load(open(path, 'rb'))
        return text_embedding_list
    elif format == 'img':
        path = 'data/'+data_name+'/img_embedding_list.pickle'
        img_embedding_list = pickle.load(open(path, 'rb'))
        return img_embedding_list


class fusion(nn.Module):
    def __init__(self, input_size, dropout=0.2):
        super(fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 300):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)
        self.dropout = dropout

        self.sec_pos_label = torch.arange(0, max_len).unsqueeze(0)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len + 1, 1, d_model)
        pe[1:, 0, 0::2] = torch.sin(position * div_term)
        pe[1:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        bz, lens, _ = x.size()
        sec_pos_label = self.sec_pos_label.repeat(bz, 1)

        x += self.pe[sec_pos_label.long(), 0, :] * 1e-3
        x += self.pe[sec_pos_label.long(), 0, :] * 1e-3
        return  F.dropout( x, p=self.dropout, training=self.training)  

class MMHG(nn.Module):
    def __init__(self, args, dropout=0.2):
        super(MMHG, self).__init__()

        # parameters
        self.emb_size = args.embSize
        self.pos_dim = args.posSize
        #self.n_node = args.n_node
        self.layers = args.layer
        self.dropout = nn.Dropout(dropout)
        self.data_name = args.data_name
        self.att_head = 2
        self.batch_size = args.batch_size

        self.layer = args.layer

        self.hgConv_list1 = [HypergraphConv(self.emb_size, self.emb_size, heads=4).cuda() for _ in range(args.layer)]
        self.hgConv_list2 = [HypergraphConv(self.emb_size, self.emb_size, heads=4).cuda() for _ in range(args.layer)]

        self.pos_embedding = PositionalEncoding(self.emb_size)

     
        self.fusing_attn = TransformerBlock(input_size=self.emb_size + self.emb_size // 2, n_heads=4,
                                               attn_dropout=dropout)
     
        if self.data_name == 'WEIXIN2':
            #self.node_embedding = nn.Embedding(64704, self.emb_size)
            self.linear1 = nn.Linear(256, self.emb_size)
            self.linear2 = nn.Linear(256, self.emb_size)
            self.user_embedding = nn.Embedding(20000, self.emb_size // 2)
        elif self.data_name == 'SMP':
            #self.node_embedding = nn.Embedding(305613, self.emb_size)
            self.linear1 = nn.Linear(384, self.emb_size)
            self.linear2 = nn.Linear(2048, self.emb_size)
            self.user_embedding = nn.Embedding(38312, self.emb_size //2)
        elif self.data_name == 'SIPD2020CHALLENGE':
            self.linear1 = nn.Linear(384, self.emb_size)
            self.linear2 = nn.Linear(2048, self.emb_size)
            self.user_embedding = nn.Embedding(25000, self.emb_size // 2)

        self.linear3 = nn.Linear(self.emb_size*3, self.emb_size+self.emb_size//2)
        self.linear4 = nn.Linear(self.emb_size+self.emb_size//2, 1)
        self.linear5 = nn.Linear(self.emb_size*2, self.emb_size)

    
        self.dense = torch.nn.Sequential(
            nn.Linear(self.emb_size+self.emb_size//2, self.emb_size+self.emb_size//2),
            nn.ReLU(),
        )

        self.reset_parameters()

        self.text_embedding = torch.from_numpy(np.array(load_embedding('text', self.data_name))).cuda().to(torch.float32)
        self.img_embedding = torch.from_numpy(np.array(load_embedding('img', self.data_name))).cuda().to(torch.float32)

        #### optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)
        self.loss_function = nn.MSELoss()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hg_idx, related_items, label, uid):


        user_embedding = self.user_embedding(uid)
        text_embedding = self.linear1(self.text_embedding[related_items])
        text_embedding = torch.reshape(text_embedding, (-1, 300, self.emb_size))

        img_embedding = self.linear2(self.img_embedding[related_items])
        img_embedding = torch.reshape(img_embedding, (-1, 300, self.emb_size))

        text_embedding = self.pos_embedding(text_embedding)
        img_embedding = self.pos_embedding(img_embedding)

        bsz, lens, _ = img_embedding.size()

        text_embed = torch.reshape(text_embedding, (-1, self.emb_size))
        img_embed = torch.reshape(img_embedding, (-1, self.emb_size))

        text_gcn_output = text_embed
        img_gcn_output = img_embed

        all_text_emb = []
        all_img_emb = []

        for layer in range(self.layer):

            text_gcn_output =  self.hgConv_list1[layer](text_gcn_output, hg_idx, img_gcn_output)
            img_gcn_output = self.hgConv_list2[layer](img_gcn_output, hg_idx, text_gcn_output)
            all_text_emb += [text_gcn_output]
            all_img_emb += [img_gcn_output]

        text_gcn_output = all_text_emb[-1]
        img_gcn_output = all_img_emb[-1]

        text_output = text_gcn_output.detach().cpu().numpy()
        img_output = img_gcn_output.detach().cpu().numpy()
        pickle.dump(text_output, open('text.pickle', 'wb'))
        pickle.dump(img_output, open('img.pickle', 'wb'))


        text_gcn_output = torch.reshape(text_gcn_output, (bsz, lens, -1))
        img_gcn_output  = torch.reshape(img_gcn_output,  (bsz, lens, -1))

        text_gcn_output = text_gcn_output[:,0, :]
        img_gcn_output  = img_gcn_output[:,0, :]

        text_user = torch.cat([text_gcn_output, user_embedding], 1)
        img_user = torch.cat([img_gcn_output, user_embedding], 1)

        text_img_user = torch.cat([text_user, img_user], -1)
        text_img_user = self.linear3(text_img_user)#b*d
        text_img_user = text_img_user.unsqueeze(1)#b*1*d


        text_user_t = text_user.unsqueeze(1)
        img_user_t = img_user.unsqueeze(1)
        text_img_user_t = torch.cat([text_user_t, img_user_t], 1) #b*2*d

        output = self.fusing_attn(text_img_user, text_img_user_t, text_img_user_t)

        output = self.dense(torch.squeeze(output))
        output = self.linear4(output)

        return output