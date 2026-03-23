import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from attr_clip import RobustClipAttributeEncoder, AADB_PROMPTS_11
from swin_transformer import swin_base_patch4_window7_224_in22k


class EncoderText(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()["hidden_size"]

        self.rnn = nn.GRU(
            embedding_dim,
            2048,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.25
        )

        self.dropout = nn.Dropout(0.25)

    def forward(self, text, attention_mask=None):
        bert_out = self.bert(
            input_ids=text,
            attention_mask=attention_mask
        )
        embedded = bert_out[0]

        outs, hidden = self.rnn(embedded)
        outs = (outs[:, :, :outs.size(2) // 2] + outs[:, :, outs.size(2) // 2:]) / 2
        o = torch.mean(outs, dim=1)

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        return o, outs


class Attention_M(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1,
                 score_function='scaled_dot_product', dropout=0):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function

        self.w_kx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.w_qx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        self.w_kx.data.uniform_(-stdv, stdv)
        self.w_qx.data.uniform_(-stdv, stdv)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)

        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]

        kx = k.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)
        qx = q.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)

        kx = torch.bmm(kx, self.w_kx).view(-1, k_len, self.hidden_dim)
        qx = torch.bmm(qx, self.w_qx).view(-1, q_len, self.hidden_dim)

        if self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')

        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output


class MIMN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hops = 3

        self.attention_text = Attention_M(2048, score_function='mlp')
        self.attention_img = Attention_M(2048, score_function='mlp')
        self.attention_text2img = Attention_M(2048, score_function='mlp')
        self.attention_img2text = Attention_M(2048, score_function='mlp')

        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 2048)

    def forward(self, image_feature, attr_feature, txt_feature):
        et_text = attr_feature
        et_img = attr_feature

        for _ in range(self.hops):
            it_al_text2text = self.attention_text(txt_feature, et_text).squeeze(dim=1)
            it_al_img2text = self.attention_img2text(txt_feature, et_img).squeeze(dim=1)
            it_al_text = (it_al_text2text + it_al_img2text) / 2

            it_al_img2img = self.attention_img(image_feature, et_img).squeeze(dim=1)
            it_al_text2img = self.attention_text2img(image_feature, et_text).squeeze(dim=1)
            it_al_img = (it_al_img2img + it_al_text2img) / 2

            et_text = self.fc1(it_al_text)
            et_img = self.fc2(it_al_img)

        et = torch.cat((et_text, et_img), dim=-1)
        et = et.sum(dim=1)
        return et


class catNet(nn.Module):
    def __init__(self, bert, freeze_clip=True):
        super().__init__()
        self.fusion = MIMN()
        self.txt_enc = EncoderText(bert)

        self.attr_enc = RobustClipAttributeEncoder(
            prompts=AADB_PROMPTS_11,
            out_dim=2048,
            freeze_clip=freeze_clip
        )

        self.img_enc = swin_base_patch4_window7_224_in22k()

        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6144, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc4 = nn.Linear(1024, 2048)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, text, text_mask, image_att_or_feat, return_attr_weights=False):
        txt_result, word_feature = self.txt_enc(text, text_mask)

        img_feature = self.img_enc(image)
        img_feature = self.fc4(img_feature)

        if return_attr_weights:
            img_attr, attr_weights = self.attr_enc(image_att_or_feat, return_weights=True)
        else:
            img_attr = self.attr_enc(image_att_or_feat, return_weights=False)
            attr_weights = None

        out = self.fusion(img_feature, img_attr, word_feature)
        h = torch.cat((out, txt_result), dim=1)

        h = self.drop(h)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        h = self.softmax(h)

        if return_attr_weights:
            return h, attr_weights
        return h