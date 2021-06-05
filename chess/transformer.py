import torch
import torch.nn as nn
import numpy as np

from Seq2SlateModel import FeatureEmbedding

# architecture
# positional encoding is added to each of the feature vectors

# One transformer encoder block:
# Attention(x) = LayerNorm(x+Dropout(MultiHeadAtt(x)))

# FFN(x) = LayerNorm(x+Dropout(Dense(ReLU(Dense(x))))) # elementwise


class FFN(nn.Module):
    def __init__(self,hidden_dim,inner_dim):
        super().__init__()
        with self.name_scope():
            self.linear1 = nn.Dense(inner_dim,flatten=False)
            self.linear2 = nn.Dense(hidden_dim,flatten=False)
            self.relu = gluon.nn.Activation('relu')
    def forward(self,X):
        return self.linear2(self.relu(self.linear1(X)))

def Attention(Q,K,V,P=None):
    """Self attention mechanism, softmax(QK^T/sqrt(d))V
       assumes Q,K,V have shape (bs,n,d). Optionally includes relative
       positional encoding matrix P with shape (n,n,d)."""
    bs,n,d = Q.shape
    Kt = K.transpose((0,2,1)) # (bs,n,d)->(bs,d,n)
    scores = nd.linalg.gemm2(Q,Kt) # (bs,n,n)                          
                                #(n,bs,d) x (n,d,n) -> (n,bs,n) -> (bs,n,n)
    pos_scores = nd.linalg.gemm2(Q.transpose((1,0,2)),P).transpose((1,0,2)) if P is not None else 0
    weighting = nd.softmax((scores+pos_scores)/np.sqrt(d),axis=-1)
    weighted_values = nd.linalg.gemm2(weighting,V)
    return weighted_values

class SelfAttentionHead(gluon.Block):
    def __init__(self,head_dim,P=None):
        super().__init__()
        with self.name_scope():
            self.WQ = nn.Dense(head_dim,flatten=False)
            self.WK = nn.Dense(head_dim,flatten=False)
            self.WV = nn.Dense(head_dim,flatten=False)

    def forward(self,X,P=None):
        """Expects X shape (bs,n,d)"""
        return Attention(self.WQ(X),self.WK(X),self.WV(X),P)

class MultiHeadAttention(gluon.Block):
    k = 2 # 2 is enough for relative position see https://www.aclweb.org/anthology/N18-2074
    def __init__(self,hidden_dim,num_heads=4,rel_pos=False):
        """ The query,key, and value dimensions are hidden_dim/num_heads"""
        super().__init__()
        with self.name_scope():
            self.heads = nn.Sequential()
            for _ in range(num_heads):
                self.heads.add(SelfAttentionHead(hidden_dim//num_heads))
            self.WO = nn.Dense(hidden_dim,flatten=False) 
            self.positionalEmbedding = nn.Embedding(2*self.k+1,hidden_dim//num_heads) if rel_pos else None
    def forward(self,X):
        """Expects X shape (bs,n,hidden_dim)"""
        bs,n,hd = X.shape
        if self.positionalEmbedding is not None:
            rel_positions = ((nd.arange(n).expand_dims(-1)-nd.arange(n)).clip(-self.k,self.k)+self.k).astype(int)
            P = self.positionalEmbedding(rel_positions).transpose((0,2,1)) # (n,n,d) -> (n,d,n)
        else: P = None
        return self.WO(nd.Concat(*[head(X,P) for head in self.heads],dim=-1))


class AddAndNorm(gluon.Block):
    def __init__(self,block,dropout=0):
        super().__init__()
        with self.name_scope():
            self.block = block
            self.layerNorm = nn.LayerNorm()
            self.dropout = nn.Dropout(dropout)
    def forward(self,X):
        """Expects X shape (bs,n,d)"""
        return self.layerNorm(X+self.dropout(self.block(X)))

class TransformerBlock(gluon.Block):
    def __init__(self,hidden_dim,inner_dim,num_heads=4,dropout=0,rel_pos=True):
        super().__init__()
        with self.name_scope():
            MHA = MultiHeadAttention(hidden_dim,num_heads,rel_pos)
            self.attention_block = AddAndNorm(MHA,dropout)
            FF = FFN(hidden_dim,inner_dim)
            self.feed_forward_block = AddAndNorm(FF,dropout)
    def forward(self,X):
        """Expects X shape (bs,n,hidden_dim)"""
        return self.feed_forward_block(self.attention_block(X))


class SlateTransformer(gluon.Block):
    def __init__(self,hidden_dim,vocab_size,num_blocks,inner_dim,num_heads=4,embed_dim=16,dropout=0):
        """Hidden dim specifies the dimension of x, inner dim the middle size of the FFN """
        super().__init__()
        with self.name_scope():
            self.embedding = FeatureEmbedding(vocab_size,embed_dim,dropout=dropout)
            self.net = nn.Sequential()
            self.net.add(nn.Dense(hidden_dim,flatten=False))
            with self.net.name_scope():
                for _ in range(num_blocks):
                    self.net.add(TransformerBlock(hidden_dim,inner_dim,num_heads,dropout))
                self.net.add(nn.Dense(1,flatten=False))
    def forward(self, x):
        """assumes that x is of shape (n x bs x .)"""
        embedded_inputs = self.embedding(x).transpose((1,0,2)) # (n,bs,d) -> (bs,n,d)
        return self.net(embedded_inputs)[:,:,0].T # (bs,n,1) -> (n,bs,1)