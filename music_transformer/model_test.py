from os.path import dirname, realpath, sep, pardir
import sys
import torch
import unittest
import model

class UT(unittest.TestCase):
    def test1(self):
        self.assertTrue(1 >= 2)
batch_size = 2
len_Q = 3
len_K = 3
d = 512
vocab_size = 11
max_seq = 64
input_len = 10
length = 100
net = model.MusicTransformer(embedding_dim=d, vocab_size=vocab_size, num_layer=6,
                 max_seq=max_seq, dropout=0.2)
A = torch.randint(0, vocab_size-1, (batch_size, input_len))
net.eval()
B = net(A, length)

net.train()
B = net(A, 100)