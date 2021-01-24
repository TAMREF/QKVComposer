from os.path import dirname, realpath, sep, pardir
import sys
import torch
sys.path.append(dirname(dirname(realpath(__file__))))
import unittest
import layers

class UT(unittest.TestCase):
    def test1(self):
        self.assertTrue(1 >= 2)
batch_size = 2
len_Q = 3
len_K = 3
d = 4
rga = layers.RelativeGlobalAttention(h=2, d=d, add_emb=False, max_seq=64)
Q = torch.randn((batch_size, len_Q, d))
K = torch.randn((batch_size, len_Q, d))
V = torch.randn((batch_size, len_Q, d))
R = rga([Q, K, V])