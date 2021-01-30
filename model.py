import torch
import torch.nn as nn
import torch.nn.functional as F


class VQ(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.codes_dim = config.codes_dim
        self.codes_cnt = config.codes_cnt
        self.codes_emb = nn.Embedding(config.codes_cnt, config.codes_dim)
        self.device = config.device
        self.commit_cost = config.commit_cost
        
    def forward(self, x): #should return b,c,h,w, where c = codes_dim
        b,c, L = x.size()
        x = x.transpose(0, 1).reshape(c, -1) #c, toks
        
        ## (v - w)**2 = v**2 + w**2 - 2 v@w
        X2 = (x**2).sum(dim = 0, keepdim = True)
        Y2 = (self.codes_emb.weight**2).sum(dim = 1, keepdim = True)
        XY = torch.einsum("ct,dc->dt",x,self.codes_emb.weight)
        #print(X2.shape, Y2.shape, XY.shape)
        ords = X2 + Y2 - 2 * XY
        
        enc_ind = torch.argmin(ords, dim = 0).to(self.device)
        qs = self.codes_emb(enc_ind).transpose(0,1) # toks * codes_cnt

        loss = F.mse_loss(qs, x.detach()) + self.commit_cost * F.mse_loss(qs.detach(), x) 
        qs = x + (qs - x).detach() # this flows gradient from reconstruction to stuff before x
        
        qs = qs.reshape(c,b,L).transpose(0,1)

        return loss, qs
    
    def forward_only_indices(self, x):
        b,c,L = x.size()
        x = x.transpose(0, 1).contiguous().view(c, -1) #c, toks
        
        ## (v - w)**2 = v**2 + w**2 - 2 v@w
        X2 = (x**2).sum(dim = 0, keepdim = True)
        Y2 = (self.codes_emb.weight**2).sum(dim = 1, keepdim = True)
        XY = torch.einsum("ct,dc->dt",x,self.codes_emb.weight)
        #print(X2.shape, Y2.shape, XY.shape)
        ords = X2 + Y2 - 2 * XY
        
        enc_ind = torch.argmin(ords, dim = 0).contiguous().to(self.device)
        return enc_ind.reshape(b, -1)

    def forward_from_indices(self, idxs):
        #indices should be B, L
        qs = self.codes_emb(idxs).permute(0, 2, 1)
        return qs

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = (in_ch + out_ch)//2
        self.main = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_ch, mid_ch, 3, 1, 1, bias = False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(),
            nn.Conv1d(mid_ch, out_ch, 1, 1, bias = False)
        )
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return self.bn(x + self.main(x))

class VQ_VAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.init_c = config.init_c
        self.codes_dim = config.codes_dim
        ch = 4

        self.encoder = nn.Sequential(
            nn.Conv1d(self.init_c, 8 * ch, 4, 2, 1, bias = False),
            nn.ReLU(),
            nn.Conv1d(8 * ch,  16 * ch, 4, 2, 1, bias = False),
            nn.BatchNorm2d( 16 * ch),
            ResBlock(16 * ch, 16 * ch),
            ResBlock(16 * ch, 16 * ch),
            nn.ReLU(),
            nn.Conv1d( 16 * ch, 32 * ch, 4, 2, 1, bias = False),
            nn.BatchNorm2d(32 * ch),
            nn.ReLU(),
            nn.Conv1d(32 * ch, self.codes_dim, 3, 1, 1, bias = False)
        )

        self.vq = VQ(config)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.codes_dim, 32 * ch, 3, 1, 1, bias = False),
            nn.BatchNorm2d(32 * ch),
            nn.ReLU(),
            nn.ConvTranspose1d(32 * ch,  16 * ch, 4, 2,1, bias=False),
            nn.BatchNorm2d( 16 * ch),
            ResBlock(16 * ch, 16 * ch),
            ResBlock(16 * ch, 16 * ch),
            nn.ReLU(),
            nn.ConvTranspose1d( 16 * ch, 8 * ch, 4, 2,1, bias=False),
            nn.BatchNorm2d(8 * ch),
            nn.ReLU(),
            nn.ConvTranspose1d(8 * ch, self.init_c, 4, 2,1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):

        z = self.encoder(x)
        loss, z = self.vq(z)
        #print(z.shape)
        x_rec = self.decoder(z)

        return loss, x_rec

    def forward_from_indices(self, idxs):
        
        qs = self.vq.forward_from_indices(idxs)
        return self.decoder(qs)


class GPT(nn.Module):
    def __init__(self, d_model = 512, num_layers = 8, n_vocab = 256, dim_feedforward = 512, n_head = 4, max_len = 256, device = "cpu"):
        super().__init__()
        
        self.tok_emb_dec = nn.Embedding(n_vocab, d_model)
        self.pos_emb_dec = nn.Parameter(torch.zeros(1, max_len, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward= dim_feedforward)        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = num_layers, norm = nn.LayerNorm(d_model))

        mask = torch.triu(torch.ones(max_len, max_len), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))

        self.mask = mask.reshape(max_len, max_len).to(device)
        self.head = nn.Linear(d_model, n_vocab)
        self.init_token = n_vocab
        self.device = device

    def forward(self, x):
        
        # appended 0:
        B, T_d = x.size()
        x = self.tok_emb_dec(x)
        x = self.pos_emb_dec[:, :T_d, :] + x
        
        mask = self.mask[:T_d, :T_d]
        
        output = self.decoder(tgt = x.transpose(0, 1), memory = x.transpose(0, 1), tgt_mask = mask, memory_mask = mask).transpose(0, 1)
        output = self.head(output)
        return output

    def sample(self, init_class = 0, gen_len = 10, T = 0.7, n = 500):
        
        gens = torch.ones(n, 1) * (init_class)
        gens = gens.long().to(self.device)
        with torch.no_grad():
            for i in range(1, gen_len + 1):
                
                src = self.tok_emb_dec(gens)
                #print(src.shape)
                src = self.pos_emb_dec[:, :i, :] + src
                out= self.decoder(tgt = src.transpose(0, 1), memory = src.transpose(0, 1), tgt_mask = self.mask[:i, :i], memory_mask = self.mask[:i, :i]).transpose(0, 1)
                out = self.head(out)[:, -1, :]

                #print(torch.max(out, dim = -1))
                out = out - out.max()
                out = out[:, :512]
                out = F.softmax(out/T, dim = -1) + 1e-9
                
                out = torch.multinomial(out, num_samples=1)
                out = out.reshape(n, 1)
                gens = torch.cat([gens, out], dim = 1)
        
        return gens[:, 1:]
