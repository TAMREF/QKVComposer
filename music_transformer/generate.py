import custom
from custom import criterion
from custom.layers import *
from custom.config import config
from model import MusicTransformer
from data import Data
import utils
from midi_processor.processor import decode_midi, encode_midi

import datetime
import argparse

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preprocess'))
import torch

from tensorboardX import SummaryWriter


parser = custom.get_argument_parser()
args = parser.parse_args()
config.load(args.model_dir, args.configs, initialize=True)

# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')


current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
gen_log_dir = 'logs/mt_decoder/generate_'+current_time+'/generate'
gen_summary_writer = SummaryWriter(gen_log_dir)

#define model from args.model_dir
mt = MusicTransformer(
    embedding_dim=config.embedding_dim,
    vocab_size=config.vocab_size,
    num_layer=config.num_layers,
    max_seq=config.max_seq,
    dropout=0,
    debug=False)
mt.load_state_dict(torch.load(args.model_dir+'/final.pth'))
mt.test()

#from config file, make input tensor
if config.condition_file is not None:
    #from condition file
    condition_midi = torch.load(config.condition_file)
    inputs = np.array([condition_midi[:config.condition_length]])
else:
    #start with one chord
    inputs = np.array([[24, 28, 31]])

#generate event tensor using model
inputs = torch.from_numpy(inputs)
result = mt(inputs, config.length, gen_summary_writer)

#save midi file using tensor2list, list2midi
from preprocess_utils import *

LIST_rec = tensor2list(result)
list2midi(LIST_rec, ofpath = os.path.join(config.save_path, "test_rec.midi"))

gen_summary_writer.close()
