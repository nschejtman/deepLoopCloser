from argparse import ArgumentParser

from input.CvInputParser import Parser
from model.DenoisingAutoencoderVariant import DAVariant, defaults
from utils.BufferedFileReader import BufferedReader
from utils.DictMerger import overwrite_dict

# Initialize argument parser
arg_parser = ArgumentParser(description="Use this main file to train the model")
arg_parser.add_argument('dir', help='Path to the dataset directory')
arg_parser.add_argument('ext', help='Extension of the image files in the dataset directory')
arg_parser.add_argument('batch_size', help='Batch size for training', type=int)
arg_parser.add_argument('--keypoints', help='Number of keypoints to extract. Read paper', type=int)
arg_parser.add_argument('--patch_size', help='Patch size to expand from keypoints. Read paper', type=int)
arg_parser.add_argument('--hidden_units', help='Number of hidden units', type=int)
arg_parser.add_argument('--corruption_level', help='Percentage of input vector to corrupt. Read paper', type=float)
arg_parser.add_argument('--sparse_penalty', help='Penalty weight for the sparsity constraint. Read paper', type=float)
arg_parser.add_argument('--sparse_level', help='Threshold factor for the sparsity constraint. Read paper', type=float)
arg_parser.add_argument('--consecutive_penalty', help='Penalty weight for consecutive constraint. Read paper',
                        type=float)
arg_parser.add_argument('--learning_rate', help='Learning rate', type=float)
arg_parser.add_argument('--epochs', help='Number of epochs to train each batch', type=int)

# Merge argument parameters with default parameters
args = arg_parser.parse_args()
conf = overwrite_dict(defaults, args.__dict__, skip_nones=True)

# Initialize model
reader = BufferedReader(conf['dir'], conf['ext'], conf['batch_size'])
n_batches = len(reader)
da = DAVariant.from_dict(conf)
parser = Parser(conf['keypoints'], conf['patch_size'])

for i, batch in enumerate(reader):
    print("Started batch: " + str(i) + "/" + str(n_batches))
    parsed_batch = parser.calculate_all_from_path(batch)
    da.fit(parsed_batch, warm_start=True)
