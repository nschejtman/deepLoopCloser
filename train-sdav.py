from src.sdav.network.StackedDenoisingAutoencoderVariants import SDA
from src.sdav.network.StackedDenoisingAutoencoderVariants import SDA
from argparse import ArgumentParser

# Get arguments
parser = ArgumentParser(description='Use this main file to train the network')
# Positional arguments
parser.add_argument('operation', choices=['train', 'transform'], help='Operation to perform')
# Named arguments
parser.add_argument('--dataset_dir', help='Path to the dataset directory', required=True)
parser.add_argument('--dataset_ext', help='Extension of the image files in the dataset directory',
                    required=True)
# Named optional arguments
parser.add_argument('--input_shape', help='Shape of the input layer', type=int, nargs=2, default=[30, 1681])
parser.add_argument('--hidden_units', help='Number of hidden units', type=int, nargs='+',
                    default=[2500, 2500, 2500, 2500, 2500])
parser.add_argument('--batch_size', help='Batch size for training', type=int, default=10)
parser.add_argument('--corruption_level', help='Percentage of input vector to corrupt', type=float, default=0.3)
parser.add_argument('--sparse_penalty', help='Penalty weight for the sparsity constraint', type=float,
                    default=1.0)
parser.add_argument('--sparse_level', help='Threshold factor for the sparsity constraint', type=float,
                    default=0.05)
parser.add_argument('--consecutive_penalty', help='Penalty weight for consecutive constraint', type=float,
                    default=0.2)
parser.add_argument('--learning_rate', help='Learning rate', type=float, default=0.1)
parser.add_argument('--epochs', help='Number of epochs to train each batch', type=int, default=100)
parser.add_argument('--verbose', help='Verbosity level for operations', type=bool, default=True)

conf = parser.parse_args()

# Create network
model = SDA(conf.input_shape,
            conf.hidden_units,
            sparse_level=conf.sparse_level,
            sparse_penalty=conf.sparse_penalty,
            consecutive_penalty=conf.consecutive_penalty,
            batch_size=conf.batch_size,
            learning_rate=conf.learning_rate,
            epochs=conf.epochs,
            corruption_level=conf.corruption_level)

dataset_path = ('%s/*.%s' % (conf.dataset_dir, conf.dataset_ext)).replace('*..', '*.').replace('//', '/')

if conf.operation == 'train':
    model.fit(dataset_path)
