import argparse
from estimator import run

parser = argparse.ArgumentParser()
parser.add_argument("mode")
parser.add_argument("base_path")
parser.add_argument("model_dir")
parser.add_argument("-B", "--batch_size",
                    type=int,
                    default=128)
parser.add_argument("-L", "--learning_rate",
                    nargs=2,
                    type=float,
                    default=[0.001, 0.0000001],
                    help="Initial/final learning rate.")
parser.add_argument("-D", "--decay",
                    nargs=2,
                    type=float,
                    default=[100000, 2.0],
                    help="Decay steps and power.")
parser.add_argument("-R", "--reg",
                    nargs=2,
                    default=[None, 0.0],
                    help="Regularization type and coefficient.")
parser.add_argument("-M", "--mel",
                    action="store_true",
                    help="Use mel spectrogram data instead (from TFR!!).")
parser.add_argument("-X", "--mlp",
                    type=int,
                    default=0,
                    help="Use MLP with FLAG hidden units...")
parser.add_argument("-Y", "--dropout",
                    action="store_true",
                    help="Use dropout in hidden layer.")
parser.add_argument("-C", "--conv",
                    action="store_true",
                    help="Use CNN.")
args = parser.parse_args()

run(args.mode, args.base_path, args.model_dir,
    args.batch_size, args.learning_rate, args.decay, args.reg, args.mel,
    args.mlp, args.dropout, args.conv)
