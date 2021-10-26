import argparse

def arg_parser():
    parser = argparse.ArgumentParser('Semantic Segmentation', add_help=False)

    # File name (Mandatory)
    parser.add_argument('--exp_name', type=str)

    # Hyper-parameters
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--scheduler', default='cosine', type=str)

    # transformation
    parser.add_argument('--transform', default='coffee', type=str)

    return parser

