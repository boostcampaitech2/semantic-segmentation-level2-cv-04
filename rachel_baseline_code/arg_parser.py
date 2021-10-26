import argparse
import yaml

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

    # Transformation
    parser.add_argument('--transform', default='coffee', type=str)

    return parser

def arg_parser_infer():
    parser = argparse.ArgumentParser('Semantic Segmentation Inference', add_help=False)

    # Config file path from training
    parser.add_argument('--yaml_path', type=str)

    # Transformation for TTA
    parser.add_argument('--transform', default='water', type=str)

    # Read yaml file
    args = parser.parse_args()
    with open(args.yaml_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    # Import parameters from yaml
    parser.add_argument('--seed', default=data_loaded['seed'], type=int)
    parser.add_argument('--batch_size', default=data_loaded['batch_size'], type=int)
    parser.add_argument('--model_path', default=data_loaded['output_path'], type=str)

    return parser