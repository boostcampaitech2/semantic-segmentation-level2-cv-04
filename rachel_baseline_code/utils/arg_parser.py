import argparse
import yaml

def arg_parser():
    parser = argparse.ArgumentParser('Semantic Segmentation', add_help=False)

    # File name (Mandatory)
    parser.add_argument('--exp_name', type=str)

    # Hyper-parameters
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_epochs', default=60, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--scheduler', default='cosine', type=str)
    parser.add_argument('--loss', default='CE', type=str)
    parser.add_argument('--opt_name', default='Adam', type=str)

    # Modeling
    parser.add_argument('--model', default='DeepLabV3Plus_xception71', type=str)
    parser.add_argument('--train_path', default='/train_0.json', type=str)
    parser.add_argument('--valid_path', default='/valid_0.json', type=str)

    # Transformation
    parser.add_argument('--transform', default='coffee', type=str)

    # argparse에서 boolean 값 받아오는 함수
    def str2bool(v): 
        if isinstance(v, bool): 
            return v 
        if v.lower() in ('yes', 'true', 't', 'y', '1'): 
            return True 
        elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
            return False 
        else: 
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # WandB
    parser.add_argument('--wandb', default=True, type=str2bool)
    parser.add_argument('--wandb_project', default='segmentation', type=str)
    parser.add_argument('--wandb_entity', default='cv4', type=str)
    parser.add_argument('--wandb_custom_name', default='test', type=str)

    return parser

def arg_parser_infer():
    parser = argparse.ArgumentParser('Semantic Segmentation Inference', add_help=False)

    # Config file path from training
    parser.add_argument('--yaml_path', type=str)

    # Transformation for TTA
    parser.add_argument('--tta', default='water', type=str)

    # Read yaml file
    args = parser.parse_args()
    with open(args.yaml_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    # Import parameters from yaml
    parser.add_argument('--exp_name', default=data_loaded['exp_name'], type=str)
    parser.add_argument('--seed', default=data_loaded['seed'], type=int)
    parser.add_argument('--batch_size', default=data_loaded['batch_size'], type=int)
    parser.add_argument('--model_path', default=data_loaded['output_path'], type=str)
    parser.add_argument('--model', default=data_loaded['model'], type=str)

    return parser
