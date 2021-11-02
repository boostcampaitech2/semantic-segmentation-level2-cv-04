import os
import json
import shutil

def make_dir(saved_dir, saved_name):
    path = os.path.join(saved_dir, saved_name)
    os.makedirs(path, exist_ok=True)

    return path

def config_logger(args):
    with open(os.path.join(args.saved_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

def yaml_logger(args, cfg):
    file_name = args.config.split('/')[-1]
    shutil.copyfile(args.config, os.path.join(cfg['saved_dir'], file_name))

def best_logger(saved_dir, epoch, num_epochs, best_mIoU, IoU_by_class):
    with open(os.path.join(saved_dir, 'config.txt'), 'a', encoding='utf-8') as f:
        f.write(f"Epoch [{epoch+1}/{num_epochs}], Best mIoU :{best_mIoU}\n")
        f.write(f"IoU by class : {IoU_by_class}\n")