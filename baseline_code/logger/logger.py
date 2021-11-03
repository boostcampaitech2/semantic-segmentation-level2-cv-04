import os
import shutil

# 디렉토리 생성
def make_dir(saved_dir, saved_name):
    path = os.path.join(saved_dir, saved_name)
    os.makedirs(path, exist_ok=True)

    return path

# yaml 파일 saved 폴더에 저장
def yaml_logger(args, cfg):
    file_name = f"{cfg['exp_name']}.yaml"
    shutil.copyfile(args.config, os.path.join(cfg['saved_dir'], file_name))

# Best validation score 갱신해서 모델 저장될 때
# best_log.txt 파일에 mIoU 값과 클래스별 IoU 값 기록
def best_logger(saved_dir, epoch, num_epochs, best_mIoU, IoU_by_class):
    with open(os.path.join(saved_dir, 'best_log.txt'), 'a', encoding='utf-8') as f:
        f.write(f"Epoch [{epoch+1}/{num_epochs}], Best mIoU :{best_mIoU}\n")
        f.write(f"IoU by class : {IoU_by_class}\n")