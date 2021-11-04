from easydict import EasyDict as eDict

def getArg():
	arg = eDict()

	arg.batch = 12
	arg.epoch = 80
	arg.lr = 1e-4
	arg.seed = 42
	arg.save_capacity = 1
	arg.patience = 15
	
	arg.image_root = "../../input/data"
	arg.train_json = "stratified_json/train_0.json"
	arg.val_json = "stratified_json/valid_0.json"
	arg.output_path = "./output"

	arg.test_json = "test.json"
	arg.submission_path = "./submission"

	arg.train_worker = 4
	arg.valid_worker = 4
	arg.test_worker = 4

	arg.model = "DeepLabV3Plus" # CamelCase 형태
	arg.backbone = "resnest269e" # 전체 소문자 형태
	arg.loss = "cross_entropy" # snake_case 형태
	arg.transform = "flip_transform" # snake_case 형태

	arg.inference_transform = "inference_transform"

	arg.wandb = True
	arg.wandb_project = "segmentation"
	arg.wandb_entity = "seunghyukshin"

	arg.custom_name = "17_DeepLabV3Plus_ResNest269e_flipAug_patience15"

	# 원하는 객체 best IoU 가지는 epoch 뽑고 싶은 클래스 번호 
	arg.object_best = [1, 3, 9, 10]

	return arg