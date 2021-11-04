from easydict import EasyDict as eDict

def getArg():
	arg = eDict()

	arg.batch = 16
	arg.epoch = 60
	arg.lr = 1e-4
	arg.seed = 42
	arg.save_capacity = 1
	arg.patience = 10
	
	arg.image_root = "/opt/ml/segmentation/input/data"
	arg.train_json = "stratified_json/train_0.json"
	arg.val_json = "stratified_json/valid_0.json"
	arg.output_path = "/opt/ml/segmentation/custom/basecode/output"

	arg.test_json = "test.json"
	arg.submission_path = "/opt/ml/segmentation/custom/basecode/submission"

	arg.train_worker = 4
	arg.valid_worker = 4
	arg.test_worker = 4

	arg.wandb = True
	arg.wandb_project = "segmentation"
	arg.wandb_entity = "seunghyukshin"

	arg.custom_name = "11_UnetPlusPlus_ResNet101_basic"

	# 원하는 객체 best IoU 가지는 epoch 뽑고 싶은 클래스 번호 
	arg.object_best = [1, 3, 9, 10]

	return arg