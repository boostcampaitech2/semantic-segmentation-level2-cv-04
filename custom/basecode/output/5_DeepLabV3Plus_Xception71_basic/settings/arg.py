from easydict import EasyDict as eDict

def getArg():
	arg = eDict()

	arg.batch = 12
	arg.epoch = 60
	arg.lr = 1e-4
	arg.seed = 42
	arg.save_capacity = 3
	arg.patience = 7
	
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

	arg.custom_name = "5_DeepLabV3Plus_Xception71_basic"

	return arg