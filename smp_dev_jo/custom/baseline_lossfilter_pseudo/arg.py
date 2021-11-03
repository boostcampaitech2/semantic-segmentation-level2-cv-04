from easydict import EasyDict as eDict

def getArg():
	arg = eDict()

	arg.batch = 16
	arg.epoch = 20
	arg.lr = 1e-4
	arg.seed = 21
	arg.save_capacity = 5
	
	arg.image_root = "../input/data"
	arg.train_json = "train_0.json"
	arg.val_json = "valid_0.json"
	arg.output_path = "../output"

	arg.train_worker = 4
	arg.valid_worker = 4

	arg.wandb = True
	arg.wandb_project = "segmentation"
	arg.wandb_entity = "cv4"

	arg.custom_name = "efficient_baseline_lossfilter_pseudo"

	return arg