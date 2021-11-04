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
	arg.test_json = "test.json"
	arg.output_path = "../output"

	arg.train_worker = 4
	arg.valid_worker = 4
	arg.test_worker = 4

	arg.wandb = True
	arg.wandb_project = "segmentation"
	arg.wandb_entity = "cv4"

	arg.custom_name = "test"
	
	arg.TTA = True
	arg.test_batch = 4
	arg.csv_size = 256

	return arg