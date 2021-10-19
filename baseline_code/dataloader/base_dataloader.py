from torch.utils.data.dataloader import DataLoader

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


class BaseDataLoader(DataLoader):

	def __init__(self, *args, **kwargs):
		super().__init__(collate_fn=collate_fn, *args, **kwargs)

