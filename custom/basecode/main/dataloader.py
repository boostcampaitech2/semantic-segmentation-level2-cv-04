from torch.utils.data.dataloader import DataLoader

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def getDataloader(trainDataset, validDataset, batch, trainWorker, validWorker):


	trainDataloader = DataLoader(trainDataset, batch_size=batch, shuffle=True, pin_memory=True, num_workers=trainWorker, collate_fn=collate_fn)
	validDataloader = DataLoader(validDataset, batch_size=batch, shuffle=False,pin_memory=True, num_workers=validWorker, collate_fn=collate_fn)


	return trainDataloader, validDataloader


def getTrainDataloader(trainDataset, batch, trainWorker):


	trainDataloader = DataLoader(trainDataset, batch_size=batch, shuffle=True, pin_memory=True, num_workers=trainWorker, collate_fn=collate_fn)


	return trainDataloader


def getTestDataloader(testDataset, batch, testWorker):
	testDataloader = DataLoader(testDataset, batch_size=batch, num_workers=testWorker, collate_fn=collate_fn)

	return testDataloader