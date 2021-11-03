from collections import deque
import os
import torch

class SaveHelper:
	def __init__(self, capacity, saved_dir) -> None:
		self.savedList = deque()
		self.capacity = max(capacity,2)
		self.bestEpoch = 0
		self.bestLoss = 9999999
		self.savedDir = saved_dir

	@staticmethod
	def _fileFormat(epoch):
		return f"epoch{epoch}.pth"

	def checkBestLoss(self, avrg_loss, epoch):
		
		ok = avrg_loss<self.bestLoss
		
		self.best_loss = avrg_loss
		
		self.savedList.append(epoch)
		self.bestEpoch = epoch

		return ok
	
	def _concatSaveDir(self, fileName):
		return os.path.join(self.savedDir,fileName)

	def _concatSaveDirByEpoch(self, epoch):
		return self._concatSaveDir(self._fileFormat(epoch))

	def removeModel(self):

		if len(self.savedList) <= self.capacity:
			return

		delTarget = self.savedList.popleft()
		os.remove(self._concatSaveDirByEpoch(delTarget))
	
	def saveModel(self,epoch, model):
		torch.save(model.state_dict(), self._concatSaveDirByEpoch(epoch))
  
	def renameBestModel(self):
		os.rename(self._concatSaveDirByEpoch(self.bestEpoch),self._concatSaveDir("best.pth"))
