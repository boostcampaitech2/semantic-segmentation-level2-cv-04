from collections import deque
import os
import torch

class SaveHelper:
	def __init__(self, capacity) -> None:
		self.savedList = deque()
		self.capacity = max(capacity,2)
		self.bestEpoch = 0
		self.bestLoss = 9999999
	
	@staticmethod
	def _fileFormat(epoch):
		return f"epoch{epoch}.pth"

	def checkBestLoss(self, avrg_loss, epoch):
		
		ok = avrg_loss<self.bestLoss
		
		self.best_loss = avrg_loss
		
		self.savedList.append(epoch)
		self.bestEpoch = epoch

		return ok
	
	def removeModel(self,saved_dir):

		if len(self.savedList) <= self.capacity:
			return

		delTarget = self.savedList.popleft()
		os.remove(os.path.join(saved_dir,self._fileFormat(delTarget)))
	
	def saveModel(self,epoch, saved_dir, model):
		torch.save(model.state_dict(), os.path.join(saved_dir,self._fileFormat(epoch)))
  
	def renameBestModel(self):
		os.rename(self._fileFormat(self.bestEpoch),"best.pth")
