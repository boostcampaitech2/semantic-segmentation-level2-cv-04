from collections import deque
import os
import torch

class SaveHelper:
	def __init__(self, capacity, saved_dir) -> None:
		self.savedList = deque()
		self.capacity = max(capacity,2)
		self.bestEpoch = 0
		self.bestLoss = 9999999
		self.bestmIoU = 0
		self.savedDir = saved_dir

		# General trash, Paper pack, Battery, Clothing 가장 잘 나온 epoch 모델 저장 (1, 3, 9, 10)
		self.bestEpochObject = [0] * 11
		self.bestmIoUObject = [0] * 11
		self.object_dict = {
			0 : 'background', 1 : 'general_trash', 2 : 'paper', 3 : 'paper_pack', 4 : 'metal', 
			5 : 'glass', 6 : 'plastic', 7 : 'styrofoam', 8 : 'plastic_bag', 9 : 'battery', 10 : 'clothing'}


	@staticmethod
	def _fileFormat(epoch):
		return f"epoch{epoch}.pt"

	# def checkBestLoss(self, avrg_loss, epoch):
		
	# 	ok = avrg_loss<self.bestLoss
		
	# 	self.best_loss = avrg_loss
		
	# 	self.savedList.append(epoch)
	# 	self.bestEpoch = epoch

	# 	return ok

	def checkBestmIoU(self, mIoU, epoch):
		if mIoU > self.bestmIoU:
			self.bestmIoU = mIoU

			self.savedList.append(epoch)
			self.bestEpoch = epoch

			return True
		
		return False
	
	def _concatSaveDir(self, fileName):
		return os.path.join(self.savedDir,fileName)

	def _concatSaveDirByEpoch(self, epoch):
		return self._concatSaveDir(self._fileFormat(epoch))

	def removeModel(self):

		if len(self.savedList) <= self.capacity:
			return

		delTarget = self.savedList.popleft()
		os.remove(self._concatSaveDirByEpoch(delTarget))
	
	def saveModel(self, epoch, model, optimizer):
		state = {
			'epoch' : epoch,
			'state_dict' : model.state_dict(),
			'optimizer' : optimizer.state_dict()
		}
		torch.save(state, self._concatSaveDirByEpoch(epoch))
  
	def renameBestModel(self):
		os.rename(self._concatSaveDirByEpoch(self.bestEpoch),self._concatSaveDir(f"best_epoch_{self.bestEpoch}.pt"))


	# best IoU 객체 epoch 저장
	def saveBestIoUObject(self, IoU, epoch, object_best, model):
		for num in object_best:
			if IoU[num] > self.bestmIoUObject[num]:
				self.bestmIoUObject[num] = IoU[num]
				self.bestEpochObject[num] = epoch

				state = {
					'state_dict' : model.state_dict(),
				}
				torch.save(state, os.path.join(self.savedDir, f"{self.object_dict[num]}.pt"))

	def renameBestObjectModel(self, object_best):
		for num in object_best:
			old_name = os.path.join(self.savedDir, f"{self.object_dict[num]}.pt")
			new_name = os.path.join(self.savedDir, f"{self.object_dict[num]}_epoch_{self.bestEpochObject[num]}.pt")
			os.rename(old_name, new_name)


		


