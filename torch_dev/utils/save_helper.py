from collections import deque
import os
import torch

class SaveHelper:
	'''
	모든 모델을 저장할 때는 용량 부담이 크기때문에 저장되는 모델의 수를 설정하는 클래스입니다
	deque 형태로 이름을 가지고있고, 새로운 최고값이 들어왔을 때 큐가 꽉찼다면 가장 낮은 점수를 가진 모델을 제거합니다
	'''

	def __init__(self, capacity, saved_dir) -> None:
		self.savedList = deque()
		self.capacity = max(capacity,2)
		self.bestEpoch = 0
		self.bestIoU = 0
		self.savedDir = saved_dir+"/models"

	@staticmethod
	def _fileFormat(epoch):
		return f"epoch{epoch}.pth"

	def checkBestIoU(self, avrg_iou, epoch):
		
		ok = avrg_iou.item() > self.bestIoU

		if ok:
			self.bestIoU = avrg_iou.item()	
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
	
	def saveModel(self,epoch, model, optimizer, scheduler):

		saveDict = {
			'epoch' : epoch, 
			'model': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
			'scheduler' : scheduler,
		}

		torch.save(saveDict, self._concatSaveDirByEpoch(epoch))
  
