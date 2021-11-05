from tqdm import tqdm

class TQDM:
	'''
	TQDM용 static method들을 모아놓은 Helper 클래스입니다.
	'''

	@classmethod
	def _makeProcessBar(cls,iterable,desc):
		return tqdm(iterable,desc=desc)

	@classmethod
	def makeMainProcessBar(cls, epoch):
		'''
		epoch당 update되는 pbar
		'''
		return cls._makeProcessBar(range(epoch),desc="Start Training")
	
	@classmethod
	def makePbar(cls, loader ,epoch, isTrain):
		'''
		batch 마다 update되는 pbar
		'''
		if isTrain:
			desc = f"Train #{epoch} "
		else:
			desc = f"Validation #{epoch} "
		
		return cls._makeProcessBar(loader,desc)
	
	@classmethod
	def _setDesctiption(cls, pbar, text):
		pbar.set_description_str(text)

	@classmethod
	def setMainPbarDescInSaved(cls,pbar ,epoch, mIoU):
		cls._setDesctiption(pbar,f"Last save epoch #{epoch}, mIoU: {mIoU:.3f}")
	
	@classmethod
	def _setPostfix(cls, pbar, text):
		pbar.set_postfix_str(text)

	@classmethod
	def setMainPbarPostInValid(cls,pbar,avrgLoss):
		cls._setPostfix(pbar,f"Avrg Loss: {avrgLoss.item():.2f}")
	
	@classmethod
	def setPbarPostInStep(cls, pbar, acc, clsAccMean, loss, mIoU):
		cls._setPostfix(pbar, 
			f"Acc: {acc.item():.2f}, AccCls: {clsAccMean.item():.2f}, Loss: {loss.item():.2f}, mIoU: {mIoU.item():.2f}")


	