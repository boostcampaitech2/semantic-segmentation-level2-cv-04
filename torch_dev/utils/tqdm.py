from tqdm import tqdm

class TQDM:

	@classmethod
	def _makeProcessBar(cls,iterable,desc):
		return tqdm(iterable,desc=desc)

	@classmethod
	def makeMainProcessBar(cls, epoch):
		return cls._makeProcessBar(range(epoch),desc="Start Training")
	
	@classmethod
	def makePbar(cls, loader ,epoch, isTrain):
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


	