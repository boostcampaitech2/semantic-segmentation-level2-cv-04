- 각종 Custom 인자들 (model, arg, loss, dataloader, optimizer, transform)을 torch 레벨에서 수정할 수 있게 설계해놨습니다.  

1. sample  
   가장 기본적인 구조로 수정하기 좋게 설정해놨습니다.  

2. PAnet + Resnest269e  
   smp 제공 모델중 mIoU 0.715로 괜찮은 점수를 기록한 모델입니다.  

3. PAnet + Swin  
   swinT encoder를 따로 붙여 smp에 이식하여 mIoU 0.745를 기록한 좋은 모델입니다.  
