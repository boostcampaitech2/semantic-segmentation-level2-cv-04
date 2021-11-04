# https://smp.readthedocs.io/en/latest/index.html
# https://smp.readthedocs.io/en/latest/encoders_timm.html
# timm encoder 쓸땐 encoder name에 tu- 붙이기

# code based on https://smp.readthedocs.io/en/latest/insights.html#

import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders._base import EncoderMixin
from .src.swin import SwinTransformer
from typing import List

def getModel():
	
	register_encoder()

	model = smp.PAN(
			encoder_name="swin_encoder",
			encoder_weights="imagenet",
            encoder_output_stride=32,
			in_channels=3,
			classes=11
		)
	
	return model

# Custom SwinEncoder 정의
class SwinEncoder(torch.nn.Module, EncoderMixin):

    def __init__(self, **kwargs):
        super().__init__()

        # A number of channels for each encoder feature tensor, list of integers
        self._out_channels: List[int] = [128, 256, 512, 1024]

        # A number of stages in decoder (in other words number of downsampling operations), integer
        # use in in forward pass to reduce number of returning features
        self._depth: int = 3

        # Default number of input channels in first Conv2d layer for encoder (usually 3)
        self._in_channels: int = 3
        kwargs.pop('depth')

        self.model = SwinTransformer(**kwargs)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = self.model(x)
        return list(outs)

    def load_state_dict(self, state_dict, **kwargs):
        self.model.load_state_dict(state_dict['model'], strict=False, **kwargs)

# Swin을 smp의 encoder로 사용할 수 있게 등록
def register_encoder():
    smp.encoders.encoders["swin_encoder"] = {
    "encoder": SwinEncoder, # encoder class here
    "pretrained_settings": { # pretrained 값 설정
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth",
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": { # 기본 파라미터
        "pretrain_img_size": 384,
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32],
        "window_size": 12,
        "drop_path_rate": 0.3,
    }
}