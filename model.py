from mmseg.models import build_segmentor
from config import PRETRAINED_MODEL_PATH

def get_model():
    norm_cfg = dict(type='BN', requires_grad=True)
    
    # Model configuration
    model_cfg = dict(
        type='EncoderDecoder',
        pretrained=None,
        backbone=dict(
            type='MixVisionTransformer',
            in_channels=3,
            embed_dims=64,
            num_stages=4,
            num_layers=[3, 8, 27, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            init_cfg=dict(type="Pretrained", checkpoint=PRETRAINED_MODEL_PATH)
        ),
        decode_head=dict(
            type='SegformerHead',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=256,
            dropout_ratio=0.1,
            num_classes=3,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
            )
        ),
        train_cfg=dict(),
        test_cfg=dict(mode='whole')
    )
    
    model = build_segmentor(model_cfg)
    return model
