import torch
from torch import nn

from models import resnet, DenseNet,C3DNet
from models import ViT as vit

def generate_model(opt):
    # import pdb;pdb.set_trace()
    assert opt.model in [
        'resnet', 'c3d', 'wideresnet', 'resnext', 'densenet','vit'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        from models.resnet import get_fine_tuning_parameters

        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
    elif opt.model == 'vit':
        model = vit.ViT(
            num_classes=opt.n_classes,
            channels=opt.channels,
            image_size = opt.sample_size,
            frames = opt.sample_duration,
            image_patch_size=4,#(batchsize,3,16,32,32)
            frame_patch_size=2,#8 sequence
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    elif opt.model == 'densenet':
        model = DenseNet.generate_model(
            num_classes = opt.n_classes,
            channels = opt.channels,
            model_depth = opt.model_depth
        )
    elif opt.model =='c3d':
        model = C3DNet.get_model(
            num_classes = opt.n_classes,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            
        )
    return model, model.parameters()
