import torch.nn as nn
from torchvision import models


def initialize_model(num_classes, use_pretrained=True):
    # 加载预训练的EfficientNet模型
    model = models.efficientnet_b0(pretrained=use_pretrained)

    # 替换分类器
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    return model
