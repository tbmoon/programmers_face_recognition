import torch
import torch.nn as nn
import torchvision.models as models


class BaseModel(nn.Module):
    
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        model = models.resnet34(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        self.model = model

    def forward(self, image):
        return self.model(image)
