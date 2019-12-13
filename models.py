import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BaseModel(nn.Module):
    def __init__(self, bn_mom, embed_size, num_classes):
        super(BaseModel, self).__init__()
        backbone = models.vgg11_bn(pretrained=False)
        num_features = backbone.classifier[0].in_features
        self.features = nn.Sequential(
            backbone.features
        )
        self.neck = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, embed_size, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embed_size),
            nn.Linear(embed_size, embed_size, bias=False),
            nn.BatchNorm1d(embed_size)
        )
        self.arc_margin_product = ArcMarginProduct(embed_size, num_classes)
        self.head = nn.Linear(embed_size, num_classes)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_mom

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (7, 7))
        x = x.view(x.size(0), -1)
        embedding = self.neck(x)
        return self.arc_margin_product(embedding), self.head(embedding)


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        target = torch.nn.functional.one_hot(target, num_classes=6)
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.sum()


class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        logits = logits.float()
        cosine = logits
        labels = torch.nn.functional.one_hot(labels, num_classes=6)
        labels = labels.float()
        
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (labels * phi) * ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss / 2


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine
