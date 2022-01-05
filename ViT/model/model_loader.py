import torchvision.models as models
from torch import nn

def model_loader(model_name='resnet18', num_class=10, pretrain=False):
    assert model_name in ['resnet18', 'resnet34', 'resnet50', 'alexnet', 'vgg16', 'squeezenet1_0', 'densenet161', 'inception_v3', 'googlenet',
                          'mobilenet_v2', 'mobilenet_v3_large', 'mnasnet'], "no model in 'models' module."
    model = getattr(models, model_name)(pretrained=pretrain)
    if not num_class == model.fc.out_features:
        input_feature_fc_layer = model.fc.in_features
        model.fc = nn.Linear(input_feature_fc_layer, num_class, bias=False)
    return model

# resnet18 = models.resnet18()
# alexnet = models.alexnet()
# vgg16 = models.vgg16()
# squeezenet = models.squeezenet1_0()
# densenet = models.densenet161()
# inception = models.inception_v3()
# googlenet = models.googlenet()
# shufflenet = models.shufflenet_v2_x1_0()
# mobilenet_v2 = models.mobilenet_v2()
# mobilenet_v3_large = models.mobilenet_v3_large()
# mobilenet_v3_small = models.mobilenet_v3_small()
# resnext50_32x4d = models.resnext50_32x4d()
# wide_resnet50_2 = models.wide_resnet50_2()
# mnasnet = models.mnasnet1_0()
