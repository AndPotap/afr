import torch
import torchvision
import types
import timm
from .preresnet import PreResNet
from .model_utils import _replace_fc


def domino_preresnet20(output_dim):
    return _replace_fc(PreResNet(domino=True, depth=20), output_dim)


def cifar_preresnet20(output_dim):
    return _replace_fc(PreResNet(domino=False, depth=20), output_dim)


def _base_resnet18_cifar():
    model = torchvision.models.resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model


def cifar_resnet18(output_dim):
    model = _base_resnet18_cifar()
    return _replace_fc(model, output_dim)


def domino_resnet18(output_dim):
    model = _base_resnet18_cifar()
    model = _replace_fc(model, output_dim)
    return model


def simclr_cifar_resnet18_twolayerhead(output_dim):
    hidden_dim = 2048
    model = _base_resnet18_cifar()
    d = model.fc.in_features
    model.fc = torch.nn.Sequential(torch.nn.Linear(d, hidden_dim), torch.nn.ReLU(),
                                   torch.nn.Linear(hidden_dim, output_dim))
    model.fc.in_features = d

    return model


def imagenet_resnet50(output_dim):
    return _replace_fc(torchvision.models.resnet50(pretrained=False), output_dim)


def imagenet_resnet50_pretrained(output_dim):
    return _replace_fc(torchvision.models.resnet50(pretrained=True), output_dim)


def imagenet_resnet50_timm(output_dim):
    return _replace_fc(timm.create_model('resnet50', pretrained=True), output_dim)


def imagenet_resnet18_pretrained(output_dim):
    return _replace_fc(torchvision.models.resnet18(pretrained=True), output_dim)


def imagenet_resnet34_pretrained(output_dim):
    return _replace_fc(torchvision.models.resnet34(pretrained=True), output_dim)


def imagenet_resnet101_pretrained(output_dim):
    return _replace_fc(torchvision.models.resnet101(pretrained=True), output_dim)


def imagenet_resnet152_pretrained(output_dim):
    return _replace_fc(torchvision.models.resnet152(pretrained=True), output_dim)


def imagenet_wide_resnet50_2_pretrained(output_dim):
    return _replace_fc(torchvision.models.wide_resnet50_2(pretrained=True), output_dim)


def imagenet_resnext50_32x4d_pretrained(output_dim):
    return _replace_fc(torchvision.models.resnext50_32x4d(pretrained=True), output_dim)


def _densenet_replace_fc(model, output_dim):
    model.fc = torch.nn.Identity()
    model.fc.in_features = model.classifier.in_features
    delattr(model, "classifier")

    def classifier(self, x):
        return self.fc(x)

    model.classifier = types.MethodType(classifier, model)
    return _replace_fc(model, output_dim)


def imagenet_densenet121_pretrained(output_dim):
    return _densenet_replace_fc(torchvision.models.densenet121(pretrained=True), output_dim)


def imagenet_densenet121(output_dim):
    return _densenet_replace_fc(torchvision.models.densenet121(pretrained=False), output_dim)
