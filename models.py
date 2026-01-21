import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x) 
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        # 1x1 conv: Giảm số channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 conv: Xử lý không gian
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv: Tăng số channels lên gấp 4 lần (expansion)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.relu = nn.ReLU()
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=6, use_dropout=False):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.use_dropout = use_dropout

        # Lớp đầu vào (Initial Conv)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Các lớp Residual (4 layers)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def inject_imagenet_weights(custom_model, version):
    if version == 18:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif version == 34:
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif version == 50:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif version == 101:
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif version == 152:
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    pretrained_dict = model.state_dict()
    model_dict = custom_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                       if k in model_dict and v.shape == model_dict[k].shape}
    custom_model.load_state_dict(pretrained_dict, strict=False)
    return custom_model


def ResNet18(num_classes):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model = inject_imagenet_weights(model, 18)
    return model


def ResNet34(num_classes):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model = inject_imagenet_weights(model, 34)
    return model


def ResNet50(num_classes):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model = inject_imagenet_weights(model, 50)
    return model


def ResNet101(num_classes):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model = inject_imagenet_weights(model, 101)
    return model


def ResNet152(num_classes):
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model = inject_imagenet_weights(model, 152)
    return model


def load_or_train_model(version, num_classes, device, train_loader=None, val_loader=None, 
                       epochs=20, lr=5.5e-5, fit_func=None):
    model_path = f'resnet{version}_model.pth'
    
    # Nếu model đã tồn tại, load từ file
    if os.path.exists(model_path):
        print(f"Loading pre-trained ResNet{version}...")
        model = torch.load(model_path, map_location=device)
        model.eval()
        return model, None
    
    # Nếu chưa có, train model
    if train_loader is None or val_loader is None or fit_func is None:
        raise ValueError(f"Model file '{model_path}' not found. Please provide train_loader, val_loader, and fit_func to train.")
    
    print(f"Training ResNet{version}... (Model file not found)")
    
    # Khởi tạo model
    if version == 18:
        model = ResNet18(num_classes)
    elif version == 34:
        model = ResNet34(num_classes)
    elif version == 50:
        model = ResNet50(num_classes)
    elif version == 101:
        model = ResNet101(num_classes)
    elif version == 152:
        model = ResNet152(num_classes)
    else:
        raise ValueError(f"Invalid ResNet version: {version}")
    
    model.to(device)
    
    # Train
    history = fit_func(epochs, lr, model, train_loader, val_loader)
    
    # Save model
    torch.save(model, model_path)
    print(f"Model saved to '{model_path}'")
    
    model.eval()
    return model, history