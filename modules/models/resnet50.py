import torch
from torch import nn
from modules.model_registry import Registry


class ResidualBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # Squeeze (reduce channel)
        self.squeeze_conv_block =nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ) 

        # main spatial conv,
        self.spatial_conv_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False,
                                            stride=stride, padding=1),
            nn.BatchNorm2d(out_channels)
        ) 

        # expand channels by factor of 4
        self.expansion_conv_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

        self.relu = nn.ReLU(inplace=True) # inplace=True  modifies the input tensor directly in place instead of allocating a completely new output tensor for meamory efficiency
        self.downsample = downsample
        self.stride=stride

    def forward(self, x):
        identity = x

        out = self.relu(self.squeeze_conv_block(x))
        out = self.relu(self.spatial_conv_block(out))
        out = self.expansion_conv_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

@Registry.register_model
class Resnet50(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        self.in_channels = 64

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape, 64, kernel_size=7, 
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer_1 = self._make_layers(out_channels=64, blocks=3, stride=1)
        self.layer_2 = self._make_layers(out_channels=128, blocks=4, stride=2)
        self.layer_3 = self._make_layers(out_channels=256, blocks=6, stride=2)
        self.layer_4 = self._make_layers(out_channels=512, blocks=3, stride=2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512*ResidualBlock.expansion, output_shape)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layers(self, out_channels, blocks, stride):
        downsample = None
        if stride!=1 or self.in_channels != out_channels*ResidualBlock.expansion:
            ds_out = out_channels*ResidualBlock.expansion
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, ds_out, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(ds_out)
            )
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride=stride, downsample=downsample))

        self.in_channels = out_channels * ResidualBlock.expansion

        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers) # *layers unpacks all the layers in the list 

    def forward(self, x):
        x = self.conv(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    from torchinfo import summary
    torch.manual_seed(42)
    r = torch.rand([1, 3, 224, 224])
    try:
        model = Resnet50(input_shape=3, output_shape=3)
        model.eval()
        with torch.inference_mode():
            pred = model(r)
            print(f"Successful\n>> Model : {model.__class__.__name__}\n>> Prediction Shape : {list(pred.shape)}")
        summary(model=model, input_data=r)
    except Exception as e:
        print(f"Failed\n>> Model : {model.__class__.__name__}\n>> Error: {e}")

