import torch
from torch import nn
from modules.model_registry import Registry


@Registry.register_model
class VGG16(nn.Module):
    def __init__(self, input_shape, output_shape, kernel_size=3, padding=1, **kwargs):
        
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=64,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            # A Dropout layer is a powerful regularization technique used during training to prevent neural networks from overfitting.
            # During each training step (forward pass), dropout temporarily "turns off" (sets to zero) a random percentage of neurons (e.g., 50% of them).
            # Because neurons are randomly dropped out, individual neurons cannot rely too heavily on neighboring neurons. They are forced to learn robust, 
            # generalized features independently rather than memorizing specific patterns in the training data.
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=output_shape),
        )
    def forward(self, x: torch.Tensor)->torch.Tensor:
        # print(f"IN Featureas: {list(x.shape)}")
        x = self.conv_1(x)
        # print(f"Conv1: {list(x.shape)}")
        x = self.conv_2(x)
        # print(f"Conv2: {list(x.shape)}")
        x = self.conv_3(x)
        # print(f"Conv3: {list(x.shape)}")
        x = self.conv_4(x)
        # print(f"Conv4: {list(x.shape)}")
        x = self.conv_5(x)
        # print(f"Conv5: {list(x.shape)}")
        x = self.classifier(x)
        # print(f"Classifier: {list(x.shape)}")
        return x

if __name__ == "__main__":
    from torchinfo import summary
    torch.manual_seed(42)
    r = torch.rand([1, 3, 224, 224])
    try:
        model = VGG16(input_shape=3, output_shape=3)
        model.eval()
        with torch.inference_mode():
            pred = model(r)
            print(f"Successful\n>> Model : {model.__class__.__name__}\n>> Prediction Shape : {list(pred.shape)}")
        summary(model=model, input_data=r)
    except Exception as e:
        print(f"Failed\n>> Model : {model.__class__.__name__}\n>> Error: {e}")