import torch
import torch.nn as nn

class TinyVGG(nn.Module):
    """Creates the TinyVGG architecture.

    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
    See the original architecture here: https://poloclub.github.io/cnn-explainer/

    Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
    kernel_size: An integer or Tuple indicating the window size
    stride: An integer indivating how many pixel the window will skip
    padding: An integer determining how many pisels to add around the border
    """
    def __init__(self, input_shape, output_shape, hidden_units=224, kernel_size=3, stride=1, padding=0, **kwargs):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*53*53,
                      out_features=output_shape,
                      bias=True)
        )

    def forward(self, x: torch.Tensor)->torch.Tensor:
        # print(f"IN Featureas: {list(x.shape)}")
        x = self.conv_block_1(x)
        # print(f"Conv1: {list(x.shape)}")
        x = self.conv_block_2(x)
        # print(f"Conv2: {list(x.shape)}")
        x = self.classifier(x)
        # print(f"Classifier: {list(x.shape)}")
        return x