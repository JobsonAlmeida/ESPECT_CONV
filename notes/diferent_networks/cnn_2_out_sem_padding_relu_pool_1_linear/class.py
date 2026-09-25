
import torch.nn as nn

class Space1Space2FrequencyTimeCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # Entrada:
        # (batch, 9, 65, 18, 18)
        #
        # C = time = 9
        # D = frequency = 65
        # H = space2 = 18
        # W = space1 = 18

        self.conv1 = nn.Conv3d(
            in_channels=9,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # (batch, 2, 63, 16, 16)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool3d(
            kernel_size=(2, 2, 2),
            ceil_mode=True
        )
        # (batch, 2, 32, 8, 8)

        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
            nn.Linear(
                2 * 32 * 8 * 8,
                4
            ),
        )


    def extract_features(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.flatten(x)

        return x


    def forward(self, x):

        x = self.extract_features(x)

        x = self.linear(x)

        return x