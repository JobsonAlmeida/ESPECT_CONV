
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
            out_channels=18,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # (batch, 18, 65, 18, 18)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool3d(
            kernel_size=(2, 2, 2),
            ceil_mode=True
        )
        # (batch, 18, 33, 9, 9)

        self.conv2 = nn.Conv3d(
            in_channels=18,
            out_channels=36,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # (batch, 36, 33, 9, 9)

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool3d(
            kernel_size=(2, 2, 2),
            ceil_mode=True
        )
        # (batch, 36, 17, 5, 5)

        self.conv3 = nn.Conv3d(
            in_channels=36,
            out_channels=72,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # (batch, 72, 17, 5, 5)

        self.relu3 = nn.ReLU()

        self.pool3 = nn.MaxPool3d(
            kernel_size=(2, 2, 2),
            ceil_mode=True
        )
        # (batch, 72, 9, 3, 3)


        self.conv4 = nn.Conv3d(
            in_channels=72,
            out_channels=144,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # (batch, 144, 9, 3, 3)

        self.relu4 = nn.ReLU()

        self.pool4 = nn.MaxPool3d(
            kernel_size=(2, 2, 2),
            ceil_mode=True
        )
        # (batch, 144, 5, 2, 2)

        self.conv5 = nn.Conv3d(
            in_channels=144,
            out_channels=288,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # (batch, 288, 5, 2, 2)

        self.relu5 = nn.ReLU()

        self.pool5 = nn.MaxPool3d(
            kernel_size=(2, 2, 2),
            ceil_mode=True
        )
        # (batch, 288, 3, 1, 1)

        self.flatten = nn.Flatten()

        self.fully = nn.Sequential(
            nn.Linear(
               288 * 3 * 1 * 1,
                4
            ),
        )        
        #(batch, 4)


    def extract_features(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        x = self.flatten(x)

        return x


    def forward(self, x):

        x = self.extract_features(x)

        x = self.classifier(x)

        return x
