import torch.nn as nn


# ==========================================================
# BRANCH 1
# SPACE1 x SPACE2 x FREQUENCY
# CHANNEL = TIME
# ==========================================================


class Space1Space2FrequencyTimeCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # Entrada:
        # (batch, channels, frequency, row, column)
        # (8, 5, 65, 21, 21)
        #
        # channels = 5 janelas de tempo

        self.conv1 = nn.Conv3d(
            in_channels=9,
            out_channels=10,
            kernel_size=3,
            padding=1
        )
        # Saída:
        # (8, 10, 65, 21, 21)

        self.relu1 = nn.ReLU()
        # Saída:
        # (8, 10, 65, 21, 21)

        self.pool1 = nn.MaxPool3d(
            kernel_size=(1,2,2)
        )
        # Saída:
        # (8, 10, 65, 10, 10)


        self.conv2 = nn.Conv3d(
            in_channels=10,
            out_channels=8,
            kernel_size=3,
            padding=1
        )
        # Saída:
        # (8, 8, 65, 10, 10)

        self.relu2 = nn.ReLU()
        # Saída:
        # (8, 8, 65, 10, 10)

        self.pool2 = nn.MaxPool3d(
            kernel_size=(1,2,2)
        )
        # Saída:
        # (8, 8, 65, 5, 5)


        self.flatten = nn.Flatten()
        # Saída:
        # (8, ---)


        self.classifier = nn.Linear(
            8 * 65 * 5 * 5,
            4
        )
        # Saída:
        # (8, 4)


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.classifier(x)

        return x
        

# class Space1Space2FrequencyTimeCNN(nn.Module):

#     def __init__(self):

#         super().__init__()

#         # Entrada:
#         # (batch, 9, 65, 21, 21)
#         #
#         # C = time = 9
#         # D = frequency = 65
#         # H = space2 = 21
#         # W = space1 = 21

#         self.conv1 = nn.Conv3d(
#             in_channels=9,
#             out_channels=10,
#             kernel_size=3,
#             padding=1
#         )
#         # (batch, 10, 65, 21, 21)

#         self.relu1 = nn.ReLU()

#         self.pool1 = nn.MaxPool3d(
#             kernel_size=(2, 2, 2)
#         )
#         # (batch, 10, 32, 10, 10)

#         self.conv2 = nn.Conv3d(
#             in_channels=10,
#             out_channels=8,
#             kernel_size=3,
#             padding=1
#         )
#         # (batch, 8, 32, 10, 10)

#         self.relu2 = nn.ReLU()

#         self.pool2 = nn.MaxPool3d(
#             kernel_size=(2, 2, 2)
#         )
#         # (batch, 8, 16, 5, 5)

#         self.flatten = nn.Flatten()

#         self.classifier = nn.Sequential(
#             nn.Linear(
#                 8 * 16 * 5 * 5,
#                 1800
#             ),
#             nn.Linear(
#                 1800,
#                 4
#             ),

#             nn.ReLU()
#         )


#     def extract_features(self, x):

#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)

#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)

#         x = self.flatten(x)

#         return x


#     def forward(self, x):

#         x = self.extract_features(x)

#         x = self.classifier(x)

#         return x


    
# class Space1Space2FrequencyTimeCNN(nn.Module):

#     def __init__(self):

#         super().__init__()

#         # Entrada:
#         # (batch, 9, 65, 21, 21)
#         #
#         # C = time = 9
#         # D = frequency = 65
#         # H = space2 = 21
#         # W = space1 = 21

#         self.conv1 = nn.Conv3d(
#             in_channels=9,
#             out_channels=6,
#             kernel_size=3,
#             padding=1
#         )

#         # (batch, 6, 65, 21, 21)

#         self.relu1 = nn.ReLU()

#         self.pool1 = nn.MaxPool3d(
#             kernel_size=(2, 2, 2)
#         )

#         # (batch, 6, 32, 10, 10)

#         self.conv2 = nn.Conv3d(
#             in_channels=6,
#             out_channels=4,
#             kernel_size=3,
#             padding=1
#         )

#         # (batch, 4, 32, 10, 10)

#         self.relu2 = nn.ReLU()

#         self.pool2 = nn.MaxPool3d(
#             kernel_size=(2, 2, 2)
#         )

#         # (batch, 4, 16, 5, 5)

#         self.flatten = nn.Flatten()

#         # 4 x 16 x 5 x 5 = 1600

#         self.classifier = nn.Linear(
#             4 * 16 * 5 * 5,
#             4
#         )


#     def extract_features(self, x):

#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)

#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)

#         x = self.flatten(x)

#         # (batch, 1600)

#         return x


#     def forward(self, x):

#         x = self.extract_features(x)

#         x = self.classifier(x)

#         return x



# ==========================================================
# BRANCH 2
# SPACE1 x SPACE2 x TIME
# CHANNEL = FREQUENCY
# ==========================================================

class Space1Space2TimeFrequencyCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # Entrada:
        # (batch, 65, 9, 21, 21)
        #
        # C = frequency = 65
        # D = time = 9
        # H = space2 = 21
        # W = space1 = 21

        self.conv1 = nn.Conv3d(
            in_channels=65,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # (batch, 32, 9, 21, 21)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool3d(
            kernel_size=(2, 2, 2)
        )

        # (batch, 32, 4, 10, 10)

        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # (batch, 16, 4, 10, 10)

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool3d(
            kernel_size=(2, 2, 2)
        )

        # (batch, 16, 2, 5, 5)

        self.flatten = nn.Flatten()

        # 16 x 2 x 5 x 5 = 800

        self.classifier = nn.Linear(
            16 * 2 * 5 * 5,
            4
        )


    def extract_features(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)

        # (batch, 800)

        return x


    def forward(self, x):

        x = self.extract_features(x)

        x = self.classifier(x)

        return x



# ==========================================================
# BRANCH 3
# TIME x FREQUENCY x SPACE2
# CHANNEL = SPACE1
# ==========================================================

class TimeFrequencySpace2Space1CNN(nn.Module):

    def __init__(self):

        super().__init__()

        # Entrada:
        # (batch, 21, 21, 65, 9)
        #
        # C = space1 = 21
        # D = space2 = 21
        # H = frequency = 65
        # W = time = 9

        self.conv1 = nn.Conv3d(
            in_channels=21,
            out_channels=10,
            kernel_size=3,
            padding=1
        )

        # (batch, 10, 21, 65, 9)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool3d(
            kernel_size=(2, 2, 2)
        )

        # (batch, 10, 10, 32, 4)

        self.conv2 = nn.Conv3d(
            in_channels=10,
            out_channels=5,
            kernel_size=3,
            padding=1
        )

        # (batch, 5, 10, 32, 4)

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool3d(
            kernel_size=(2, 2, 2)
        )

        # (batch, 5, 5, 16, 2)

        self.flatten = nn.Flatten()

        # 5 x 5 x 16 x 2 = 800

        self.classifier = nn.Linear(
            5 * 5 * 16 * 2,
            4
        )


    def extract_features(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)

        # (batch, 800)

        return x


    def forward(self, x):

        x = self.extract_features(x)

        x = self.classifier(x)

        return x



# ==========================================================
# BRANCH 4
# TIME x FREQUENCY x SPACE1
# CHANNEL = SPACE2
# ==========================================================

class TimeFrequencySpace1Space2CNN(nn.Module):

    def __init__(self):

        super().__init__()

        # Entrada:
        # (batch, 21, 21, 65, 9)
        #
        # C = space2 = 21
        # D = space1 = 21
        # H = frequency = 65
        # W = time = 9

        self.conv1 = nn.Conv3d(
            in_channels=21,
            out_channels=10,
            kernel_size=3,
            padding=1
        )

        # (batch, 10, 21, 65, 9)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool3d(
            kernel_size=(2, 2, 2)
        )

        # (batch, 10, 10, 32, 4)

        self.conv2 = nn.Conv3d(
            in_channels=10,
            out_channels=5,
            kernel_size=3,
            padding=1
        )

        # (batch, 5, 10, 32, 4)

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool3d(
            kernel_size=(2, 2, 2)
        )

        # (batch, 5, 5, 16, 2)

        self.flatten = nn.Flatten()

        # 5 x 5 x 16 x 2 = 800

        self.classifier = nn.Linear(
            5 * 5 * 16 * 2,
            4
        )


    def extract_features(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)

        # (batch, 800)

        return x


    def forward(self, x):

        x = self.extract_features(x)

        x = self.classifier(x)

        return x