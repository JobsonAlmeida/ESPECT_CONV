import torch.nn as nn

# =========================================================
# MODELS FOR INNER SPEECH
# =========================================================

# ==========================================================
# BRANCH 1
# SPACE1 x SPACE2 x FREQUENCY
# CHANNEL = TIME
# ==========================================================
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
            padding=1
        )
        # (batch, 2, 65, 18, 18)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool3d(
            kernel_size=(2, 2, 2),
            ceil_mode=True
        )
        # (batch, 2, 33, 9, 9)

        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
            nn.Linear(
                2 * 33 * 9 * 9,
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

# ==========================================================
# BRANCH 2
# SPACE1 x SPACE2 x TIME
# CHANNEL = FREQUENCY
# ==========================================================

class Space1Space2TimeFrequencyCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # Entrada:
        # (batch, 65, 9, 18, 18)
        #
        # C = frequency = 65
        # D = time = 9
        # H = space2 = 18
        # W = space1 = 18

        self.conv1 = nn.Conv3d(
            in_channels=65,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # (batch, 2, 9, 18, 18)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool3d(
            kernel_size=(2, 2, 2),
            ceil_mode=True
        )
        # (batch, 2, 5, 9, 9)

        self.flatten = nn.Flatten()
        # 2 x 5 x 9 x 9 

        self.linear = nn.Linear(
            2 * 5 * 9 * 9,
            4
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



# ==========================================================
# BRANCH 3
# TIME x FREQUENCY x SPACE2
# CHANNEL = SPACE1
# ==========================================================

class TimeFrequencySpace2Space1CNN(nn.Module):

    def __init__(self):

        super().__init__()

        # Entrada:
        # (batch, 18, 18, 65, 9)
        #
        # C = space1 = 18
        # D = space2 = 18
        # H = frequency = 65
        # W = time = 9

        self.conv1 = nn.Conv3d(
            in_channels=18,
            out_channels=2,
            kernel_size=3,
            padding=1
        )

        # (batch, 2, 18, 65, 9)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool3d(
            kernel_size=(2, 2, 2),
            ceil_mode=True
        )
        # (batch, 2, 9, 33, 5)

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(
            2 * 9 * 33 * 5,
            4
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



# ==========================================================
# BRANCH 4
# TIME x FREQUENCY x SPACE1
# CHANNEL = SPACE2
# ==========================================================

class TimeFrequencySpace1Space2CNN(nn.Module):

    def __init__(self):

        super().__init__()

        # Entrada:
        # (batch, 18, 18, 65, 9)
        #
        # C = space2 = 18
        # D = space1 = 18
        # H = frequency = 65
        # W = time = 9

        self.conv1 = nn.Conv3d(
            in_channels=18,
            out_channels=2,
            kernel_size=3,
            padding=1
        )

        # (batch, 2, 18, 65, 9)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool3d(
            kernel_size=(2, 2, 2),
            ceil_mode=True
        )
        # (batch, 2, 9, 33, 5)

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(
            2 * 9 * 33 * 5,
            4
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