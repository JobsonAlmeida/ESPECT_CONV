import torch.nn as nn

# =========================================================
# MODELS FOR PRONOUNCED SPEECH
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

        self.flatten = nn.Flatten()

        self.linear1 = nn.Sequential(
            nn.Linear(
                9 * 65 * 18 * 18,
                4
            ),
        )

        self.relu1 = nn.ReLU()



    def extract_features(self, x):

        x = self.flatten(x)

        x = self.linear1(x)

        x = self.relu1(x)

        return x


    def forward(self, x):

        x = self.extract_features(x)

        return x