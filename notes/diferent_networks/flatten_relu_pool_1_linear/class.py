import torch
import torch.nn as nn

class Space1Space2FrequencyTimeLinear(nn.Module):

    def __init__(self):
        super().__init__()

        # Entrada esperada:
        # (batch, 9, 65, 18, 18)
        #
        # C = time = 9
        # D = frequency = 65
        # H = space2 = 18
        # W = space1 = 18

        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
            nn.Linear(
                9 * 65 * 18 * 18, 
                4                
            ),
        )

    def extract_features(self, x):

        x = self.flatten(x)
        return x
    

    def forward(self, x):

        x = self.extract_features(x)
        
        x = self.linear(x)
        return x
