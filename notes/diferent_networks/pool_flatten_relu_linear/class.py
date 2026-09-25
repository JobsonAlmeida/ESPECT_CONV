
import torch.nn as nn

class Space1Space2FrequencyTimeCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Entrada esperada no forward: (batch, 9, 65, 18, 18)
        # Onde:
        # C (canais) = 9 (tempo)
        # D (profundidade) = 65 (frequência)
        # H (altura) = 18 (space2)
        # W (largura) = 18 (space1)

        # Aplicamos o pooling 3D nas dimensões (Freq, Space2, Space1)
        # kernel_size=2 vai reduzir todas essas 3 dimensões pela metade
        self.pool3d = nn.MaxPool3d(
            kernel_size=2,
            ceil_mode=True
        )
        # (batch, 9, 33, 9, 9)

        self.flatten = nn.Flatten()

        self.relu1 = nn.ReLU()
        
        self.linear1 = nn.Linear(9 * 33 * 9 * 9, 4)


    def extract_features(self, x):
        # x entra como (batch, 9, 65, 18, 18)
        
        # 1. Aplica o MaxPool3D antes do flatten
        x = self.pool3d(x)  # Saída: (batch, 9, 33, 9, 9)

        # 2. Achata os dados para a camada linear
        x = self.flatten(x) # Saída: (batch, 24057)

        # 3. Ativação e Linear
        x = self.relu1(x)
        x = self.linear1(x)

        return x


    def forward(self, x):
        x = self.extract_features(x)
        return x
