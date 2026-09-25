import torch.nn as nn


class Space1Space2FrequencyTimeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # (batch, 9, 65, 18, 18) e-> (batch, 9 * 65* 18 * 18 = 189540)
        self.flatten = nn.Flatten()
        
        self.mlp = nn.Sequential(

            # Camada de Entrada -> Primeira Camada Oculta
            nn.Linear(9 * 65 * 18 * 18, 512), 
            nn.ReLU(),
            # nn.Dropout(0.5), 
            
            # Segunda Camada Oculta
            nn.Linear(512, 128),
            nn.ReLU(),
            # nn.Dropout(0.3),
            
            # Terceira Camada Oculta
            nn.Linear(128, 32),
            nn.ReLU(),
            
            # Camada de Saída (Entrega os 4 logits para a CrossEntropyLoss)
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = self.flatten(x)
        
        logits = self.mlp(x)
        
        return logits