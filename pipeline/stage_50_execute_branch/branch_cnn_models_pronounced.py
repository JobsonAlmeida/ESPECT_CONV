import torch.nn as nn

# =========================================================
# MODELS FOR PRONOUNCED SPEECH
# =========================================================

# ==========================================================
# BRANCH 1
# SPACE1 x SPACE2 x FREQUENCY
# CHANNEL = TIME
# ==========================================================



# class Space1Space2FrequencyTimeCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # (batch, 9, 65, 18, 18) e-> (batch, 9 * 65* 18 * 18 = 189540)
#         self.flatten = nn.Flatten()
        
#         self.mlp = nn.Sequential(

#             # Camada de Entrada -> Primeira Camada Oculta
#             nn.Linear(9 * 65 * 18 * 18, 512), 
#             nn.ReLU(),
#             # nn.Dropout(0.5), 
            
#             # Segunda Camada Oculta
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             # nn.Dropout(0.3),
            
#             # Terceira Camada Oculta
#             nn.Linear(128, 32),
#             nn.ReLU(),
            
#             # Camada de Saída (Entrega os 4 logits para a CrossEntropyLoss)
#             nn.Linear(32, 4)
#         )

#     def forward(self, x):
#         x = self.flatten(x)
        
#         logits = self.mlp(x)
        
#         return logits



import torch
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

        # self.conv1 = nn.Conv3d(
        #     in_channels=9,
        #     out_channels=1,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1
        # )
        # # (batch, 1, 65, 18, 18)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool3d(
            kernel_size=(2, 2, 2),
            ceil_mode=True
        )
        # Saída do pooling: (batch, 9, 33, 9, 9)

        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
            nn.Linear(
                9 * 33 * 9 * 9, # 23571 neurônios de entrada
                4
            ),
        )


    def extract_features(self, x):

        x = self.relu1(x)
        x = self.pool1(x)
        x = self.flatten(x)

        return x


    def forward(self, x):

        x = self.extract_features(x)

        x = self.linear(x)

        return x


# class Space1Space2FrequencyTimeCNN(nn.Module):

#     def __init__(self):

#         super().__init__()

#         # Entrada:
#         # (batch, 9, 65, 18, 18)
#         #
#         # C = time = 9
#         # D = frequency = 65
#         # H = space2 = 18
#         # W = space1 = 18

#         # self.conv1 = nn.Conv3d(
#         #     in_channels=9,
#         #     out_channels=1,
#         #     kernel_size=3,
#         #     stride=1,
#         #     padding=1
#         # )
#         # # (batch, 2, 65, 18, 18)

#         self.relu1 = nn.ReLU()

#         self.pool1 = nn.MaxPool3d(
#             kernel_size=(2, 2, 2),
#             ceil_mode=True
#         )
#         # (batch, 2, 33, 9, 9)

#         self.flatten = nn.Flatten()

#         self.linear = nn.Sequential(
#             nn.Linear(
#                 9 * 65 * 18 * 18,
#                 4
#             ),
#         )


#     def extract_features(self, x):

#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)

#         x = self.flatten(x)

#         return x


#     def forward(self, x):

#         x = self.extract_features(x)

#         x = self.linear(x)

#         return x

# class Space1Space2FrequencyTimeCNN(nn.Module):

#     def __init__(self):

#         super().__init__()

#         # Entrada:
#         # (batch, 9, 65, 18, 18)
#         #
#         # C = time = 9
#         # D = frequency = 65
#         # H = space2 = 18
#         # W = space1 = 18

#         self.conv1 = nn.Conv3d(
#             in_channels=9,
#             out_channels=4,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         )
#         # (batch, 8, 65, 18, 18)

#         self.relu1 = nn.ReLU()

#         self.pool1 = nn.MaxPool3d(
#             kernel_size=(2, 2, 2),
#             ceil_mode=True
#         )
#         # (batch, 4, 33, 9, 9)

#         # self.conv2 = nn.Conv3d(
#         #     in_channels=16,
#         #     out_channels=32,
#         #     kernel_size=3,
#         #     stride=1,
#         #     padding=1
#         # )
#         # # (batch, 32, 33, 9, 9)

#         # self.relu2 = nn.ReLU()

#         # self.pool2 = nn.MaxPool3d(
#         #     kernel_size=(2, 2, 2),
#         #     ceil_mode=True
#         # )
#         # # (batch, 32, 17, 5, 5)

#         # self.conv3 = nn.Conv3d(
#         #     in_channels=32,
#         #     out_channels=64,
#         #     kernel_size=3,
#         #     stride=1,
#         #     padding=1
#         # )
#         # # (batch, 64, 17, 5, 5)

#         # self.relu3 = nn.ReLU()

#         # self.pool3 = nn.MaxPool3d(
#         #     kernel_size=(2, 2, 2),
#         #     ceil_mode=True
#         # )
#         # # (batch, 64, 9, 3, 3)


#         # self.conv4 = nn.Conv3d(
#         #     in_channels=64,
#         #     out_channels=128,
#         #     kernel_size=3,
#         #     stride=1,
#         #     padding=1
#         # )
#         # # (batch, 128, 9, 3, 3)

#         # self.relu4 = nn.ReLU()

#         # self.pool4 = nn.MaxPool3d(
#         #     kernel_size=(2, 2, 2),
#         #     ceil_mode=True
#         # )
#         # # (batch, 128, 5, 2, 2)

#         # self.conv5 = nn.Conv3d(
#         #     in_channels=128,
#         #     out_channels=256,
#         #     kernel_size=3,
#         #     stride=1,
#         #     padding=1
#         # )
#         # # (batch, 256, 5, 2, 2)

#         # self.relu5 = nn.ReLU()

#         # self.pool5 = nn.MaxPool3d(
#         #     kernel_size=(2, 2, 2),
#         #     ceil_mode=True
#         # )
#         # # (batch, 256, 3, 1, 1)

#         self.flatten = nn.Flatten()

#         self.classifier = nn.Sequential(
#             nn.Linear(
#                4 * 33 * 9 * 9,
#                 4
#             ),
#         )

#         # self.softmax_classifier = nn.Softmax(
#         #     16 * 33 * 9 * 9,
#         #     4
#         # )

        
#         #(batch, 8, 4)


#     def extract_features(self, x):

#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)

#         # x = self.conv2(x)
#         # x = self.relu2(x)
#         # x = self.pool2(x)

#         # x = self.conv3(x)
#         # x = self.relu3(x)
#         # x = self.pool3(x)

#         # x = self.conv4(x)
#         # x = self.relu4(x)
#         # x = self.pool4(x)

#         # x = self.conv5(x)
#         # x = self.relu5(x)
#         # x = self.pool5(x)

#         x = self.flatten(x)

#         return x


#     def forward(self, x):

#         x = self.extract_features(x)

#         x = self.classifier(x)

#         return x



# class Space1Space2FrequencyTimeCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # 1. Primeira camada convolucional (extrai 32 características)
#         self.conv1 = nn.Conv3d(9, 32, kernel_size=3, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), ceil_mode=True) # Saída: (batch, 32, 33, 9, 9)
        
#         # 2. Nova camada convolucional para reduzir os canais de 32 para 4 (número de classes)
#         self.conv_classes = nn.Conv3d(32, 4, kernel_size=1) # Convolução 1x1 apenas mapeia os canais
        
#         # 3. Global Average Pooling (tira a média de todo o espaço restante)
#         # Como o que sobou foi 33x9x9, pedimos para o PyTorch reduzir tudo para 1x1x1
#         self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        
#         # 4. Remove as dimensões que viraram 1 para entregar os 4 logits limpos
#         self.flatten = nn.Flatten() 

#     def forward(self, x):
#         x = self.pool1(self.relu1(self.conv1(x)))
#         x = self.conv_classes(x)
#         x = self.gap(x)
#         x = self.flatten(x) # Saída direta: (batch, 4) - Pronta para a CrossEntropyLoss
#         return x
    
# class Space1Space2FrequencyTimeCNN(nn.Module):

#     def __init__(self):

#         super().__init__()

#         # Entrada:
#         # (batch, 9, 65, 18, 18)
#         #
#         # C = time = 9
#         # D = frequency = 65
#         # H = space2 = 18
#         # W = space1 = 18

#         self.conv1 = nn.Conv3d(
#             in_channels=9,
#             out_channels=32,
#             kernel_size=3,
#             stride=1,
#             padding=1
#         )
#         # (batch, 32, 65, 18, 18)

#         self.relu1 = nn.ReLU()

#         self.pool1 = nn.MaxPool3d(
#             kernel_size=(2, 2, 2),
#             ceil_mode=True
#         )
#         # (batch, 64, 33, 9, 9)

#         # self.conv2 = nn.Conv3d(
#         #     in_channels=16,
#         #     out_channels=32,
#         #     kernel_size=3,
#         #     stride=1,
#         #     padding=1
#         # )
#         # # (batch, 32, 33, 9, 9)

#         # self.relu2 = nn.ReLU()

#         # self.pool2 = nn.MaxPool3d(
#         #     kernel_size=(2, 2, 2),
#         #     ceil_mode=True
#         # )
#         # # (batch, 32, 17, 5, 5)

#         # self.conv3 = nn.Conv3d(
#         #     in_channels=32,
#         #     out_channels=64,
#         #     kernel_size=3,
#         #     stride=1,
#         #     padding=1
#         # )
#         # # (batch, 64, 17, 5, 5)

#         # self.relu3 = nn.ReLU()

#         # self.pool3 = nn.MaxPool3d(
#         #     kernel_size=(2, 2, 2),
#         #     ceil_mode=True
#         # )
#         # # (batch, 64, 9, 3, 3)


#         # self.conv4 = nn.Conv3d(
#         #     in_channels=64,
#         #     out_channels=128,
#         #     kernel_size=3,
#         #     stride=1,
#         #     padding=1
#         # )
#         # # (batch, 128, 9, 3, 3)

#         # self.relu4 = nn.ReLU()

#         # self.pool4 = nn.MaxPool3d(
#         #     kernel_size=(2, 2, 2),
#         #     ceil_mode=True
#         # )
#         # # (batch, 128, 5, 2, 2)

#         # self.conv5 = nn.Conv3d(
#         #     in_channels=128,
#         #     out_channels=256,
#         #     kernel_size=3,
#         #     stride=1,
#         #     padding=1
#         # )
#         # # (batch, 256, 5, 2, 2)

#         # self.relu5 = nn.ReLU()

#         # self.pool5 = nn.MaxPool3d(
#         #     kernel_size=(2, 2, 2),
#         #     ceil_mode=True
#         # )
#         # # (batch, 256, 3, 1, 1)

#         self.flatten = nn.Flatten()

#         self.classifier = nn.Sequential(
#             nn.Linear(
#                32 * 33 * 9 * 9,
#                 4
#             ),
#         )

#         # self.softmax_classifier = nn.Softmax(
#         #     16 * 33 * 9 * 9,
#         #     4
#         # )

        
#         #(batch, 8, 4)


#     def extract_features(self, x):

#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)

#         # x = self.conv2(x)
#         # x = self.relu2(x)
#         # x = self.pool2(x)

#         # x = self.conv3(x)
#         # x = self.relu3(x)
#         # x = self.pool3(x)

#         # x = self.conv4(x)
#         # x = self.relu4(x)
#         # x = self.pool4(x)

#         # x = self.conv5(x)
#         # x = self.relu5(x)
#         # x = self.pool5(x)

#         x = self.flatten(x)

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