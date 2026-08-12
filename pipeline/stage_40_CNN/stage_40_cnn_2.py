import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.model_selection import StratifiedKFold


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

N_FOLDS = 5
N_EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001

RANDOM_STATE = 42
RANDOM_STATE_LOADER = 42

# Inicialização dos pesos e outras operações aleatórias do PyTorch
torch.manual_seed(RANDOM_STATE)


current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[2]

INPUT_DIR = PROJECT_ROOT / "processed_data" / "stage_30_array_assembly"
OUTPUT_DIR = PROJECT_ROOT / "processed_data" / "stage_40_cnn"

# ==========================================================
# 1. CARREGAR OS DADOS
# ==========================================================

power_array = np.load(
    INPUT_DIR / "sub-01_ses-01_power.npy"
)

labels = np.load(
    INPUT_DIR / "sub-01_ses-01_labels.npy"
)

print("Power:", power_array.shape)
print("Labels:", labels.shape)


# ==========================================================
# 2. REORGANIZAR AS DIMENSÕES
# ==========================================================

# Antes:
#
# (epoch, row, column, frequency, time)
#
# Depois:
#
# (epoch, time, frequency, row, column)

power_array = np.transpose(
    power_array,
    (0, 4, 3, 1, 2)
)

print("Power reorganizado:", power_array.shape)


# ==========================================================
# 3. CONVERTER PARA TENSOR
# ==========================================================

X = torch.from_numpy(
    power_array
).float()

y = torch.from_numpy(
    labels
).long()

print("X:", X.shape)
print("y:", y.shape)


# ==========================================================
# 4. DEFINIR A REDE
# ==========================================================

class SpaceFrequencyCNN(nn.Module):

    def __init__(self):

        super().__init__()

        #entrada: (epoch, time, frequency, row, column) = (8, 9, 65, 21, 21)

        self.conv1 = nn.Conv3d(  
            in_channels=9,
            out_channels=16,
            kernel_size=3,
            padding=1
        ) # saída: (epoch, time, frequency, row, column) = (8, 16, 65, 21, 21)

        self.relu1 = nn.ReLU() # saída: (epoch, time, frequency, row, column) = (8, 16, 65, 21, 21)

        self.pool1 = nn.MaxPool3d(
            kernel_size=2
        ) # saída: (epoch, time, frequency, row, column) = (8, 16, 32, 10, 10)


        self.conv2 = nn.Conv3d(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            padding=1
        )# saída: (epoch, time, frequency, row, column) = (8, 8, 32, 10, 10)

        self.relu2 = nn.ReLU() # saída: (epoch, time, frequency, row, column) = (8, 8, 32, 10, 10)

        self.pool2 = nn.MaxPool3d(
            kernel_size=2
        ) # saída: (epoch, time, frequency, row, column) = (8, 8, 16, 5, 5)


        self.flatten = nn.Flatten() # saída:  (8, 8x16x5x5) = (8, 3200)


        self.latent = nn.Linear(
            8 * 16 * 5 * 5,
            3
        )


        self.classifier = nn.Linear(
            3,
            4
        )


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.latent(x)

        x = self.classifier(x)

        return x


# ==========================================================
# 5. CPU OU GPU
# ==========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)
print(torch.cuda.is_available())      # Deve retornar False no seu caso
print(torch.version.cuda)             # Mostra qual versão do CUDA o PyTorch espera (se retornar None, você instalou a versão errada)
print(torch.cuda.get_device_name(0))  # Tentará forçar o nome da sua placa (provavelmente vai dar erro se o de cima for False)


# ==========================================================
# 6. CRIAR OS 5 FOLDS
# ==========================================================

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ==========================================================
# 7. ARMAZENAR RESULTADO DOS FOLDS
# ==========================================================

fold_accuracies = []


# ==========================================================
# 8. LOOP DOS 5 FOLDS
# ==========================================================

for fold, (train_indices, test_indices) in enumerate(
    skf.split(X, labels),
    start=1
):

    print()
    print("=" * 60)
    print(f"FOLD {fold}/{N_FOLDS}")
    print("=" * 60)


    # ======================================================
    # 8.1 SEPARAR TREINO E TESTE
    # ======================================================

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]


    print("Treino:", X_train.shape)
    print("Teste:", X_test.shape)


    # ======================================================
    # 8.2 CRIAR DATASETS
    # ======================================================

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    test_dataset = TensorDataset(
        X_test,
        y_test
    )


    # ======================================================
    # 8.3 CRIAR DATALOADERS
    # ======================================================

    generator = torch.Generator()
    generator.manual_seed(RANDOM_STATE_LOADER)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # ======================================================
    # 8.4 CRIAR UMA NOVA REDE PARA ESTE FOLD
    # ======================================================

    model = SpaceFrequencyCNN().to(device)


    # ======================================================
    # 8.5 FUNÇÃO DE PERDA
    # ======================================================

    criterion = nn.CrossEntropyLoss()


    # ======================================================
    # 8.6 OTIMIZADOR
    # ======================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ======================================================
    # 8.7 TREINAMENTO
    # ======================================================

    for epoch in range(N_EPOCHS):

        model.train()

        total_loss = 0.0

        correct = 0
        total = 0


        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)


            # ----------------------------------------------
            # Zerar gradientes
            # ----------------------------------------------

            optimizer.zero_grad()


            # ----------------------------------------------
            # Forward
            # ----------------------------------------------

            outputs = model(
                X_batch
            )


            # ----------------------------------------------
            # Loss
            # ----------------------------------------------

            loss = criterion(
                outputs,
                y_batch
            )


            # ----------------------------------------------
            # Backpropagation
            # ----------------------------------------------

            loss.backward()


            # ----------------------------------------------
            # Atualizar pesos
            # ----------------------------------------------

            optimizer.step()


            # ----------------------------------------------
            # Acumular loss
            # ----------------------------------------------

            total_loss += loss.item()


            # ----------------------------------------------
            # Classes previstas
            # ----------------------------------------------

            predictions = outputs.argmax(
                dim=1
            )


            # ----------------------------------------------
            # Número de acertos
            # ----------------------------------------------

            correct += (
                predictions == y_batch
            ).sum().item()

            total += y_batch.size(0)


        # ==================================================
        # RESULTADOS DA ÉPOCA
        # ==================================================

        train_loss = (
            total_loss
            / len(train_loader)
        )

        train_accuracy = (
            correct
            / total
        )


        print(
            f"Fold {fold} | "
            f"Epoch {epoch + 1:02d}/{N_EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f}"
        )


    # ======================================================
    # 9. TESTAR O FOLD
    # ======================================================

    model.eval()

    correct = 0
    total = 0


    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)


            outputs = model(
                X_batch
            )


            predictions = outputs.argmax(
                dim=1
            )


            correct += (
                predictions == y_batch
            ).sum().item()

            total += y_batch.size(0)


    # ======================================================
    # 10. ACURÁCIA DO FOLD
    # ======================================================

    fold_accuracy = (
        correct
        / total
    )

    fold_accuracies.append(
        fold_accuracy
    )


    print()
    print(
        f"Acurácia Fold {fold}: "
        f"{fold_accuracy:.4f}"
    )


# ==========================================================
# 11. RESULTADOS DOS 5 FOLDS
# ==========================================================

fold_accuracies = np.array(
    fold_accuracies
)


print()
print("=" * 60)
print("RESULTADO FINAL")
print("=" * 60)

for fold, accuracy in enumerate(
    fold_accuracies,
    start=1
):

    print(
        f"Fold {fold}: "
        f"{accuracy:.4f}"
    )


print()

print(
    "Acurácia média:",
    fold_accuracies.mean()
)

print(
    "Desvio padrão:",
    fold_accuracies.std()
)