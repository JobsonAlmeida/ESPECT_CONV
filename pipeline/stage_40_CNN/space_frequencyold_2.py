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
N_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 0.001

RANDOM_STATE = 42
RANDOM_STATE_LOADER = 42

current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_30_array_assembly"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_40_cnn"
)

# ==========================================================
# DEFINIR A REDE
# ==========================================================

class SpaceFrequencyCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=9,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool3d(
            kernel_size=(2, 2, 2)
        )

        self.conv2 = nn.Conv3d(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool3d(
            kernel_size=(2, 2, 2)
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Linear(
            8 * 16 * 5 * 5,
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

        return x


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


# ==========================================================
# MAIN
# ==========================================================

def main():

    # Inicialização das operações aleatórias do PyTorch
    torch.manual_seed(RANDOM_STATE)

    subjects = [f"sub-{i:02d}" for i in range(3, 4)]
    sessions = [f"ses-{i:02d}" for i in range(1, 4)]

    for subject in subjects: 

        # ==========================================================
        # CARREGAR OS DADOS
        # ==========================================================

        power_array = np.load(
            INPUT_DIR/
            f"{subject}_power.npy")

        labels = np.load(
            INPUT_DIR / 
            f"{subject}_labels.npy")

        print("Power:", power_array.shape)
        print("Labels:", labels.shape)

        # ==========================================================
        # DIRETÓRIO DE SALVAMENTO DOS MODELOS
        # ==========================================================

        MODEL_DIR = (
            OUTPUT_DIR
            / "space_frequency_models"
            / subject
        )

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        # ==========================================================
        # REORGANIZAR AS DIMENSÕES 
        # ==========================================================

        # Antes:        
        # (epoch, row, column, frequency, time)
        #
        # Depois:        
        # (epoch, time, frequency, row, column)

        power_array = np.transpose(
            power_array,
            (0, 4, 3, 1, 2)
        )

        print(
            "Power reorganizado:",
            power_array.shape
        )


        # ==========================================================
        # 3. CONVERTER PARA TENSOR
        # ==========================================================

        X = torch.from_numpy(
            power_array
        ).float()

        y = torch.from_numpy(
            labels
        ).long()

        print(
            "X:",
            X.shape
        )

        print(
            "y:",
            y.shape
        )

        print(
            "Min:",
            X.min()
        )

        print(
            "Max:",
            X.max()
        )

        print(
            "Mean:",
            X.mean()
        )

        print(
            "Std:",
            X.std()
        )

        print(
            "Valores diferentes de zero:",
            torch.count_nonzero(X).item()
        )

        print(
            "Proporção diferente de zero:",
            torch.count_nonzero(X).item()
            / X.numel()
        )


        # ==========================================================
        # 4. CPU OU GPU
        # ==========================================================

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "Device:",
            device
        )

        print(
            "CUDA disponível:",
            torch.cuda.is_available()
        )

        print(
            "Versão CUDA:",
            torch.version.cuda
        )

        if torch.cuda.is_available():

            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )


        # ==========================================================
        # 5. CRIAR OS 5 FOLDS
        # ==========================================================

        skf = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE
        )


        FOLDS_DIR = (
            OUTPUT_DIR
            / "folds"
        )

        FOLDS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        # ==========================================================
        # 6. ARMAZENAR RESULTADOS DOS FOLDS
        # ==========================================================

        fold_accuracies = []

        # ==========================================================
        # 7. LOOP DOS 5 FOLDS
        # ==========================================================

        for fold, (train_indices, test_indices) in enumerate(skf.split(X,labels), start=1):


            print()
            print("=" * 60)
            print(f"FOLD {fold}/{N_FOLDS}")
            print("=" * 60)

            # ======================================================
            # 7.0 SALVAR OS ÍNDICES DE TREINO E TESTE
            # ======================================================

            np.savez(
                FOLDS_DIR
                / f"fold_{fold}.npz",

                train_indices=train_indices,
                test_indices=test_indices
            )


            # ======================================================
            # 7.1 SEPARAR TREINO E TESTE
            # ======================================================

            X_train = X[train_indices]
            y_train = y[train_indices]

            X_test = X[test_indices]
            y_test = y[test_indices]

            # ======================================================
            # NORMALIZAÇÃO
            # ======================================================

            # power_max é calculado somente
            # com os dados de treinamento

            power_max = X_train.max()

            X_train = (X_train / power_max)
            X_test = (X_test / power_max)


            print("Treino:", X_train.shape)

            print("Teste:", X_test.shape)

            # ======================================================
            # 7.2 CRIAR DATASETS
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
            # 7.3 CRIAR DATALOADERS
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
            # CRIAR UMA NOVA REDE PARA ESTE FOLD
            # ======================================================

            # Todos os folds começam com
            # a mesma inicialização dos pesos

            torch.manual_seed(RANDOM_STATE)
            model = SpaceFrequencyCNN().to(device)

            # ======================================================
            # 7.5 FUNÇÃO DE PERDA
            # ======================================================

            criterion = nn.CrossEntropyLoss()


            # ======================================================
            # 7.6 OTIMIZADOR
            # ======================================================

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LEARNING_RATE
            )


            # ======================================================
            # HISTÓRICO DESTE FOLD
            # ======================================================

            train_accuracies_epoch = []

            test_accuracies_epoch = []

            train_losses_epoch = []

            test_losses_epoch = []


            # ======================================================
            # 7.7 TREINAMENTO
            # ======================================================

            for epoch in range(N_EPOCHS):

                # ==================================================
                # TREINAMENTO DA ÉPOCA
                # ==================================================

                model.train()

                total_loss = 0.0
                correct_train = 0
                total = 0

                for (X_batch, y_batch) in train_loader:

                    X_batch = X_batch.to(device)

                    y_batch = y_batch.to(device)

                    # ----------------------------------------------
                    # ZERAR GRADIENTES
                    # ----------------------------------------------

                    optimizer.zero_grad()


                    # ----------------------------------------------
                    # FORWARD
                    # ----------------------------------------------

                    outputs = model(X_batch)


                    # ----------------------------------------------
                    # LOSS
                    # ----------------------------------------------

                    loss = criterion(
                        outputs,
                        y_batch
                    )


                    # ----------------------------------------------
                    # BACKPROPAGATION
                    # ----------------------------------------------

                    loss.backward()


                    # ----------------------------------------------
                    # ATUALIZAR PESOS
                    # ----------------------------------------------

                    optimizer.step()


                    # ----------------------------------------------
                    # ACUMULAR LOSS
                    # ----------------------------------------------

                    total_loss += (
                        loss.item()
                    )


                    # ----------------------------------------------
                    # CLASSES PREVISTAS
                    # ----------------------------------------------

                    predictions = outputs.argmax(
                        dim=1
                    )


                    # ----------------------------------------------
                    # NÚMERO DE ACERTOS
                    # ----------------------------------------------

                    correct_train += (
                        predictions
                        == y_batch
                    ).sum().item()


                    total += (
                        y_batch.size(0)
                    )


                # ==================================================
                # RESULTADOS DE TREINAMENTO DA ÉPOCA
                # ==================================================

                train_loss = (total_loss/ len(train_loader))

                train_accuracy = (correct_train/ total)


                # ==================================================
                # TESTE DA ÉPOCA
                # ==================================================

                model.eval()

                test_total_loss = 0.0
                test_correct = 0
                test_total = 0

                with torch.no_grad():

                    for (X_batch,y_batch) in test_loader:


                        X_batch = X_batch.to(device)

                        y_batch = y_batch.to(device)


                        # ------------------------------------------
                        # FORWARD
                        # ------------------------------------------

                        outputs = model(X_batch)

                        # ------------------------------------------
                        # LOSS
                        # ------------------------------------------

                        loss = criterion(
                            outputs,
                            y_batch
                        )


                        test_total_loss += (
                            loss.item()
                        )

                        # ------------------------------------------
                        # PREVISÕES
                        # ------------------------------------------

                        predictions = outputs.argmax(
                            dim=1
                        )


                        # ------------------------------------------
                        # ACERTOS
                        # ------------------------------------------

                        test_correct += (predictions == y_batch).sum().item()


                        test_total += (y_batch.size(0))


                # ==================================================
                # RESULTADOS DE TESTE DA ÉPOCA
                # ==================================================

                test_loss = (test_total_loss/ len(test_loader))


                test_accuracy = (test_correct/ test_total)


                # ==================================================
                # GUARDAR HISTÓRICO
                # ==================================================

                train_accuracies_epoch.append(
                    train_accuracy
                )


                test_accuracies_epoch.append(
                    test_accuracy
                )


                train_losses_epoch.append(
                    train_loss
                )


                test_losses_epoch.append(
                    test_loss
                )


                # ==================================================
                # MOSTRAR RESULTADOS DA ÉPOCA
                # ==================================================

                print(
                    f"Fold {fold} | "
                    f"Epoch {epoch + 1:03d}/{N_EPOCHS} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Test Loss: {test_loss:.4f} | "
                    f"Train Acc: {train_accuracy:.4f} | "
                    f"Test Acc: {test_accuracy:.4f}"
                )


            # ======================================================
            # 8. TESTE FINAL DO FOLD
            # ======================================================

            model.eval()

            test_total_loss = 0.0
            test_correct = 0
            test_total = 0


            # Guardar todas as previsões
            # e labels do fold

            all_predictions = []

            all_labels = []


            with torch.no_grad():


                for (X_batch, y_batch) in test_loader:


                    X_batch = X_batch.to(device)

                    y_batch = y_batch.to(device)


                    outputs = model(X_batch)

                    loss = criterion(outputs,y_batch)

                    test_total_loss += (loss.item())

                    predictions = outputs.argmax(dim=1)

                    test_correct += (predictions == y_batch).sum().item()

                    test_total += (y_batch.size(0))


                    # ----------------------------------------------
                    # GUARDAR PREVISÕES
                    # ----------------------------------------------

                    all_predictions.extend(
                        predictions
                        .cpu()
                        .numpy()
                    )


                    # ----------------------------------------------
                    # GUARDAR LABELS VERDADEIRAS
                    # ----------------------------------------------

                    all_labels.extend(
                        y_batch
                        .cpu()
                        .numpy()
                    )


            # ======================================================
            # RESULTADOS FINAIS DO TESTE
            # ======================================================

            test_loss = (test_total_loss / len(test_loader))


            test_accuracy = (test_correct / test_total)


            # ======================================================
            # VER DISTRIBUIÇÃO DAS PREVISÕES
            # ======================================================

            print()

            print(
                "--- Resultados do Teste ---"
            )


            print(
                "Classes previstas:",
                np.bincount(
                    all_predictions,
                    minlength=4
                )
            )


            print(
                "Classes reais:",
                np.bincount(
                    all_labels,
                    minlength=4
                )
            )


            # ======================================================
            # 9. ACURÁCIA DO FOLD
            # ======================================================

            fold_accuracy = (
                test_accuracy
            )


            fold_accuracies.append(
                fold_accuracy
            )


            print()

            print(
                f"Acurácia Fold {fold}: "
                f"{fold_accuracy:.4f}"
            )


            # ======================================================
            # 9.1 VERIFICAR TAMANHO DO HISTÓRICO
            # ======================================================

            print(
                "Épocas armazenadas:", len(train_accuracies_epoch)
            )

            # ======================================================
            # 9.2 SALVAR O MODELO DO FOLD
            # ======================================================

            torch.save(
                {
                    # ----------------------------------------------
                    # MODELO
                    # ----------------------------------------------

                    "model_state_dict": model.state_dict(),

                    # ----------------------------------------------
                    # NORMALIZAÇÃO
                    # ----------------------------------------------

                    "power_max": power_max.item(),

                    # ----------------------------------------------
                    # FOLD
                    # ----------------------------------------------

                    "fold": fold,

                    # ----------------------------------------------
                    # ÍNDICES
                    # ----------------------------------------------

                    "train_indices": train_indices,

                    "test_indices": test_indices,

                    # ----------------------------------------------
                    # HISTÓRICO DAS ACURÁCIAS
                    # ----------------------------------------------

                    "train_accuracies_epoch": train_accuracies_epoch,

                    "test_accuracies_epoch": test_accuracies_epoch,

                    # ----------------------------------------------
                    # HISTÓRICO DAS LOSSES
                    # ----------------------------------------------

                    "train_losses_epoch": train_losses_epoch,

                    "test_losses_epoch": test_losses_epoch,

                    # ----------------------------------------------
                    # RESULTADO FINAL DO FOLD
                    # ----------------------------------------------

                    "fold_accuracy": fold_accuracy
                },

                MODEL_DIR
                / f"space_frequency_fold_{fold}.pth"
            )


        # ==========================================================
        # 10. RESULTADOS DOS 5 FOLDS
        # ==========================================================

        fold_accuracies = np.array(
            fold_accuracies
        )


        print()

        print(
            "=" * 60
        )

        print(
            "RESULTADO FINAL"
        )

        print(
            "=" * 60
        )


        for (
            fold,
            accuracy
        ) in enumerate(
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


# ==========================================================
# EXECUTAR
# ==========================================================

if __name__ == "__main__":

    main()