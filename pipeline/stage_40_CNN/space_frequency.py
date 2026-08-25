import copy
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split
)


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

N_FOLDS = 6
N_EPOCHS = 300
BATCH_SIZE = 8
LEARNING_RATE = 0.001

# 20% dos 80% de desenvolvimento serão validação
VALIDATION_SIZE = 0.20

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

       # (batch, 9, 65, 21, 21)

       self.conv1 = nn.Conv3d(
           in_channels=9,
           out_channels=16,
           kernel_size=3,
           padding=1
       )
       # (batch, 16, 65, 21, 21)


       self.relu1 = nn.ReLU()


       self.pool1 = nn.MaxPool3d(
           kernel_size=(2, 2, 2)
       )
       # (batch, 16, 32, 10, 10)


       self.conv2 = nn.Conv3d(
           in_channels=16,
           out_channels=8,
           kernel_size=3,
           padding=1
       )
       # (batch, 8, 32, 10, 10)

       self.relu2 = nn.ReLU()


       self.pool2 = nn.MaxPool3d(
           kernel_size=(2, 2, 2)
       )
       # (batch, 8, 16, 5, 5)


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

subjects = [f"sub-{i:02d}" for i in range(1, 11)]

def space_frequency(subjects = subjects):

    torch.manual_seed(
        RANDOM_STATE
    )

    # ==========================================================
    # LOOP DOS SUJEITOS
    # ==========================================================

    for subject in subjects:

        print()
        print("#" * 70)
        print(f"SUJEITO: {subject}")
        print("#" * 70)


        # ======================================================
        # CARREGAR OS DADOS
        # ======================================================

        power_array = np.load(
            INPUT_DIR
            / f"{subject}_power.npy"
        )


        labels = np.load(
            INPUT_DIR
            / f"{subject}_labels.npy"
        )


        print(
            "Power:",
            power_array.shape
        )

        print(
            "Labels:",
            labels.shape
        )


        # ======================================================
        # DIRETÓRIO DOS MODELOS
        # ======================================================

        MODEL_DIR = (
            OUTPUT_DIR
            / "space_frequency_models"
            / subject
        )


        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        # ======================================================
        # DIRETÓRIO DOS ÍNDICES DOS FOLDS
        # ======================================================

        FOLDS_DIR = (
            OUTPUT_DIR
            / "folds"
            / subject
        )


        FOLDS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        # ======================================================
        # REORGANIZAR AS DIMENSÕES
        # ======================================================

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


        print(
            "Power reorganizado:",
            power_array.shape
        )


        # ======================================================
        # CONVERTER PARA TENSOR
        # ======================================================

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


        # ======================================================
        # CPU OU GPU
        # ======================================================

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


        # ======================================================
        # CRIAR OS 5 FOLDS EXTERNOS
        # ======================================================

        skf = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE
        )


        # ======================================================
        # RESULTADOS DOS FOLDS
        # ======================================================

        fold_accuracies = []

        fold_losses = []


        # ======================================================
        # LOOP DOS 5 FOLDS
        # ======================================================

        for fold, (
            development_indices,
            test_indices
        ) in enumerate(
            skf.split(
                X,
                labels
            ),
            start=1
        ):


            print()
            print("=" * 70)
            print(
                f"FOLD {fold}/{N_FOLDS}"
            )
            print("=" * 70)


            # ==================================================
            # DIVIDIR OS 80% DE DESENVOLVIMENTO
            # EM TREINAMENTO E VALIDAÇÃO
            # ==================================================

            # development_indices contém aproximadamente
            # 80% de todas as amostras.
            #
            # Agora 20% desses 80% serão usados para validação.
            #
            # Resultado em relação ao conjunto total:
            #
            # 64% treino
            # 16% validação
            # 20% teste


            train_indices, val_indices = train_test_split(
                development_indices,
                test_size=VALIDATION_SIZE,
                stratify=labels[
                    development_indices
                ],
                random_state=RANDOM_STATE
            )


            # ==================================================
            # MOSTRAR TAMANHOS
            # ==================================================

            print(
                "Total:",
                len(X)
            )

            print(
                "Treinamento:",
                len(train_indices)
            )

            print(
                "Validação:",
                len(val_indices)
            )

            print(
                "Teste:",
                len(test_indices)
            )


            print(
                "Treinamento (%):",
                len(train_indices)
                / len(X)
            )

            print(
                "Validação (%):",
                len(val_indices)
                / len(X)
            )

            print(
                "Teste (%):",
                len(test_indices)
                / len(X)
            )


            # ==================================================
            # SALVAR OS ÍNDICES
            # ==================================================

            # Esses são índices relativos ao conjunto ORIGINAL.
            #
            # Portanto, o ramo espaço-tempo poderá carregá-los
            # diretamente.

            np.savez(
                FOLDS_DIR
                / f"fold_{fold}.npz",

                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices
            )


            # ==================================================
            # SEPARAR OS DADOS
            # ==================================================

            X_train = X[
                train_indices
            ]

            y_train = y[
                train_indices
            ]


            X_val = X[
                val_indices
            ]

            y_val = y[
                val_indices
            ]


            X_test = X[
                test_indices
            ]

            y_test = y[
                test_indices
            ]


            # ==================================================
            # NORMALIZAÇÃO
            # ==================================================

            # Muito importante:
            #
            # power_max é obtido SOMENTE
            # a partir do conjunto de treinamento.

            power_max = X_train.max()


            X_train = (
                X_train
                / power_max
            )


            X_val = (
                X_val
                / power_max
            )


            X_test = (
                X_test
                / power_max
            )


            print(
                "X_train:",
                X_train.shape
            )

            print(
                "X_val:",
                X_val.shape
            )

            print(
                "X_test:",
                X_test.shape
            )


            # ==================================================
            # DATASETS
            # ==================================================

            train_dataset = TensorDataset(
                X_train,
                y_train
            )


            val_dataset = TensorDataset(
                X_val,
                y_val
            )


            test_dataset = TensorDataset(
                X_test,
                y_test
            )


            # ==================================================
            # DATALOADERS
            # ==================================================

            generator = torch.Generator()


            generator.manual_seed(
                RANDOM_STATE_LOADER
            )


            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=generator
            )


            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False
            )


            test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False
            )


            # ==================================================
            # CRIAR NOVO MODELO PARA ESTE FOLD
            # ==================================================

            torch.manual_seed(
                RANDOM_STATE
            )


            model = SpaceFrequencyCNN().to(
                device
            )


            # ==================================================
            # FUNÇÃO DE PERDA
            # ==================================================

            criterion = nn.CrossEntropyLoss()


            # ==================================================
            # OTIMIZADOR
            # ==================================================

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LEARNING_RATE
            )


            # ==================================================
            # HISTÓRICO DO FOLD
            # ==================================================

            train_accuracies_epoch = []

            val_accuracies_epoch = []


            train_losses_epoch = []

            val_losses_epoch = []


            # ==================================================
            # MELHOR MODELO
            # ==================================================

            best_val_loss = float(
                "inf"
            )


            best_epoch = 0


            best_model_state = None


            # ==================================================
            # TREINAMENTO
            # ==================================================

            for epoch in range(N_EPOCHS):


                # ==============================================
                # TREINAMENTO DA ÉPOCA
                # ==============================================

                model.train()


                train_total_loss = 0.0

                train_correct = 0

                train_total = 0


                for (
                    X_batch,
                    y_batch
                ) in train_loader:


                    X_batch = X_batch.to(
                        device
                    )

                    y_batch = y_batch.to(
                        device
                    )


                    # ------------------------------------------
                    # ZERAR GRADIENTES
                    # ------------------------------------------

                    optimizer.zero_grad()


                    # ------------------------------------------
                    # FORWARD
                    # ------------------------------------------

                    outputs = model(
                        X_batch
                    )


                    # ------------------------------------------
                    # LOSS
                    # ------------------------------------------

                    loss = criterion(
                        outputs,
                        y_batch
                    )


                    # ------------------------------------------
                    # BACKPROPAGATION
                    # ------------------------------------------

                    loss.backward()


                    # ------------------------------------------
                    # ATUALIZAR PESOS
                    # ------------------------------------------

                    optimizer.step()


                    # ------------------------------------------
                    # ACUMULAR LOSS
                    # ------------------------------------------

                    train_total_loss += (
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

                    train_correct += (
                        predictions
                        == y_batch
                    ).sum().item()


                    train_total += (
                        y_batch.size(0)
                    )


                # ==============================================
                # RESULTADOS DE TREINAMENTO
                # ==============================================

                train_loss = (
                    train_total_loss
                    / len(train_loader)
                )


                train_accuracy = (
                    train_correct
                    / train_total
                )


                # ==============================================
                # VALIDAÇÃO DA ÉPOCA
                # ==============================================

                model.eval()


                val_total_loss = 0.0

                val_correct = 0

                val_total = 0


                with torch.no_grad():


                    for (
                        X_batch,
                        y_batch
                    ) in val_loader:


                        X_batch = X_batch.to(
                            device
                        )

                        y_batch = y_batch.to(
                            device
                        )


                        outputs = model(
                            X_batch
                        )


                        loss = criterion(
                            outputs,
                            y_batch
                        )


                        val_total_loss += (
                            loss.item()
                        )


                        predictions = outputs.argmax(
                            dim=1
                        )


                        val_correct += (
                            predictions
                            == y_batch
                        ).sum().item()


                        val_total += (
                            y_batch.size(0)
                        )


                # ==============================================
                # RESULTADOS DE VALIDAÇÃO
                # ==============================================

                val_loss = (
                    val_total_loss
                    / len(val_loader)
                )


                val_accuracy = (
                    val_correct
                    / val_total
                )


                # ==============================================
                # GUARDAR HISTÓRICO
                # ==============================================

                train_accuracies_epoch.append(
                    train_accuracy
                )


                val_accuracies_epoch.append(
                    val_accuracy
                )


                train_losses_epoch.append(
                    train_loss
                )


                val_losses_epoch.append(
                    val_loss
                )


                # ==============================================
                # VERIFICAR SE É O MELHOR MODELO
                # ==============================================

                if val_loss < best_val_loss:

                    best_val_loss = (
                        val_loss
                    )


                    best_epoch = (
                        epoch
                    )


                    best_model_state = copy.deepcopy(
                        model.state_dict()
                    )


                # ==============================================
                # MOSTRAR RESULTADOS
                # ==============================================

                print(
                    f"Fold {fold} | "
                    f"Epoch {epoch + 1:03d}/{N_EPOCHS} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {train_accuracy:.4f} | "
                    f"Val Acc: {val_accuracy:.4f}"
                )


            # ==================================================
            # RESTAURAR O MELHOR MODELO
            # ==================================================

            # model atualmente contém os pesos da época 100.
            #
            # Aqui voltamos para os pesos correspondentes
            # à menor validation loss.

            # model.load_state_dict(
            #     best_model_state
            # )


            print()

            print(
                f"Melhor época: {best_epoch}"
            )


            print(
                f"Menor Validation Loss: "
                f"{best_val_loss:.6f}"
            )


            # ==================================================
            # TESTE FINAL
            # ==================================================

            # O conjunto de teste aparece SOMENTE agora,
            # depois que o modelo já foi escolhido.

            model.eval()


            test_total_loss = 0.0

            test_correct = 0

            test_total = 0


            all_predictions = []

            all_labels = []


            with torch.no_grad():


                for (
                    X_batch,
                    y_batch
                ) in test_loader:


                    X_batch = X_batch.to(
                        device
                    )

                    y_batch = y_batch.to(
                        device
                    )


                    outputs = model(
                        X_batch
                    )


                    loss = criterion(
                        outputs,
                        y_batch
                    )


                    test_total_loss += (
                        loss.item()
                    )


                    predictions = outputs.argmax(
                        dim=1
                    )


                    test_correct += (
                        predictions
                        == y_batch
                    ).sum().item()


                    test_total += (
                        y_batch.size(0)
                    )


                    all_predictions.extend(
                        predictions
                        .cpu()
                        .numpy()
                    )


                    all_labels.extend(
                        y_batch
                        .cpu()
                        .numpy()
                    )


            # ==================================================
            # RESULTADOS DO TESTE
            # ==================================================

            test_loss = (
                test_total_loss
                / len(test_loader)
            )


            test_accuracy = (
                test_correct
                / test_total
            )


            fold_accuracies.append(
                test_accuracy
            )


            fold_losses.append(
                test_loss
            )


            print()

            print(
                "--- RESULTADO DO TESTE ---"
            )


            print(
                f"Fold {fold}"
            )


            print(
                f"Best Epoch: "
                f"{best_epoch}"
            )


            print(
                f"Best Validation Loss: "
                f"{best_val_loss:.6f}"
            )


            print(
                f"Test Loss: "
                f"{test_loss:.6f}"
            )


            print(
                f"Test Accuracy: "
                f"{test_accuracy:.4f}"
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


            # ==================================================
            # SALVAR O CHECKPOINT DO FOLD
            # ==================================================

            torch.save(
                {

                    # ------------------------------------------
                    # MELHOR MODELO
                    # ------------------------------------------

                    "model_state_dict":
                        model.state_dict(),


                    # ------------------------------------------
                    # NORMALIZAÇÃO
                    # ------------------------------------------

                    "power_max":
                        power_max.item(),


                    # ------------------------------------------
                    # FOLD
                    # ------------------------------------------

                    "fold":
                        fold,


                    # ------------------------------------------
                    # ÍNDICES
                    # ------------------------------------------

                    "train_indices":
                        train_indices,

                    "val_indices":
                        val_indices,

                    "test_indices":
                        test_indices,


                    # ------------------------------------------
                    # MELHOR ÉPOCA
                    # ------------------------------------------

                    "best_epoch":
                        best_epoch,

                    "best_val_loss":
                        best_val_loss,


                    # ------------------------------------------
                    # HISTÓRICO DAS ACURÁCIAS
                    # ------------------------------------------

                    "train_accuracies_epoch":
                        train_accuracies_epoch,

                    "val_accuracies_epoch":
                        val_accuracies_epoch,


                    # ------------------------------------------
                    # HISTÓRICO DAS LOSSES
                    # ------------------------------------------

                    "train_losses_epoch":
                        train_losses_epoch,

                    "val_losses_epoch":
                        val_losses_epoch,


                    # ------------------------------------------
                    # RESULTADO DO TESTE
                    # ------------------------------------------

                    "test_loss":
                        test_loss,

                    "test_accuracy":
                        test_accuracy,

                    "all_predictions":
                        np.array(
                            all_predictions
                        ),

                    "all_labels":
                        np.array(
                            all_labels
                        ),

                    "fold_accuracies": np.array(fold_accuracies),

                    "fold_losses": np.array(fold_losses),

                },

                MODEL_DIR
                / f"space_frequency_fold_{fold}.pth"
            )


        # ==========================================================
        # RESULTADOS DOS 5 FOLDS
        # ==========================================================

        fold_accuracies = np.array(
            fold_accuracies
        )


        fold_losses = np.array(
            fold_losses
        )


        print()
        print("=" * 70)
        print(
            f"RESULTADO FINAL - {subject}"
        )
        print("=" * 70)


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


        print(
            "Test Loss média:",
            fold_losses.mean()
        )


# ==========================================================
# EXECUTAR
# ==========================================================

if __name__ == "__main__":

    space_frequency(subjects=["sub-02"])