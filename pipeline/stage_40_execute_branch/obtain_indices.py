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

from torchinfo import summary

from tools import save_summary_as_pdf

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

N_FOLDS = 6
N_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 0.001

# Percentual de desenvolvimento que serão de validação
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
    / "stage_40_execute_branch"
)


# ==========================================================
# MAIN FUNCTION
# ==========================================================

subjects = [f"sub-{i:02d}" for i in range(1, 11)]

def obtain_indices(
        subjects = subjects,
        N_FOLDS = N_FOLDS,
        VALIDATION_SIZE = VALIDATION_SIZE,
        RANDOM_STATE = RANDOM_STATE,
):

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
        # DIRETÓRIO DOS ÍNDICES DOS FOLDS
        # ======================================================

        FOLDS_DIR = (
            OUTPUT_DIR
            / "fold_indices"
            / subject
        )


        FOLDS_DIR.mkdir(
            parents=True,
            exist_ok=True
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
        # LOOP DOS 5 FOLDS
        # ======================================================

        for fold, ( development_indices, test_indices ) in enumerate( skf.split( X, labels), start=1 ):


            print()
            print("=" * 70)
            print(
                f"FOLD {fold}/{N_FOLDS}"
            )
            print("=" * 70)


            # ==================================================
            # DIVIDIR A PORCENTAGEM DE DESENVOLVIMENTO EM
            # TREINAMENTO E VALIDAÇÃO
            # ==================================================

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


# ==========================================================
# EXECUTAR
# ==========================================================

if __name__ == "__main__":

    obtain_indices()