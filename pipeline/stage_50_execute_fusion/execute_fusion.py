import copy
import numpy as np
from pathlib import Path
import sys
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)




# ==========================================================
# CONFIGURAÇÕES DE CAMINHO & IMPORTS LOCAIS
# ==========================================================

current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[2]
PIPELINE_ROOT = current_file.parents[1]

if str(PIPELINE_ROOT) not in sys.path:
    sys.path.append(str(PIPELINE_ROOT))

from tools import save_summary

from stage_40_execute_branch.branch_cnn_models import (
    Space1Space2FrequencyTimeCNN,
    Space1Space2TimeFrequencyCNN,
    TimeFrequencySpace2Space1CNN,
    TimeFrequencySpace1Space2CNN,
)

from fusion_cnn_models import GatedFusion

from fusion_individual_subject_plots import fusion_individual_subject_plots

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

N_FOLDS = 6
N_EPOCHS_STAGE_1 = 600
N_EPOCHS_STAGE_2 = 100
BATCH_SIZE = 8
LEARNING_RATE_STAGE_1 = 0.001
LEARNING_RATE_STAGE_2 = 0.0001

RANDOM_STATE = 42
RANDOM_STATE_LOADER = 42

INPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_30_array_assembly"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_50_execute_fusion"
)


INPUT_PREVIOUS_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_40_execute_branch"
)

# ==========================================================
# FUNÇÕES AUXILIARES
# ==========================================================

def test_model(
    model,
    model_state_dict,
    test_loader,
    device,
    criterion,
):
    """
    Testa um checkpoint específico sem usar o conjunto de teste
    para escolher o modelo.
    """

    model.load_state_dict(
        model_state_dict
    )

    model.eval()

    test_total_loss = 0.0
    test_correct = 0
    test_total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for (
            X_s1_s2_f_t_batch,
            X_s1_s2_t_f_batch,
            X_t_f_s2_s1_batch,
            X_t_f_s1_s2_batch,
            y_batch
        ) in test_loader:

            X_s1_s2_f_t_batch = (
                X_s1_s2_f_t_batch.to(device)
            )

            X_s1_s2_t_f_batch = (
                X_s1_s2_t_f_batch.to(device)
            )

            X_t_f_s2_s1_batch = (
                X_t_f_s2_s1_batch.to(device)
            )

            X_t_f_s1_s2_batch = (
                X_t_f_s1_s2_batch.to(device)
            )

            y_batch = y_batch.to(
                device
            )

            outputs = model(
                X_s1_s2_f_t_batch,
                X_s1_s2_t_f_batch,
                X_t_f_s2_s1_batch,
                X_t_f_s1_s2_batch
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
                predictions == y_batch
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

    test_loss = (
        test_total_loss
        / len(test_loader)
    )

    test_accuracy = (
        test_correct
        / test_total
    )

    return (
        test_loss,
        test_accuracy,
        all_labels,
        all_predictions,
    )


def print_fold_results(
    model_type,
    fold,
    best_epoch,
    focus_property,
    test_loss,
    test_accuracy,
    all_labels,
    all_predictions,
):

    print()

    match model_type:

        case "minimum_loss":

            message = (
                "--- TEST WITH MINIMUM LOSS ---"
            )

            property_message = (
                f"Minimum Validation Loss: "
                f"{focus_property:.6f}"
            )

        case "maximum_accuracy":

            message = (
                "--- TEST WITH MAXIMUM ACCURACY ---"
            )

            property_message = (
                f"Maximum Validation Accuracy: "
                f"{focus_property:.6f}"
            )

        case _:
            raise ValueError(
                f"Invalid model_type {model_type}"
            )

    print(message)

    print(
        f"Fold {fold}"
    )

    print(
        f"Best Epoch: {best_epoch + 1}"
    )

    print(
        property_message
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


def print_folds_summary(
    title,
    fold_accuracies,
    fold_losses,
):

    fold_accuracies = np.array(
        fold_accuracies
    )

    fold_losses = np.array(
        fold_losses
    )

    print()
    print("=" * 70)
    print(title)
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


def train_stage(
    model,
    train_loader,
    val_loader,
    device,
    criterion,
    optimizer,
    n_epochs,
    fold,
    frozen_models=None,
):
    """
    Treina um estágio e guarda dois checkpoints:

    1. menor validation loss
    2. maior validation accuracy

    Em caso de empate de acurácia, escolhe o modelo
    com menor validation loss.
    """

    train_accuracies_epoch = []
    val_accuracies_epoch = []

    train_losses_epoch = []
    val_losses_epoch = []

    min_val_loss = float("inf")
    min_val_loss_epoch = 0
    min_val_loss_model_state = None

    max_val_accuracy = -1.0
    best_val_loss_to_max_accuracy = float("inf")
    max_val_accuracy_epoch = 0
    max_val_accuracy_model_state = None

    for epoch in range(n_epochs):

        # ==================================================
        # TREINAMENTO
        # ==================================================

        model.train()

        # Se os ramos estiverem congelados,
        # eles devem permanecer em eval(), pois model.train()
        # coloca todos os submódulos em modo de treinamento.
        if frozen_models is not None:

            for frozen_model in frozen_models:
                frozen_model.eval()

        train_total_loss = 0.0
        train_correct = 0
        train_total = 0

        for (
            X_s1_s2_f_t_batch,
            X_s1_s2_t_f_batch,
            X_t_f_s2_s1_batch,
            X_t_f_s1_s2_batch,
            y_batch
        ) in train_loader:

            X_s1_s2_f_t_batch = (
                X_s1_s2_f_t_batch.to(device)
            )

            X_s1_s2_t_f_batch = (
                X_s1_s2_t_f_batch.to(device)
            )

            X_t_f_s2_s1_batch = (
                X_t_f_s2_s1_batch.to(device)
            )

            X_t_f_s1_s2_batch = (
                X_t_f_s1_s2_batch.to(device)
            )
     
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(
                X_s1_s2_f_t_batch,
                X_s1_s2_t_f_batch,
                X_t_f_s2_s1_batch,
                X_t_f_s1_s2_batch,
            )

            loss = criterion(
                outputs,
                y_batch
            )

            loss.backward()

            optimizer.step()

            train_total_loss += (
                loss.item()
            )

            predictions = outputs.argmax(
                dim=1
            )

            train_correct += (
                predictions == y_batch
            ).sum().item()

            train_total += (
                y_batch.size(0)
            )

        train_loss = (
            train_total_loss
            / len(train_loader)
        )

        train_accuracy = (
            train_correct
            / train_total
        )

        # ==================================================
        # VALIDAÇÃO
        # ==================================================

        model.eval()

        val_total_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for (
                X_s1_s2_f_t_batch,
                X_s1_s2_t_f_batch,
                X_t_f_s2_s1_batch,
                X_t_f_s1_s2_batch,
                y_batch
            ) in val_loader:

                X_s1_s2_f_t_batch = (
                    X_s1_s2_f_t_batch.to(device)
                )

                X_s1_s2_t_f_batch = (
                    X_s1_s2_t_f_batch.to(device)
                )

                X_t_f_s2_s1_batch = (
                    X_t_f_s2_s1_batch.to(device)
                )
    
                X_t_f_s1_s2_batch = (
                    X_t_f_s1_s2_batch.to(device)
                )
                y_batch = y_batch.to(
                    device
                )

                outputs = model(
                    X_s1_s2_f_t_batch,
                    X_s1_s2_t_f_batch,
                    X_t_f_s2_s1_batch,
                    X_t_f_s1_s2_batch,
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
                    predictions == y_batch
                ).sum().item()

                val_total += (
                    y_batch.size(0)
                )

        val_loss = (
            val_total_loss
            / len(val_loader)
        )

        val_accuracy = (
            val_correct
            / val_total
        )

        # ==================================================
        # HISTÓRICO
        # ==================================================

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

        # ==================================================
        # MELHOR MODELO: MENOR LOSS
        # ==================================================

        if val_loss < min_val_loss:

            min_val_loss = (
                val_loss
            )

            min_val_loss_epoch = (
                epoch
            )

            min_val_loss_model_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

        # ==================================================
        # MELHOR MODELO: MAIOR ACURÁCIA
        # ==================================================

        if (
            val_accuracy > max_val_accuracy
            or (
                val_accuracy == max_val_accuracy
                and val_loss < best_val_loss_to_max_accuracy
            )
        ):

            max_val_accuracy = (
                val_accuracy
            )

            best_val_loss_to_max_accuracy = (
                val_loss
            )

            max_val_accuracy_epoch = (
                epoch
            )

            max_val_accuracy_model_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

        print(
            f"Fold {fold} | "
            f"Epoch {epoch + 1:03d}/{n_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

    return {
        "train_losses_epoch":
            train_losses_epoch,

        "val_losses_epoch":
            val_losses_epoch,

        "train_accuracies_epoch":
            train_accuracies_epoch,

        "val_accuracies_epoch":
            val_accuracies_epoch,

        "min_val_loss":
            min_val_loss,

        "min_val_loss_epoch":
            min_val_loss_epoch,

        "min_val_loss_model_state":
            min_val_loss_model_state,

        "max_val_accuracy":
            max_val_accuracy,

        "best_val_loss_to_max_accuracy":
            best_val_loss_to_max_accuracy,

        "max_val_accuracy_epoch":
            max_val_accuracy_epoch,

        "max_val_accuracy_model_state":
            max_val_accuracy_model_state,
    }



def get_scalar(value):
    """
        Converte tensor para escalar
    """

    if torch.is_tensor(value):
        return value.item()

    return float(value)



# ==========================================================
# MAIN
# ==========================================================

subjects = [
    f"sub-{i:02d}"
    for i in range(1, 11)
]


def execute_fusion(
    fusion,
    branch_model_state = "minimum_loss_model",
    subjects=subjects,
    N_FOLDS=N_FOLDS,
    N_EPOCHS_STAGE_1=N_EPOCHS_STAGE_1,
    N_EPOCHS_STAGE_2=N_EPOCHS_STAGE_2,
    BATCH_SIZE=BATCH_SIZE,
    LEARNING_RATE_STAGE_1=LEARNING_RATE_STAGE_1,
    LEARNING_RATE_STAGE_2=LEARNING_RATE_STAGE_2,
    RANDOM_STATE=RANDOM_STATE,
    RANDOM_STATE_LOADER=RANDOM_STATE_LOADER,
):

    torch.manual_seed(
        RANDOM_STATE
    )

    branch_s1_s2_f_t = (
        "space1_space2_frequency_time"
    )

    branch_s1_s2_t_f = (
        "space1_space2_time_frequency"
    )

    branch_t_f_s2_s1 = (
        "time_frequency_space2_space1"
    )

    branch_t_f_s1_s2 = (
        "time_frequency_space1_space2"
    )    

    # ==========================================================
    # LOOP DOS SUJEITOS
    # ==========================================================

    for subject in subjects:

        print()
        print("#" * 70)
        print(
            f"SUJEITO: {subject}"
        )
        print("#" * 70)

        # ======================================================
        # CARREGAR DADOS
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
        # DIRETÓRIOS
        # ======================================================

        S1_S2_F_T_MODEL_DIR = (
            INPUT_PREVIOUS_DIR
            / f"{branch_s1_s2_f_t}_branch"
        )

        S1_S2_F_T_SUB_DIR = (
            S1_S2_F_T_MODEL_DIR
            / subject
        )

        S1_S2_T_F_MODEL_DIR = (
            INPUT_PREVIOUS_DIR
            / f"{branch_s1_s2_t_f}_branch"
        )

        S1_S2_T_F_SUB_DIR = (
            S1_S2_T_F_MODEL_DIR
            / subject
        )

        T_F_S2_S1_MODEL_DIR = (
            INPUT_PREVIOUS_DIR
            / f"{branch_t_f_s2_s1}_branch"
        )

        T_F_S2_S1_SUB_DIR = (
            T_F_S2_S1_MODEL_DIR
            / subject
        )

        T_F_S1_S2_MODEL_DIR = (
            INPUT_PREVIOUS_DIR
            / f"{branch_t_f_s1_s2}_branch"
        )

        T_F_S1_S2_SUB_DIR = (
            T_F_S1_S2_MODEL_DIR
            / subject
        )

        MODEL_DIR = (
            OUTPUT_DIR
            / f"{fusion}_from_{branch_model_state}_in_branches_fusion"
        )

        SUB_DIR = (
            MODEL_DIR
            / subject
        )

        SUB_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        FOLDS_DIR = (
            INPUT_PREVIOUS_DIR
            / "fold_indices"
            / subject
        )

        FOLDS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # ======================================================
        # REORGANIZAR DIMENSÕES
        # ======================================================

        #original: 


        power_array_s1_s2_f_t = np.transpose(
            power_array,
            (0, 4, 3, 1, 2)
        )

        power_array_s1_s2_t_f = np.transpose(
            power_array,
            (0, 3, 4, 1, 2)
        )

        power_array_t_f_s2_s1 = np.transpose(
            power_array,
            (0, 2, 1, 3, 4)
        )

        power_array_t_f_s1_s2 = np.transpose(
            power_array,
            (0, 1, 2, 3, 4)
        )

        print(
            "Power Space1 x Space2 x Frequency x Time reorganizado:",
            power_array_s1_s2_f_t.shape
        )

        print(
            "Power Space1 x Space2 x Time x Frequency reorganizado:",
            power_array_s1_s2_t_f.shape
        )

        print(
            "Power Time x Frequency x Space2 x Space1 reorganizado:",
            power_array_t_f_s2_s1.shape
        )

        print(
            "Power Time x Frequency x Space1 x Space2 reorganizado:",
            power_array_t_f_s1_s2.shape
        )

        # ======================================================
        # CONVERTER PARA TENSOR
        # ======================================================

        X_s1_s2_f_t = torch.from_numpy(
            power_array_s1_s2_f_t
        ).float()

        X_s1_s2_t_f = torch.from_numpy(
            power_array_s1_s2_t_f
        ).float()

        X_t_f_s2_s1 = torch.from_numpy(
            power_array_t_f_s2_s1
        ).float()

        X_t_f_s1_s2 = torch.from_numpy(
            power_array_t_f_s1_s2
        ).float()

        y = torch.from_numpy(
            labels
        ).long()

        # obtendo os sizes de cada tensor
        summary_input_sizes = [
            (1, *X_s1_s2_f_t.shape[1:]),
            (1, *X_s1_s2_t_f.shape[1:]),
            (1, *X_t_f_s2_s1.shape[1:]),
            (1, *X_t_f_s1_s2.shape[1:]),
        ]


        print(
            "X_s1_s2_f_t:",
            X_s1_s2_f_t.shape
        )

        print(
            "X_s1_s2_t_f:",
            X_s1_s2_t_f.shape
        )

        print(
            "X_t_f_s2_s1:",
            X_t_f_s2_s1.shape
        )

        print(
            "X_t_f_s1_s2:",
            X_t_f_s1_s2.shape
        )

        print(
            "y:",
            y.shape
        )

        # ======================================================
        # DEVICE
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
        # RESULTADOS POR FOLD
        # ======================================================

        stage_1_fold_accuracies_ml = []
        stage_1_fold_losses_ml = []

        stage_1_fold_accuracies_ma = []
        stage_1_fold_losses_ma = []

        stage_2_from_ml_fold_accuracies_ml = []
        stage_2_from_ml_fold_losses_ml = []

        stage_2_from_ml_fold_accuracies_ma = []
        stage_2_from_ml_fold_losses_ma = []

        stage_2_from_ma_fold_accuracies_ml = []
        stage_2_from_ma_fold_losses_ml = []

        stage_2_from_ma_fold_accuracies_ma = []
        stage_2_from_ma_fold_losses_ma = []

        # ======================================================
        # LOOP DOS FOLDS
        # ======================================================

        for fold in range(
            1,
            N_FOLDS + 1
        ):

            print()
            print("=" * 70)
            print(
                f"FOLD {fold}/{N_FOLDS}"
            )
            print("=" * 70)

            # ==================================================
            # ÍNDICES
            # ==================================================

            fold_data = np.load(
                FOLDS_DIR
                / f"fold_{fold}.npz"
            )

            train_indices = (
                fold_data["train_indices"]
            )

            val_indices = (
                fold_data["val_indices"]
            )

            test_indices = (
                fold_data["test_indices"]
            )

            print(
                "Total:",
                len(X_s1_s2_f_t)
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

            # ==================================================
            # SEPARAR DADOS
            # ==================================================

            X_s1_s2_f_t_train = (
                X_s1_s2_f_t[train_indices]
            )

            X_s1_s2_f_t_val = (
                X_s1_s2_f_t[val_indices]
            )

            X_s1_s2_f_t_test = (
                X_s1_s2_f_t[test_indices]
            )

            X_s1_s2_t_f_train = (
                X_s1_s2_t_f[train_indices]
            )

            X_s1_s2_t_f_val = (
                X_s1_s2_t_f[val_indices]
            )

            X_s1_s2_t_f_test = (
                X_s1_s2_t_f[test_indices]
            )

            X_t_f_s2_s1_train = (
                X_t_f_s2_s1[train_indices]
            )

            X_t_f_s2_s1_val = (
                X_t_f_s2_s1[val_indices]
            )

            X_t_f_s2_s1_test = (
                X_t_f_s2_s1[test_indices]
            )

            X_t_f_s1_s2_train = (
                X_t_f_s1_s2[train_indices]
            )

            X_t_f_s1_s2_val = (
                X_t_f_s1_s2[val_indices]
            )

            X_t_f_s1_s2_test = (
                X_t_f_s1_s2[test_indices]
            )

            y_train = (
                y[train_indices]
            )

            y_val = (
                y[val_indices]
            )

            y_test = (
                y[test_indices]
            )

            # ==================================================
            # CARREGAR CHECKPOINT DOS RAMOS
            # ==================================================

            checkpoint_s1_s2_f_t = torch.load(
                S1_S2_F_T_SUB_DIR
                / (
                    "space1_space2_frequency_time_"
                    f"fold_{fold}.pth"
                ),
                map_location=device,
                weights_only=False
            )

            checkpoint_s1_s2_t_f = torch.load(
                S1_S2_T_F_SUB_DIR
                / (
                    "space1_space2_time_frequency_"
                    f"fold_{fold}.pth"
                ),
                map_location=device,
                weights_only=False
            )

            checkpoint_t_f_s2_s1 = torch.load(
                T_F_S2_S1_SUB_DIR
                / (
                    "time_frequency_space2_space1_"
                    f"fold_{fold}.pth"
                ),
                map_location=device,
                weights_only=False
            )

            checkpoint_t_f_s1_s2 = torch.load(
                T_F_S1_S2_SUB_DIR
                / (
                    "time_frequency_space1_space2_"
                    f"fold_{fold}.pth"
                ),
                map_location=device,
                weights_only=False
            )

            power_max_s1_s2_f_t = get_scalar(
                checkpoint_s1_s2_f_t["power_max"]
            )

            power_max_s1_s2_t_f = get_scalar(
                checkpoint_s1_s2_t_f["power_max"]
            )

            power_max_t_f_s2_s1 = get_scalar(
                checkpoint_t_f_s2_s1["power_max"]
            )

            power_max_t_f_s1_s2 = get_scalar(
                checkpoint_t_f_s1_s2["power_max"]
            )

            # ==================================================
            # NORMALIZAÇÃO
            # ==================================================

            X_s1_s2_f_t_train = (
                X_s1_s2_f_t_train
                / power_max_s1_s2_f_t
            )

            X_s1_s2_f_t_val = (
                X_s1_s2_f_t_val
                / power_max_s1_s2_f_t
            )

            X_s1_s2_f_t_test = (
                X_s1_s2_f_t_test
                / power_max_s1_s2_f_t
            )

            X_s1_s2_t_f_train = (
                X_s1_s2_t_f_train
                / power_max_s1_s2_t_f
            )

            X_s1_s2_t_f_val = (
                X_s1_s2_t_f_val
                / power_max_s1_s2_t_f
            )

            X_s1_s2_t_f_test = (
                X_s1_s2_t_f_test
                / power_max_s1_s2_t_f
            )

            X_t_f_s2_s1_train = (
                X_t_f_s2_s1_train
                / power_max_t_f_s2_s1
            )

            X_t_f_s2_s1_val = (
                X_t_f_s2_s1_val
                / power_max_t_f_s2_s1
            )

            X_t_f_s2_s1_test = (
                X_t_f_s2_s1_test
                / power_max_t_f_s2_s1
            )

            X_t_f_s1_s2_train = (
                X_t_f_s1_s2_train
                / power_max_t_f_s1_s2
            )

            X_t_f_s1_s2_val = (
                X_t_f_s1_s2_val
                / power_max_t_f_s1_s2
            )

            X_t_f_s1_s2_test = (
                X_t_f_s1_s2_test
                / power_max_t_f_s1_s2
            )

            # ==================================================
            # DATASETS
            # ==================================================

            train_dataset = TensorDataset(
                X_s1_s2_f_t_train,
                X_s1_s2_t_f_train,
                X_t_f_s2_s1_train,
                X_t_f_s1_s2_train,
                y_train
            )

            val_dataset = TensorDataset(
                X_s1_s2_f_t_val,
                X_s1_s2_t_f_val,
                X_t_f_s2_s1_val,
                X_t_f_s1_s2_val,
                y_val
            )

            test_dataset = TensorDataset(
                X_s1_s2_f_t_test,
                X_s1_s2_t_f_test,
                X_t_f_s2_s1_test,
                X_t_f_s1_s2_test,
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
            # MODELOS DOS RAMOS
            # ==================================================

            torch.manual_seed(
                RANDOM_STATE
            )

            model_s1_s2_f_t = (
                Space1Space2FrequencyTimeCNN()
                .to(device)
            )

            model_s1_s2_f_t.load_state_dict(
                checkpoint_s1_s2_f_t[branch_model_state][
                    "model_state_dict"
                ]
            )

            model_s1_s2_t_f = (
                Space1Space2TimeFrequencyCNN()
                .to(device)
            )

            model_s1_s2_t_f.load_state_dict(
                checkpoint_s1_s2_t_f[branch_model_state][
                    "model_state_dict"
                ]
            )

            model_t_f_s2_s1 = (
                TimeFrequencySpace2Space1CNN()
                .to(device)
            )

            model_t_f_s2_s1.load_state_dict(
                checkpoint_t_f_s2_s1[branch_model_state][
                    "model_state_dict"
                ]
            )

            model_t_f_s1_s2 = (
                TimeFrequencySpace1Space2CNN()
                .to(device)
            )

            model_t_f_s1_s2.load_state_dict(
                checkpoint_t_f_s1_s2[branch_model_state][
                    "model_state_dict"
                ]
            )

            # ==================================================
            # CONGELAR RAMOS PARA STAGE 1
            # ==================================================

            for parameter in (
                model_s1_s2_f_t.parameters()
            ):

                parameter.requires_grad = (
                    False
                )

            for parameter in (
                model_s1_s2_t_f.parameters()
            ):

                parameter.requires_grad = (
                    False
                )

            for parameter in (
                model_t_f_s2_s1.parameters()
            ):

                parameter.requires_grad = (
                    False
                )

            for parameter in (
                model_t_f_s1_s2.parameters()
            ):

                parameter.requires_grad = (
                    False
                )

            # ==================================================
            # MODELO DE FUSÃO
            # ==================================================

            match fusion:

                case "gated_fusion":

                    model = GatedFusion(
                        model_s1_s2_f_t,
                        model_s1_s2_t_f,
                        model_t_f_s2_s1,
                        model_t_f_s1_s2
                    ).to(device)

                    

                case _:

                    raise ValueError(
                        f"Invalid fusion {fusion}"
                    )

            criterion = (
                nn.CrossEntropyLoss()
            )

            # ==================================================
            # STAGE 1
            # RAMOS CONGELADOS
            # TREINA CLASSIFICADOR MAIS OUTRAS REDE QUE O MODELO
            # DE FUSÃO TIVER
            # ==================================================

            optimizer = torch.optim.Adam(
                filter(
                    lambda parameter:
                    parameter.requires_grad,
                    model.parameters()
                ),
                lr=LEARNING_RATE_STAGE_1
            )

            if fold == 1:

                save_summary(
                    model,
                    fusion,
                    f"Stage 1",
                    summary_input_sizes,
                    device,
                    MODEL_DIR
                )

            stage_1_training = train_stage(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                n_epochs=N_EPOCHS_STAGE_1,
                fold=fold,
                frozen_models=[
                    model_s1_s2_f_t,
                    model_s1_s2_t_f,
                    model_t_f_s2_s1,
                    model_t_f_s1_s2,
                ],
            )

            # ==================================================
            # TESTE STAGE 1 - WITH MINIMUM LOSS MODEL
            # ==================================================

            (
                stage_1_test_loss_ml,
                stage_1_test_accuracy_ml,
                stage_1_all_labels_ml,
                stage_1_all_predictions_ml,
            ) = test_model(
                model,
                stage_1_training[
                    "min_val_loss_model_state"
                ],
                test_loader,
                device,
                criterion,
            )

            stage_1_fold_losses_ml.append(
                stage_1_test_loss_ml
            )

            stage_1_fold_accuracies_ml.append(
                stage_1_test_accuracy_ml
            )

            # ==================================================
            # TESTE STAGE 1 - WITH MAXIMUM ACCURACY MODEL
            # ==================================================

            (
                stage_1_test_loss_ma,
                stage_1_test_accuracy_ma,
                stage_1_all_labels_ma,
                stage_1_all_predictions_ma,
            ) = test_model(
                model,
                stage_1_training[
                    "max_val_accuracy_model_state"
                ],
                test_loader,
                device,
                criterion,
            )

            stage_1_fold_losses_ma.append(
                stage_1_test_loss_ma
            )

            stage_1_fold_accuracies_ma.append(
                stage_1_test_accuracy_ma
            )

            print_fold_results(
                "minimum_loss",
                fold,
                stage_1_training[
                    "min_val_loss_epoch"
                ],
                stage_1_training[
                    "min_val_loss"
                ],
                stage_1_test_loss_ml,
                stage_1_test_accuracy_ml,
                stage_1_all_labels_ml,
                stage_1_all_predictions_ml
            )

            print_fold_results(
                "maximum_accuracy",
                fold,
                stage_1_training[
                    "max_val_accuracy_epoch"
                ],
                stage_1_training[
                    "max_val_accuracy"
                ],
                stage_1_test_loss_ma,
                stage_1_test_accuracy_ma,
                stage_1_all_labels_ma,
                stage_1_all_predictions_ma
            )

            stage_1_results = {

                "training_history": {
                    "train_losses_epoch":
                        stage_1_training[
                            "train_losses_epoch"
                        ],

                    "val_losses_epoch":
                        stage_1_training[
                            "val_losses_epoch"
                        ],

                    "train_accuracies_epoch":
                        stage_1_training[
                            "train_accuracies_epoch"
                        ],

                    "val_accuracies_epoch":
                        stage_1_training[
                            "val_accuracies_epoch"
                        ],
                },

                "minimum_loss": {

                    "validation": {
                        "value":
                            stage_1_training[
                                "min_val_loss"
                            ],

                        "epoch":
                            stage_1_training[
                                "min_val_loss_epoch"
                            ],
                    },

                    "model_state_dict":
                        copy.deepcopy(
                            stage_1_training[
                                "min_val_loss_model_state"
                            ]
                        ),

                    "test": {
                        "loss":
                            stage_1_test_loss_ml,

                        "accuracy":
                            stage_1_test_accuracy_ml,

                        "all_predictions":
                            np.array(
                                stage_1_all_predictions_ml
                            ),

                        "all_labels":
                            np.array(
                                stage_1_all_labels_ml
                            ),
                    },
                },

                "maximum_accuracy": {

                    "validation": {
                        "value":
                            stage_1_training[
                                "max_val_accuracy"
                            ],

                        "epoch":
                            stage_1_training[
                                "max_val_accuracy_epoch"
                            ],

                        "loss_at_best_accuracy":
                            stage_1_training[
                                "best_val_loss_to_max_accuracy"
                            ],
                    },

                    "model_state_dict":
                        copy.deepcopy(
                            stage_1_training[
                                "max_val_accuracy_model_state"
                            ]
                        ),

                    "test": {
                        "loss":
                            stage_1_test_loss_ma,

                        "accuracy":
                            stage_1_test_accuracy_ma,

                        "all_predictions":
                            np.array(
                                stage_1_all_predictions_ma
                            ),

                        "all_labels":
                            np.array(
                                stage_1_all_labels_ma
                            ),
                    },
                },
            }

            # ==================================================
            # LIBERAR TODOS OS PESOS PARA STAGE 2
            # ==================================================

            for parameter in (
                model_s1_s2_f_t.parameters()
            ):

                parameter.requires_grad = (
                    True
                )

            for parameter in (
                model_s1_s2_t_f.parameters()
            ):

                parameter.requires_grad = (
                    True
                )

            for parameter in (
                model_t_f_s2_s1.parameters()
            ):

                parameter.requires_grad = (
                    True
                )

            for parameter in (
                model_t_f_s1_s2.parameters()
            ):

                parameter.requires_grad = (
                    True
                )

            # ==================================================
            # STAGE 2 A PARTIR DO STAGE 1 MINIMUM LOSS
            # ==================================================

            model.load_state_dict(
                stage_1_training[
                    "min_val_loss_model_state"
                ]
            )

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LEARNING_RATE_STAGE_2
            )

            if fold == 1:

                save_summary(
                    model,
                    fusion,
                    f"Stage 2",
                    summary_input_sizes,
                    device,
                    MODEL_DIR
                )

            stage_2_from_ml_training = train_stage(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                n_epochs=N_EPOCHS_STAGE_2,
                fold=fold,
                frozen_models=None,
            )

            # TESTE: mínimo loss
            (
                stage_2_from_ml_test_loss_ml,
                stage_2_from_ml_test_accuracy_ml,
                stage_2_from_ml_all_labels_ml,
                stage_2_from_ml_all_predictions_ml,
            ) = test_model(
                model,
                stage_2_from_ml_training[
                    "min_val_loss_model_state"
                ],
                test_loader,
                device,
                criterion,
            )

            stage_2_from_ml_fold_losses_ml.append(
                stage_2_from_ml_test_loss_ml
            )

            stage_2_from_ml_fold_accuracies_ml.append(
                stage_2_from_ml_test_accuracy_ml
            )

            # TESTE: máxima acurácia
            (
                stage_2_from_ml_test_loss_ma,
                stage_2_from_ml_test_accuracy_ma,
                stage_2_from_ml_all_labels_ma,
                stage_2_from_ml_all_predictions_ma,
            ) = test_model(
                model,
                stage_2_from_ml_training[
                    "max_val_accuracy_model_state"
                ],
                test_loader,
                device,
                criterion,
            )

            stage_2_from_ml_fold_losses_ma.append(
                stage_2_from_ml_test_loss_ma
            )

            stage_2_from_ml_fold_accuracies_ma.append(
                stage_2_from_ml_test_accuracy_ma
            )

            print_fold_results(
                "minimum_loss",
                fold,
                stage_2_from_ml_training[
                    "min_val_loss_epoch"
                ],
                stage_2_from_ml_training[
                    "min_val_loss"
                ],
                stage_2_from_ml_test_loss_ml,
                stage_2_from_ml_test_accuracy_ml,
                stage_2_from_ml_all_labels_ml,
                stage_2_from_ml_all_predictions_ml
            )

            print_fold_results(
                "maximum_accuracy",
                fold,
                stage_2_from_ml_training[
                    "max_val_accuracy_epoch"
                ],
                stage_2_from_ml_training[
                    "max_val_accuracy"
                ],
                stage_2_from_ml_test_loss_ma,
                stage_2_from_ml_test_accuracy_ma,
                stage_2_from_ml_all_labels_ma,
                stage_2_from_ml_all_predictions_ma
            )

            stage_2_from_ml_results = {

                "initial_checkpoint":
                    "stage_1_minimum_loss",

                "training_history": {
                    "train_losses_epoch":
                        stage_2_from_ml_training[
                            "train_losses_epoch"
                        ],

                    "val_losses_epoch":
                        stage_2_from_ml_training[
                            "val_losses_epoch"
                        ],

                    "train_accuracies_epoch":
                        stage_2_from_ml_training[
                            "train_accuracies_epoch"
                        ],

                    "val_accuracies_epoch":
                        stage_2_from_ml_training[
                            "val_accuracies_epoch"
                        ],
                },

                "minimum_loss": {

                    "validation": {
                        "value":
                            stage_2_from_ml_training[
                                "min_val_loss"
                            ],

                        "epoch":
                            stage_2_from_ml_training[
                                "min_val_loss_epoch"
                            ],
                    },

                    "model_state_dict":
                        copy.deepcopy(
                            stage_2_from_ml_training[
                                "min_val_loss_model_state"
                            ]
                        ),

                    "test": {
                        "loss":
                            stage_2_from_ml_test_loss_ml,

                        "accuracy":
                            stage_2_from_ml_test_accuracy_ml,

                        "all_predictions":
                            np.array(
                                stage_2_from_ml_all_predictions_ml
                            ),

                        "all_labels":
                            np.array(
                                stage_2_from_ml_all_labels_ml
                            ),
                    },
                },

                "maximum_accuracy": {

                    "validation": {
                        "value":
                            stage_2_from_ml_training[
                                "max_val_accuracy"
                            ],

                        "epoch":
                            stage_2_from_ml_training[
                                "max_val_accuracy_epoch"
                            ],

                        "loss_at_best_accuracy":
                            stage_2_from_ml_training[
                                "best_val_loss_to_max_accuracy"
                            ],
                    },

                    "model_state_dict":
                        copy.deepcopy(
                            stage_2_from_ml_training[
                                "max_val_accuracy_model_state"
                            ]
                        ),

                    "test": {
                        "loss":
                            stage_2_from_ml_test_loss_ma,

                        "accuracy":
                            stage_2_from_ml_test_accuracy_ma,

                        "all_predictions":
                            np.array(
                                stage_2_from_ml_all_predictions_ma
                            ),

                        "all_labels":
                            np.array(
                                stage_2_from_ml_all_labels_ma
                            ),
                    },
                },
            }

            # ==================================================
            # STAGE 2 A PARTIR DO STAGE 1 MAXIMUM ACCURACY
            # ==================================================

            model.load_state_dict(
                stage_1_training[
                    "max_val_accuracy_model_state"
                ]
            )

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LEARNING_RATE_STAGE_2
            )

            stage_2_from_ma_training = train_stage(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                n_epochs=N_EPOCHS_STAGE_2,
                fold=fold,
                frozen_models=None,
            )

            # TESTE: mínimo loss
            (
                stage_2_from_ma_test_loss_ml,
                stage_2_from_ma_test_accuracy_ml,
                stage_2_from_ma_all_labels_ml,
                stage_2_from_ma_all_predictions_ml,
            ) = test_model(
                model,
                stage_2_from_ma_training[
                    "min_val_loss_model_state"
                ],
                test_loader,
                device,
                criterion,
            )

            stage_2_from_ma_fold_losses_ml.append(
                stage_2_from_ma_test_loss_ml
            )

            stage_2_from_ma_fold_accuracies_ml.append(
                stage_2_from_ma_test_accuracy_ml
            )

            # TESTE: máxima acurácia
            (
                stage_2_from_ma_test_loss_ma,
                stage_2_from_ma_test_accuracy_ma,
                stage_2_from_ma_all_labels_ma,
                stage_2_from_ma_all_predictions_ma,
            ) = test_model(
                model,
                stage_2_from_ma_training[
                    "max_val_accuracy_model_state"
                ],
                test_loader,
                device,
                criterion,
            )

            stage_2_from_ma_fold_losses_ma.append(
                stage_2_from_ma_test_loss_ma
            )

            stage_2_from_ma_fold_accuracies_ma.append(
                stage_2_from_ma_test_accuracy_ma
            )

            print_fold_results(
                "minimum_loss",
                fold,
                stage_2_from_ma_training[
                    "min_val_loss_epoch"
                ],
                stage_2_from_ma_training[
                    "min_val_loss"
                ],
                stage_2_from_ma_test_loss_ml,
                stage_2_from_ma_test_accuracy_ml,
                stage_2_from_ma_all_labels_ml,
                stage_2_from_ma_all_predictions_ml
            )

            print_fold_results(
                "maximum_accuracy",
                fold,
                stage_2_from_ma_training[
                    "max_val_accuracy_epoch"
                ],
                stage_2_from_ma_training[
                    "max_val_accuracy"
                ],
                stage_2_from_ma_test_loss_ma,
                stage_2_from_ma_test_accuracy_ma,
                stage_2_from_ma_all_labels_ma,
                stage_2_from_ma_all_predictions_ma
            )

            stage_2_from_ma_results = {

                "initial_checkpoint":
                    "stage_1_maximum_accuracy",

                "training_history": {
                    "train_losses_epoch":
                        stage_2_from_ma_training[
                            "train_losses_epoch"
                        ],

                    "val_losses_epoch":
                        stage_2_from_ma_training[
                            "val_losses_epoch"
                        ],

                    "train_accuracies_epoch":
                        stage_2_from_ma_training[
                            "train_accuracies_epoch"
                        ],

                    "val_accuracies_epoch":
                        stage_2_from_ma_training[
                            "val_accuracies_epoch"
                        ],
                },

                "minimum_loss": {

                    "validation": {
                        "value":
                            stage_2_from_ma_training[
                                "min_val_loss"
                            ],

                        "epoch":
                            stage_2_from_ma_training[
                                "min_val_loss_epoch"
                            ],
                    },

                    "model_state_dict":
                        copy.deepcopy(
                            stage_2_from_ma_training[
                                "min_val_loss_model_state"
                            ]
                        ),

                    "test": {
                        "loss":
                            stage_2_from_ma_test_loss_ml,

                        "accuracy":
                            stage_2_from_ma_test_accuracy_ml,

                        "all_predictions":
                            np.array(
                                stage_2_from_ma_all_predictions_ml
                            ),

                        "all_labels":
                            np.array(
                                stage_2_from_ma_all_labels_ml
                            ),
                    },
                },

                "maximum_accuracy": {

                    "validation": {
                        "value":
                            stage_2_from_ma_training[
                                "max_val_accuracy"
                            ],

                        "epoch":
                            stage_2_from_ma_training[
                                "max_val_accuracy_epoch"
                            ],

                        "loss_at_best_accuracy":
                            stage_2_from_ma_training[
                                "best_val_loss_to_max_accuracy"
                            ],
                    },

                    "model_state_dict":
                        copy.deepcopy(
                            stage_2_from_ma_training[
                                "max_val_accuracy_model_state"
                            ]
                        ),

                    "test": {
                        "loss":
                            stage_2_from_ma_test_loss_ma,

                        "accuracy":
                            stage_2_from_ma_test_accuracy_ma,

                        "all_predictions":
                            np.array(
                                stage_2_from_ma_all_predictions_ma
                            ),

                        "all_labels":
                            np.array(
                                stage_2_from_ma_all_labels_ma
                            ),
                    },
                },
            }

            # ==================================================
            # SALVAR CHECKPOINT DO FOLD
            # ==================================================

            checkpoint = {

                "fold":
                    fold,

                "subject":
                    subject,

                "fusion":
                    fusion,

                "dataset": {

                    "total_training_samples":
                        len(train_indices),

                    "total_validation_samples":
                        len(val_indices),

                    "total_test_samples":
                        len(test_indices),

                    "total_samples":
                        len(X_s1_s2_f_t),

                    "train_indices":
                        train_indices,

                    "val_indices":
                        val_indices,

                    "test_indices":
                        test_indices,
                },

                "normalization": {

                    "power_max_s1_s2_f_t":
                        power_max_s1_s2_f_t,

                    "power_max_s1_s2_t_f":
                        power_max_s1_s2_t_f,

                    "power_max_t_f_s2_s1":
                        power_max_t_f_s2_s1,

                    "power_max_t_f_s1_s2":
                        power_max_t_f_s1_s2,
                },

                "hyperparameters": {

                    "batch_size":
                        BATCH_SIZE,

                    "learning_rate_stage_1":
                        LEARNING_RATE_STAGE_1,

                    "learning_rate_stage_2":
                        LEARNING_RATE_STAGE_2,

                    "n_epochs_stage_1":
                        N_EPOCHS_STAGE_1,

                    "n_epochs_stage_2":
                        N_EPOCHS_STAGE_2,

                    "random_state":
                        RANDOM_STATE,

                    "random_state_loader":
                        RANDOM_STATE_LOADER,
                },

                "stage_1":
                    stage_1_results,

                "stage_2_from_stage_1_minimum_loss":
                    stage_2_from_ml_results,

                "stage_2_from_stage_1_maximum_accuracy":
                    stage_2_from_ma_results,
            }

            torch.save(
                checkpoint,
                SUB_DIR
                / f"{fusion}_fold_{fold}.pth"
            )

        # ======================================================
        # RESUMOS DOS FOLDS
        # ======================================================

        print_folds_summary(
            (
                f"Stage 1 - Minimum Loss - "
                f"{subject}"
            ),
            stage_1_fold_accuracies_ml,
            stage_1_fold_losses_ml
        )

        print_folds_summary(
            (
                f"Stage 1 - Maximum Accuracy - "
                f"{subject}"
            ),
            stage_1_fold_accuracies_ma,
            stage_1_fold_losses_ma
        )

        print_folds_summary(
            (
                f"Stage 2 from Stage 1 Minimum Loss "
                f"- Minimum Loss - {subject}"
            ),
            stage_2_from_ml_fold_accuracies_ml,
            stage_2_from_ml_fold_losses_ml
        )

        print_folds_summary(
            (
                f"Stage 2 from Stage 1 Minimum Loss "
                f"- Maximum Accuracy - {subject}"
            ),
            stage_2_from_ml_fold_accuracies_ma,
            stage_2_from_ml_fold_losses_ma
        )

        print_folds_summary(
            (
                f"Stage 2 from Stage 1 Maximum Accuracy "
                f"- Minimum Loss - {subject}"
            ),
            stage_2_from_ma_fold_accuracies_ml,
            stage_2_from_ma_fold_losses_ml
        )

        print_folds_summary(
            (
                f"Stage 2 from Stage 1 Maximum Accuracy "
                f"- Maximum Accuracy - {subject}"
            ),
            stage_2_from_ma_fold_accuracies_ma,
            stage_2_from_ma_fold_losses_ma
        )

        # ======================================================
        # GRÁFICOS
        # ======================================================

        fusion_individual_subject_plots(
            MODEL_DIR,
            fusion="gated_fusion",            
            subjects=[subject],
            
        )


# ==========================================================
# EXECUTAR
# ==========================================================

if __name__ == "__main__":

    # ======================================================
    # BRANCH_MODEL_STATE OPTIONS
    # ======================================================
    #
    # minimum_loss_model
    #
    # maximum_accuracy_model
    #
    # ======================================================

    execute_fusion(
        fusion="gated_fusion",
        branch_model_state = "minimum_loss_model",
        subjects=["sub-01"],
        N_EPOCHS_STAGE_1=6,
        N_EPOCHS_STAGE_2 = 2
    )
