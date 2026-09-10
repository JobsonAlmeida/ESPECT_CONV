import copy
import sys
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from torchinfo import summary

from tools import save_summary

from branch_cnn_models import Space1Space2FrequencyTimeCNN
from branch_cnn_models import Space1Space2TimeFrequencyCNN
from branch_cnn_models import TimeFrequencySpace2Space1CNN
from branch_cnn_models import TimeFrequencySpace1Space2CNN


# ==========================================================
# CONFIGURAÇÕES DE CAMINHO & IMPORTS LOCAIS
# ==========================================================

current_file = Path(__file__).resolve()

PROJECT_ROOT = current_file.parents[2]

PIPELINE_ROOT = current_file.parents[1]


if str(PIPELINE_ROOT) not in sys.path:

    sys.path.append(
        str(PIPELINE_ROOT)
    )


from individual_subject_plots import individual_subject_plots


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

N_FOLDS = 6

N_EPOCHS = 600

BATCH_SIZE = 8

LEARNING_RATE = 0.001

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
    / "stage_40_execute_branch"
)


# ==========================================================
# SUJEITOS
# ==========================================================

subjects = [
    f"sub-{i:02d}"
    for i in range(1, 11)
]


# ==========================================================
# FUNÇÃO PARA VERIFICAR OS ÍNDICES DO FOLD
# ==========================================================

def validate_fold_indices(
    train_indices,
    val_indices,
    test_indices,
    total_samples,
):

    train_set = set(
        train_indices.tolist()
    )

    val_set = set(
        val_indices.tolist()
    )

    test_set = set(
        test_indices.tolist()
    )


    # ------------------------------------------------------
    # VERIFICAR SOBREPOSIÇÃO
    # ------------------------------------------------------

    if train_set & val_set:

        raise ValueError(
            "There is overlap between "
            "training and validation indices."
        )


    if train_set & test_set:

        raise ValueError(
            "There is overlap between "
            "training and test indices."
        )


    if val_set & test_set:

        raise ValueError(
            "There is overlap between "
            "validation and test indices."
        )


    # ------------------------------------------------------
    # VERIFICAR LIMITES
    # ------------------------------------------------------

    all_indices = np.concatenate(
        (
            train_indices,
            val_indices,
            test_indices,
        )
    )


    if np.any(all_indices < 0):

        raise ValueError(
            "There are negative indices in the fold."
        )


    if np.any(all_indices >= total_samples):

        raise ValueError(
            "There are fold indices outside "
            "the dataset limits."
        )


    # ------------------------------------------------------
    # VERIFICAR DUPLICATAS DENTRO DOS CONJUNTOS
    # ------------------------------------------------------

    if len(train_set) != len(train_indices):

        raise ValueError(
            "There are duplicated training indices."
        )


    if len(val_set) != len(val_indices):

        raise ValueError(
            "There are duplicated validation indices."
        )


    if len(test_set) != len(test_indices):

        raise ValueError(
            "There are duplicated test indices."
        )


# ==========================================================
# COPIAR STATE_DICT PARA CPU
# ==========================================================

def state_dict_to_cpu(
    model_state_dict
):

    return {
        key: value.detach().cpu().clone()
        for key, value
        in model_state_dict.items()
    }


# ==========================================================
# TESTAR MODELO
# ==========================================================

def test_model(
    model,
    model_state_dict,
    test_loader,
    device,
    criterion,
):

    # ======================================================
    # CARREGAR EXATAMENTE O MODELO ESCOLHIDO NA VALIDAÇÃO
    # ======================================================

    model.load_state_dict(
        model_state_dict
    )


    model.eval()


    # ======================================================
    # VARIÁVEIS DO TESTE
    # ======================================================

    test_total_loss = 0.0

    test_correct = 0

    test_total = 0


    all_predictions = []

    all_labels = []


    # ======================================================
    # TESTE
    # ======================================================

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


            # --------------------------------------------------
            # FORWARD
            # --------------------------------------------------

            outputs = model(
                X_batch
            )


            # --------------------------------------------------
            # LOSS
            # --------------------------------------------------

            loss = criterion(
                outputs,
                y_batch
            )


            # --------------------------------------------------
            # TAMANHO REAL DO BATCH
            # --------------------------------------------------

            batch_size = y_batch.size(
                0
            )


            # --------------------------------------------------
            # LOSS TOTAL
            #
            # CrossEntropyLoss fornece a média do batch.
            # Portanto multiplicamos pelo número de amostras
            # para recuperar a soma da loss daquele batch.
            # --------------------------------------------------

            test_total_loss += (
                loss.item()
                * batch_size
            )


            # --------------------------------------------------
            # PREVISÕES
            # --------------------------------------------------

            predictions = outputs.argmax(
                dim=1
            )


            # --------------------------------------------------
            # ACERTOS
            # --------------------------------------------------

            test_correct += (
                predictions
                == y_batch
            ).sum().item()


            test_total += (
                batch_size
            )


            # --------------------------------------------------
            # SALVAR PREVISÕES E LABELS
            # --------------------------------------------------

            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
                .tolist()
            )


            all_labels.extend(
                y_batch
                .cpu()
                .numpy()
                .tolist()
            )


    # ======================================================
    # RESULTADOS
    # ======================================================

    if test_total == 0:

        raise ValueError(
            "The test dataset is empty."
        )


    test_loss = (
        test_total_loss
        / test_total
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


# ==========================================================
# IMPRIMIR RESULTADO DE UM FOLD
# ==========================================================

def print_fold_results(
    model_type,
    fold,
    best_epoch,
    validation_loss,
    validation_accuracy,
    test_loss,
    test_accuracy,
    all_labels,
    all_predictions,
):

    print()


    match model_type:

        case "minimum_loss":

            message = (
                "--- TEST WITH MINIMUM "
                "VALIDATION LOSS MODEL ---"
            )


        case "maximum_accuracy":

            message = (
                "--- TEST WITH MAXIMUM "
                "VALIDATION ACCURACY MODEL ---"
            )


        case _:

            raise ValueError(
                f"Invalid model_type: {model_type}"
            )


    print(
        message
    )


    print(
        f"Fold {fold}"
    )


    print(
        f"Best Epoch: "
        f"{best_epoch + 1}"
    )


    print(
        f"Validation Loss: "
        f"{validation_loss:.6f}"
    )


    print(
        f"Validation Accuracy: "
        f"{validation_accuracy:.6f}"
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
        "Test Labels:",
        np.bincount(
            np.asarray(
                all_labels,
                dtype=np.int64
            ),
            minlength=4
        )
    )


    print(
        "Test Predictions:",
        np.bincount(
            np.asarray(
                all_predictions,
                dtype=np.int64
            ),
            minlength=4
        )
    )


# ==========================================================
# IMPRIMIR RESUMO DE TODOS OS FOLDS
# ==========================================================

def print_folds_summary(
    model_type,
    subject,
    fold_accuracies,
    fold_losses,
):

    match model_type:

        case "minimum_loss":

            message = (
                "Test Summary - "
                "Minimum Validation Loss Model - "
                f"{subject}"
            )


        case "maximum_accuracy":

            message = (
                "Test Summary - "
                "Maximum Validation Accuracy Model - "
                f"{subject}"
            )


        case _:

            raise ValueError(
                f"Invalid model_type: {model_type}"
            )


    fold_accuracies = np.asarray(
        fold_accuracies,
        dtype=np.float64
    )


    fold_losses = np.asarray(
        fold_losses,
        dtype=np.float64
    )


    print()

    print(
        "=" * 70
    )

    print(
        message
    )

    print(
        "=" * 70
    )


    for fold, (
        accuracy,
        loss
    ) in enumerate(
        zip(
            fold_accuracies,
            fold_losses
        ),
        start=1
    ):

        print(
            f"Fold {fold}: "
            f"Accuracy = {accuracy:.4f} | "
            f"Loss = {loss:.6f}"
        )


    print()


    print(
        "Mean Accuracy:",
        fold_accuracies.mean()
    )


    print(
        "Accuracy Standard Deviation:",
        fold_accuracies.std()
    )


    print(
        "Mean Test Loss:",
        fold_losses.mean()
    )


    print(
        "Test Loss Standard Deviation:",
        fold_losses.std()
    )


# ==========================================================
# FUNÇÃO PRINCIPAL
# ==========================================================

def execute_branch(
    branch,
    subjects=subjects,
    N_FOLDS=N_FOLDS,
    N_EPOCHS=N_EPOCHS,
    BATCH_SIZE=BATCH_SIZE,
    LEARNING_RATE=LEARNING_RATE,
    RANDOM_STATE=RANDOM_STATE,
    RANDOM_STATE_LOADER=RANDOM_STATE_LOADER,
):

    # ======================================================
    # SEED GLOBAL
    # ======================================================

    torch.manual_seed(
        RANDOM_STATE
    )


    # ======================================================
    # LOOP DOS SUJEITOS
    # ======================================================

    for subject in subjects:

        print()

        print(
            "#" * 70
        )

        print(
            f"SUJEITO: {subject}"
        )

        print(
            "#" * 70
        )


        # ==================================================
        # CARREGAR OS DADOS
        # ==================================================

        power_array = np.load(
            INPUT_DIR
            / f"{subject}_power.npy"
        )


        labels = np.load(
            INPUT_DIR
            / f"{subject}_labels.npy"
        )


        if len(power_array) != len(labels):

            raise ValueError(
                f"Number of samples in power_array "
                f"({len(power_array)}) differs from "
                f"number of labels ({len(labels)})."
            )


        print(
            "Power:",
            power_array.shape
        )


        print(
            "Labels:",
            labels.shape
        )


        # ==================================================
        # DIRETÓRIO DOS MODELOS
        # ==================================================

        MODEL_DIR = (
            OUTPUT_DIR
            / f"{branch}_branch"
        )


        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        SUB_DIR = (
            MODEL_DIR
            / subject
        )


        SUB_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        # ==================================================
        # DIRETÓRIO DOS ÍNDICES DOS FOLDS
        # ==================================================

        FOLDS_DIR = (
            OUTPUT_DIR
            / "fold_indices"
            / subject
        )


        if not FOLDS_DIR.exists():

            raise FileNotFoundError(
                f"Fold directory does not exist: "
                f"{FOLDS_DIR}"
            )


        # ==================================================
        # REORGANIZAR AS DIMENSÕES
        # ==================================================

        match branch:

            # ==================================================
            # C = time
            #
            # Conv3d:
            # (N, C, D, H, W)
            #
            # Original:
            # (
            #   epoch,
            #   space2,
            #   space1,
            #   frequency,
            #   time
            # )
            #
            # Final:
            # (
            #   epoch,
            #   time,
            #   frequency,
            #   space2,
            #   space1
            # )
            # ==================================================

            case "space1_space2_frequency_time":

                power_array = np.transpose(
                    power_array,
                    (
                        0,
                        4,
                        3,
                        1,
                        2
                    )
                )


            # ==================================================
            # C = frequency
            #
            # Final:
            # (
            #   epoch,
            #   frequency,
            #   time,
            #   space2,
            #   space1
            # )
            # ==================================================

            case "space1_space2_time_frequency":

                power_array = np.transpose(
                    power_array,
                    (
                        0,
                        3,
                        4,
                        1,
                        2
                    )
                )


            # ==================================================
            # C = space1
            #
            # Final:
            # (
            #   epoch,
            #   space1,
            #   space2,
            #   frequency,
            #   time
            # )
            # ==================================================

            case "time_frequency_space2_space1":

                power_array = np.transpose(
                    power_array,
                    (
                        0,
                        2,
                        1,
                        3,
                        4
                    )
                )


            # ==================================================
            # C = space2
            #
            # Final:
            # (
            #   epoch,
            #   space2,
            #   space1,
            #   frequency,
            #   time
            # )
            # ==================================================

            case "time_frequency_space1_space2":

                power_array = np.transpose(
                    power_array,
                    (
                        0,
                        1,
                        2,
                        3,
                        4
                    )
                )


            case _:

                raise ValueError(
                    f"Invalid branch: {branch}"
                )


        print(
            "Power reorganizado:",
            power_array.shape
        )


        # ==================================================
        # CONVERTER PARA TENSOR
        # ==================================================

        X = torch.from_numpy(
            power_array
        ).float()

        y = torch.from_numpy(
            labels
        ).long()

        # obtendo o size do tensor
        X_input_size = [
            (1, *X.shape[1:]),
        ]

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
            X.min().item()
        )


        print(
            "Max:",
            X.max().item()
        )


        print(
            "Mean:",
            X.mean().item()
        )


        print(
            "Std:",
            X.std().item()
        )


        nonzero_values = (
            torch.count_nonzero(X)
            .item()
        )


        print(
            "Valores diferentes de zero:",
            nonzero_values
        )


        print(
            "Proporção diferente de zero:",
            nonzero_values
            / X.numel()
        )


        # ==================================================
        # CPU OU GPU
        # ==================================================

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
                torch.cuda.get_device_name(
                    0
                )
            )


        # ==================================================
        # RESULTADOS DE TODOS OS FOLDS
        # ==================================================

        fold_accuracies_ml = []

        fold_losses_ml = []


        fold_accuracies_ma = []

        fold_losses_ma = []


        # ==================================================
        # LOOP DOS FOLDS
        # ==================================================

        for fold in range(
            1,
            N_FOLDS + 1
        ):

            print()

            print(
                "=" * 70
            )

            print(
                f"FOLD {fold}/{N_FOLDS}"
            )

            print(
                "=" * 70
            )


            # ==============================================
            # CARREGAR ÍNDICES
            # ==============================================

            fold_file = (
                FOLDS_DIR
                / f"fold_{fold}.npz"
            )


            if not fold_file.exists():

                raise FileNotFoundError(
                    f"Fold file does not exist: "
                    f"{fold_file}"
                )


            fold_data = np.load(
                fold_file
            )


            train_indices = fold_data[
                "train_indices"
            ]


            val_indices = fold_data[
                "val_indices"
            ]


            test_indices = fold_data[
                "test_indices"
            ]


            # ==============================================
            # VERIFICAR ÍNDICES
            # ==============================================

            validate_fold_indices(
                train_indices,
                val_indices,
                test_indices,
                len(X),
            )


            # ==============================================
            # MOSTRAR TAMANHOS
            # ==============================================

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


            # ==============================================
            # SEPARAR OS DADOS
            # ==============================================

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


            # ==============================================
            # VERIFICAR CONJUNTOS VAZIOS
            # ==============================================

            if len(X_train) == 0:

                raise ValueError(
                    f"Training set is empty "
                    f"in fold {fold}."
                )


            if len(X_val) == 0:

                raise ValueError(
                    f"Validation set is empty "
                    f"in fold {fold}."
                )


            if len(X_test) == 0:

                raise ValueError(
                    f"Test set is empty "
                    f"in fold {fold}."
                )


            # ==============================================
            # NORMALIZAÇÃO
            #
            # IMPORTANTE:
            #
            # power_max é calculado EXCLUSIVAMENTE
            # usando o conjunto de treinamento.
            #
            # Validação e teste nunca participam
            # da estimação da normalização.
            # ==============================================

            power_max = X_train.max()


            if (
                not torch.isfinite(
                    power_max
                )
            ):

                raise ValueError(
                    f"power_max is not finite "
                    f"in fold {fold}: "
                    f"{power_max.item()}"
                )


            if power_max.item() <= 0:

                raise ValueError(
                    f"power_max must be > 0. "
                    f"Fold {fold}: "
                    f"{power_max.item()}"
                )


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
                "Power max:",
                power_max.item()
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


            # ==============================================
            # DATASETS
            # ==============================================

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


            # ==============================================
            # DATALOADERS
            # ==============================================

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


            # ==============================================
            # NOVA INICIALIZAÇÃO DO MODELO PARA O FOLD
            # ==============================================

            torch.manual_seed(
                RANDOM_STATE
            )


            # ==============================================
            # SELECIONAR MODELO
            # ==============================================

            match branch:

                # ------------------------------------------

                case "space1_space2_frequency_time":

                    model = (
                        Space1Space2FrequencyTimeCNN()
                        .to(device)
                    )

                # ------------------------------------------

                case "space1_space2_time_frequency":

                    model = (
                        Space1Space2TimeFrequencyCNN()
                        .to(device)
                    )


                # ------------------------------------------

                case "time_frequency_space2_space1":

                    model = (
                        TimeFrequencySpace2Space1CNN()
                        .to(device)
                    )


                # ------------------------------------------

                case "time_frequency_space1_space2":

                    model = (
                        TimeFrequencySpace1Space2CNN()
                        .to(device)
                    )

                case _:

                    raise ValueError(
                        f"Invalid branch: {branch}"
                    )


            # ==============================================
            # FUNÇÃO DE PERDA
            # ==============================================

            criterion = nn.CrossEntropyLoss()


            # ==============================================
            # OTIMIZADOR
            # ==============================================

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LEARNING_RATE
            )


            # ==============================================
            # HISTÓRICO DO FOLD
            # ==============================================

            train_accuracies_epoch = []

            val_accuracies_epoch = []


            train_losses_epoch = []

            val_losses_epoch = []


            # ==============================================
            # MODELO DE MENOR VALIDATION LOSS
            # ==============================================

            min_val_loss = float(
                "inf"
            )


            min_val_loss_epoch = None


            val_accuracy_at_min_loss = None


            min_val_loss_model_state = None


            # ==============================================
            # MODELO DE MAIOR VALIDATION ACCURACY
            # ==============================================

            max_val_accuracy = -1.0


            max_val_accuracy_epoch = None


            val_loss_at_max_accuracy = float(
                "inf"
            )


            max_val_accuracy_model_state = None


            # ==============================================
            # TREINAMENTO
            # ==============================================

            if fold == 1:

                save_summary(
                    model,
                    branch,
                    X_input_size,
                    device,
                    MODEL_DIR
                )

            for epoch in range(
                N_EPOCHS
            ):

                # ==========================================
                # TREINAMENTO DA ÉPOCA
                # ==========================================

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


                    # --------------------------------------
                    # ZERAR GRADIENTES
                    # --------------------------------------

                    optimizer.zero_grad()


                    # --------------------------------------
                    # FORWARD
                    # --------------------------------------

                    outputs = model(
                        X_batch
                    )


                    # --------------------------------------
                    # LOSS
                    # --------------------------------------

                    loss = criterion(
                        outputs,
                        y_batch
                    )


                    # --------------------------------------
                    # BACKPROPAGATION
                    # --------------------------------------

                    loss.backward()


                    # --------------------------------------
                    # ATUALIZAR PESOS
                    # --------------------------------------

                    optimizer.step()


                    # --------------------------------------
                    # TAMANHO REAL DO BATCH
                    # --------------------------------------

                    batch_size = y_batch.size(
                        0
                    )


                    # --------------------------------------
                    # ACUMULAR LOSS POR AMOSTRA
                    # --------------------------------------

                    train_total_loss += (
                        loss.item()
                        * batch_size
                    )


                    # --------------------------------------
                    # PREVISÕES
                    # --------------------------------------

                    predictions = (
                        outputs.argmax(
                            dim=1
                        )
                    )


                    # --------------------------------------
                    # ACERTOS
                    # --------------------------------------

                    train_correct += (
                        predictions
                        == y_batch
                    ).sum().item()


                    train_total += (
                        batch_size
                    )


                # ==========================================
                # RESULTADOS DO TREINAMENTO
                # ==========================================

                train_loss = (
                    train_total_loss
                    / train_total
                )


                train_accuracy = (
                    train_correct
                    / train_total
                )


                # ==========================================
                # VALIDAÇÃO DA ÉPOCA
                # ==========================================

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


                        # ----------------------------------
                        # FORWARD
                        # ----------------------------------

                        outputs = model(
                            X_batch
                        )


                        # ----------------------------------
                        # LOSS
                        # ----------------------------------

                        loss = criterion(
                            outputs,
                            y_batch
                        )


                        # ----------------------------------
                        # TAMANHO DO BATCH
                        # ----------------------------------

                        batch_size = y_batch.size(
                            0
                        )


                        # ----------------------------------
                        # LOSS TOTAL
                        # ----------------------------------

                        val_total_loss += (
                            loss.item()
                            * batch_size
                        )


                        # ----------------------------------
                        # PREVISÕES
                        # ----------------------------------

                        predictions = (
                            outputs.argmax(
                                dim=1
                            )
                        )


                        # ----------------------------------
                        # ACERTOS
                        # ----------------------------------

                        val_correct += (
                            predictions
                            == y_batch
                        ).sum().item()


                        val_total += (
                            batch_size
                        )


                # ==========================================
                # RESULTADOS DA VALIDAÇÃO
                # ==========================================

                val_loss = (
                    val_total_loss
                    / val_total
                )


                val_accuracy = (
                    val_correct
                    / val_total
                )


                # ==========================================
                # GUARDAR HISTÓRICO
                # ==========================================

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


                # ==========================================
                # MENOR VALIDATION LOSS
                # ==========================================

                if val_loss < min_val_loss:

                    min_val_loss = (
                        val_loss
                    )


                    min_val_loss_epoch = (
                        epoch
                    )


                    val_accuracy_at_min_loss = (
                        val_accuracy
                    )


                    min_val_loss_model_state = (
                        copy.deepcopy(
                            model.state_dict()
                        )
                    )


                # ==========================================
                # MAIOR VALIDATION ACCURACY
                #
                # Em caso de empate na acurácia,
                # escolhemos o modelo com menor
                # validation loss.
                # ==========================================

                if (
                    val_accuracy
                    > max_val_accuracy
                    or
                    (
                        val_accuracy
                        == max_val_accuracy
                        and
                        val_loss
                        < val_loss_at_max_accuracy
                    )
                ):

                    max_val_accuracy = (
                        val_accuracy
                    )


                    max_val_accuracy_epoch = (
                        epoch
                    )


                    val_loss_at_max_accuracy = (
                        val_loss
                    )


                    max_val_accuracy_model_state = (
                        copy.deepcopy(
                            model.state_dict()
                        )
                    )


                # ==========================================
                # MOSTRAR RESULTADOS
                # ==========================================

                print(
                    f"Fold {fold} | "
                    f"Epoch "
                    f"{epoch + 1:03d}/{N_EPOCHS} | "
                    f"Train Loss: "
                    f"{train_loss:.4f} | "
                    f"Val Loss: "
                    f"{val_loss:.4f} | "
                    f"Train Acc: "
                    f"{train_accuracy:.4f} | "
                    f"Val Acc: "
                    f"{val_accuracy:.4f}"
                )


            # ==============================================
            # VERIFICAR CHECKPOINTS
            # ==============================================

            if min_val_loss_model_state is None:

                raise RuntimeError(
                    "Minimum validation loss model "
                    "was not saved."
                )


            if max_val_accuracy_model_state is None:

                raise RuntimeError(
                    "Maximum validation accuracy model "
                    "was not saved."
                )


            # ==============================================
            # TESTAR MODELO DE MENOR VALIDATION LOSS
            # ==============================================

            (
                test_loss_ml,
                test_accuracy_ml,
                all_labels_ml,
                all_predictions_ml,
            ) = test_model(
                model=model,
                model_state_dict=(
                    min_val_loss_model_state
                ),
                test_loader=test_loader,
                device=device,
                criterion=criterion,
            )


            fold_losses_ml.append(
                test_loss_ml
            )


            fold_accuracies_ml.append(
                test_accuracy_ml
            )


            # ==============================================
            # TESTAR MODELO DE MAIOR VALIDATION ACCURACY
            # ==============================================

            (
                test_loss_ma,
                test_accuracy_ma,
                all_labels_ma,
                all_predictions_ma,
            ) = test_model(
                model=model,
                model_state_dict=(
                    max_val_accuracy_model_state
                ),
                test_loader=test_loader,
                device=device,
                criterion=criterion,
            )


            fold_losses_ma.append(
                test_loss_ma
            )


            fold_accuracies_ma.append(
                test_accuracy_ma
            )


            # ==============================================
            # IMPRIMIR RESULTADOS DO MODELO DE MENOR LOSS
            # ==============================================

            print_fold_results(
                model_type="minimum_loss",
                fold=fold,
                best_epoch=min_val_loss_epoch,
                validation_loss=min_val_loss,
                validation_accuracy=(
                    val_accuracy_at_min_loss
                ),
                test_loss=test_loss_ml,
                test_accuracy=test_accuracy_ml,
                all_labels=all_labels_ml,
                all_predictions=(
                    all_predictions_ml
                ),
            )


            # ==============================================
            # IMPRIMIR RESULTADOS DO MODELO DE MAIOR ACC
            # ==============================================

            print_fold_results(
                model_type="maximum_accuracy",
                fold=fold,
                best_epoch=(
                    max_val_accuracy_epoch
                ),
                validation_loss=(
                    val_loss_at_max_accuracy
                ),
                validation_accuracy=(
                    max_val_accuracy
                ),
                test_loss=test_loss_ma,
                test_accuracy=test_accuracy_ma,
                all_labels=all_labels_ma,
                all_predictions=(
                    all_predictions_ma
                ),
            )


            # ==============================================
            # CONVERTER OS DOIS CHECKPOINTS PARA CPU
            #
            # IMPORTANTE:
            #
            # Não usamos model.state_dict() aqui.
            #
            # Isso evita que o segundo teste sobrescreva
            # os pesos associados ao primeiro checkpoint.
            # ==============================================

            min_val_loss_model_state_cpu = (
                state_dict_to_cpu(
                    min_val_loss_model_state
                )
            )


            max_val_accuracy_model_state_cpu = (
                state_dict_to_cpu(
                    max_val_accuracy_model_state
                )
            )


            # ==============================================
            # SALVAR CHECKPOINT DO FOLD
            # ==============================================

            torch.save(
                {

                    # ======================================
                    # INFORMAÇÕES GERAIS
                    # ======================================

                    "branch":
                        branch,

                    "subject":
                        subject,

                    "fold":
                        fold,

                    "n_epochs":
                        N_EPOCHS,

                    "batch_size":
                        BATCH_SIZE,

                    "learning_rate":
                        LEARNING_RATE,

                    "random_state":
                        RANDOM_STATE,

                    "random_state_loader":
                        RANDOM_STATE_LOADER,


                    # ======================================
                    # NÚMERO DE AMOSTRAS
                    # ======================================

                    "total_training_samples":
                        len(train_indices),

                    "total_validation_samples":
                        len(val_indices),

                    "total_test_samples":
                        len(test_indices),

                    "total_samples":
                        len(X),


                    # ======================================
                    # NORMALIZAÇÃO
                    # ======================================

                    "power_max":
                        power_max.item(),


                    # ======================================
                    # ÍNDICES
                    # ======================================

                    "train_indices":
                        train_indices,

                    "val_indices":
                        val_indices,

                    "test_indices":
                        test_indices,


                    # ======================================
                    # HISTÓRICO DE TREINAMENTO
                    # ======================================

                    "train_losses_epoch":
                        train_losses_epoch,

                    "val_losses_epoch":
                        val_losses_epoch,

                    "train_accuracies_epoch":
                        train_accuracies_epoch,

                    "val_accuracies_epoch":
                        val_accuracies_epoch,


                    # ======================================
                    # MODELO DE MENOR VALIDATION LOSS
                    # ======================================

                    "minimum_loss_model": {

                        # ------------------------------
                        # CRITÉRIO DE SELEÇÃO
                        # ------------------------------

                        "validation_loss":
                            min_val_loss,

                        "validation_accuracy":
                            val_accuracy_at_min_loss,

                        "epoch":
                            min_val_loss_epoch,


                        # ------------------------------
                        # PESOS
                        # ------------------------------

                        "model_state_dict":
                            min_val_loss_model_state_cpu,


                        # ------------------------------
                        # RESULTADOS NO TESTE
                        # ------------------------------

                        "test_loss":
                            test_loss_ml,

                        "test_accuracy":
                            test_accuracy_ml,

                        "all_predictions":
                            np.asarray(
                                all_predictions_ml,
                                dtype=np.int64
                            ),

                        "all_labels":
                            np.asarray(
                                all_labels_ml,
                                dtype=np.int64
                            ),
                    },


                    # ======================================
                    # MODELO DE MAIOR VALIDATION ACCURACY
                    # ======================================

                    "maximum_accuracy_model": {

                        # ------------------------------
                        # CRITÉRIO DE SELEÇÃO
                        # ------------------------------

                        "validation_accuracy":
                            max_val_accuracy,

                        "validation_loss":
                            val_loss_at_max_accuracy,

                        "epoch":
                            max_val_accuracy_epoch,


                        # ------------------------------
                        # PESOS
                        # ------------------------------

                        "model_state_dict":
                            max_val_accuracy_model_state_cpu,


                        # ------------------------------
                        # RESULTADOS NO TESTE
                        # ------------------------------

                        "test_loss":
                            test_loss_ma,

                        "test_accuracy":
                            test_accuracy_ma,

                        "all_predictions":
                            np.asarray(
                                all_predictions_ma,
                                dtype=np.int64
                            ),

                        "all_labels":
                            np.asarray(
                                all_labels_ma,
                                dtype=np.int64
                            ),
                    },

                },

                SUB_DIR
                / f"{branch}_fold_{fold}.pth"
            )


            print(
                f"\nCheckpoint saved: "
                f"{SUB_DIR / f'{branch}_fold_{fold}.pth'}"
            )


            # ==============================================
            # LIBERAR REFERÊNCIAS DOS CHECKPOINTS DO FOLD
            # ==============================================

            del min_val_loss_model_state

            del max_val_accuracy_model_state


            if torch.cuda.is_available():

                torch.cuda.empty_cache()


        # ==================================================
        # RESUMO DOS FOLDS
        # ==================================================

        print_folds_summary(
            model_type="minimum_loss",
            subject=subject,
            fold_accuracies=(
                fold_accuracies_ml
            ),
            fold_losses=(
                fold_losses_ml
            ),
        )


        print_folds_summary(
            model_type="maximum_accuracy",
            subject=subject,
            fold_accuracies=(
                fold_accuracies_ma
            ),
            fold_losses=(
                fold_losses_ma
            ),
        )


        # ==================================================
        # SALVAR RESUMO DOS FOLDS
        # ==================================================

        fold_accuracies_ml_array = (
            np.asarray(
                fold_accuracies_ml,
                dtype=np.float64
            )
        )


        fold_losses_ml_array = (
            np.asarray(
                fold_losses_ml,
                dtype=np.float64
            )
        )


        fold_accuracies_ma_array = (
            np.asarray(
                fold_accuracies_ma,
                dtype=np.float64
            )
        )


        fold_losses_ma_array = (
            np.asarray(
                fold_losses_ma,
                dtype=np.float64
            )
        )


        summary_results = {

            "branch":
                branch,

            "subject":
                subject,

            "n_folds":
                N_FOLDS,


            # ==============================================
            # MODELOS ESCOLHIDOS POR MENOR LOSS
            # ==============================================

            "minimum_loss_model": {

                "fold_accuracies":
                    fold_accuracies_ml_array,

                "fold_losses":
                    fold_losses_ml_array,

                "mean_accuracy":
                    fold_accuracies_ml_array.mean(),

                "std_accuracy":
                    fold_accuracies_ml_array.std(),

                "mean_loss":
                    fold_losses_ml_array.mean(),

                "std_loss":
                    fold_losses_ml_array.std(),
            },


            # ==============================================
            # MODELOS ESCOLHIDOS POR MAIOR ACURÁCIA
            # ==============================================

            "maximum_accuracy_model": {

                "fold_accuracies":
                    fold_accuracies_ma_array,

                "fold_losses":
                    fold_losses_ma_array,

                "mean_accuracy":
                    fold_accuracies_ma_array.mean(),

                "std_accuracy":
                    fold_accuracies_ma_array.std(),

                "mean_loss":
                    fold_losses_ma_array.mean(),

                "std_loss":
                    fold_losses_ma_array.std(),
            },
        }


        torch.save(
            summary_results,

            SUB_DIR
            / f"{branch}_summary.pth"
        )


        print(
            f"\nSummary saved: "
            f"{SUB_DIR / f'{branch}_summary.pth'}"
        )


        # ==================================================
        # SALVAR GRÁFICOS
        # ==================================================

        individual_subject_plots(
            branch=branch,
            subjects=subject
        )


# ==========================================================
# EXECUTAR
# ==========================================================

if __name__ == "__main__":

    # ======================================================
    # BRANCH OPTIONS
    # ======================================================
    #
    # space1_space2_frequency_time
    #
    # space1_space2_time_frequency
    #
    # time_frequency_space2_space1
    #
    # time_frequency_space1_space2
    #
    # ======================================================

    execute_branch(
        branch="space1_space2_frequency_time",
        N_EPOCHS=12,       
        subjects=[
            "sub-01"
        ]
    )