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

from torchinfo import summary

from tools import save_summary_as_pdf

from branch_cnn_models import Space1Space2FrequencyTimeCNN
from branch_cnn_models import Space1Space2TimeFrequencyCNN
from branch_cnn_models import TimeFrequencySpace2Space1CNN
from branch_cnn_models import TimeFrequencySpace1Space2CNN

# ==========================================================
# CONFIGURAÇÕES DE CAMINHO & IMPORTS LOCAIS
# ==========================================================

current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[2]
PIPELINE_ROOT = current_file.parents[1]  # Caminho até a pasta 'pipeline'

# 2. Adiciona a pasta 'pipeline' ao sys.path para o Python achar a 'stage41'
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.append(str(PIPELINE_ROOT))

# 3. Agora o import funciona tanto no terminal quanto no debugger do VS Code
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




def test_model(model, model_state_dict, test_loader, device, criterion, fold_accuracies, fold_losses):

    # ==================================================
    # TESTE COM O MODELO ESCOLHIDO
    # ==================================================

    # O conjunto de teste aparece SOMENTE agora,
    # depois que o modelo já foi escolhido.

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


    fold_accuracies.append( #acurácias por fold
        test_accuracy
    )


    fold_losses.append(
        test_loss
    )

    return model, test_loss, test_accuracy, fold_losses, fold_accuracies, all_labels, all_predictions



def print_fold_results(model_type, fold, best_epoch, focus_property, test_loss, test_accuracy, all_labels, all_predictions):

    print()

    match model_type:
        case "minimum_loss":
            message = "--- TEST WITH MINIMUM LOSS ---"
            property_message =  f"Minimum Validation Loss: {focus_property:.6f}"

        case "maximum_accuracy":
            message = "--- TEST WITH MAXIMUM ACCURACY ---"
            property_message =  f"Maximum Accuracy: {focus_property:.6f}"

        case _ : 
            raise ValueError(f"Invalid model_type {model_type}")



    print(message)

    print(
        f"Fold {fold}"
    )

    print(f"Best Epoch: {best_epoch + 1}")


    print(property_message)


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
            all_labels,
            minlength=4
        )
    )

    print(
        "Test Predictions:",
        np.bincount(
            all_predictions,
            minlength=4
        )
    )





def print_folds_summary(model_type, subject, fold_accuracies, fold_losses, ):

    match model_type:
        case "minimum_loss":
            message = f"Test Summary - Lowest Loss Model - {subject}"

        case "maximum_accuracy":
            message = f"Test Summary - Maximum Accuracy Model - {subject}"

        case _:
             raise ValueError(f"Invalid model_type {model_type}")

    fold_accuracies = np.array(
        fold_accuracies
    )

    fold_losses = np.array(
        fold_losses
    )


    print()
    print("=" * 70)
    print(message)
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
# MAIN FUNCTION
# ==========================================================

subjects = [f"sub-{i:02d}" for i in range(1, 11)]

def execute_branch(
        branch,
        subjects = subjects,
        N_FOLDS = N_FOLDS,
        N_EPOCHS = N_EPOCHS,
        BATCH_SIZE = BATCH_SIZE,
        LEARNING_RATE = LEARNING_RATE,
        RANDOM_STATE = RANDOM_STATE,
        RANDOM_STATE_LOADER = RANDOM_STATE_LOADER,
):

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
        # REORGANIZAR AS DIMENSÕES
        # ======================================================

        match branch:

            case "space1_space2_frequency_time":

                # Before:
                # (epoch, row (pos_y /space_2), column (pos_x / space_1), frequency, time)

                # After:
                # N = Number → número de amostras no batch
                # C = Channels → número de canais
                # D = Depth → profundidade
                # H = Height → altura
                # W = Width → largura
                # (epoch, time, frequency, space2, space1)

                power_array = np.transpose(
                            power_array,
                            (0, 4, 3, 1, 2)
                ) 
                               
            case "space1_space2_time_frequency":

                # Before:
                # (epoch, row (pos_y /space_2), column (pos_x / space_1), frequency, time)

                # After:
                # N = Number → número de amostras no batch
                # C = Channels → número de canais
                # D = Depth → profundidade
                # H = Height → altura
                # W = Width → largura
                # (epoch, frequency, time, space2, space1)

                power_array = np.transpose(
                            power_array,
                            (0, 3, 4, 1, 2)
                ) 

            case "time_frequency_space2_space1":

                # Before:
                # (epoch, row (pos_y /space_2), column (pos_x / space_1), frequency, time)

                # After:
                # N = Number → número de amostras no batch
                # C = Channels → número de canais
                # D = Depth → profundidade
                # H = Height → altura
                # W = Width → largura
                # (epoch, space1, space2,  frequency, time, )

                power_array = np.transpose(
                            power_array,
                            (0, 2, 1, 3, 4)
                ) 

            case "time_frequency_space1_space2":

                # Before:
                # (epoch, row (pos_y /space_2), column (pos_x / space_1), frequency, time)

                # After:
                # N = Number → número de amostras no batch
                # C = Channels → número de canais
                # D = Depth → profundidade
                # H = Height → altura
                # W = Width → largura
                # (epoch, space2, space1, frequency, time, )

                power_array = np.transpose(
                            power_array,
                            (0, 1, 2, 3, 4)
                ) 

            case _:
                raise ValueError(f"Invalid branch {branch}")

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
        # ARMAZENAR RESULTADOS DOS FOLDS
        # ======================================================

        fold_accuracies_ml = []
        fold_losses_ml = []

        fold_accuracies_ma = []
        fold_losses_ma = []


        # ======================================================
        # LOOP DOS 5 FOLDS
        # ======================================================

        #for fold, (development_indices, test_indices ) in enumerate( skf.split( X, labels), start=1 ):
        for fold in range(1, N_FOLDS + 1):

            print()
            print("=" * 70)
            print(
                f"FOLD {fold}/{N_FOLDS}"
            )
            print("=" * 70)

            # ======================================================
            # CARREGAR OS ÍNDICES DESTE FOLD
            # ======================================================

            fold_data = np.load(
                FOLDS_DIR / f"fold_{fold}.npz"
            )

            train_indices = fold_data["train_indices"]
            test_indices = fold_data["test_indices"]
            val_indices = fold_data["val_indices"]

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


            # ===============================================
            # SELECIONA O MODELO
            # ===============================================
            match branch:

                case "space1_space2_frequency_time":

                    model = Space1Space2FrequencyTimeCNN().to(device)

                    if fold == 1: 
                        model_stats = summary(model, input_size=(1, 9, 65, 21, 21), device=device, verbose=0)
                        save_summary_as_pdf(branch , model_stats, save_path= MODEL_DIR)

                                
                case "space1_space2_time_frequency":

                    model = Space1Space2TimeFrequencyCNN().to(device)

                    if fold == 1:
                        model_stats = summary(model, input_size=(1, 65, 9, 21, 21), device=device, verbose=0)
                        save_summary_as_pdf(branch , model_stats, save_path= MODEL_DIR)

                case "time_frequency_space2_space1":
                                        
                    model = TimeFrequencySpace2Space1CNN().to(device)

                    if fold == 1:
                        model_stats = summary(model, input_size=(1, 21, 21, 65, 9), device=device, verbose=0)
                        save_summary_as_pdf(branch , model_stats, save_path= MODEL_DIR)

                case "time_frequency_space1_space2":

                    model = TimeFrequencySpace1Space2CNN().to(device)

                    if fold == 1:
                        model_stats = summary(model, input_size=(1, 21, 21, 65, 9), device=device, verbose=0)
                        save_summary_as_pdf(branch , model_stats, save_path= MODEL_DIR)

                case _:
                    raise ValueError(f"Invalid branch {branch}")




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
            # INICIALIZAÇÃO PARA O MELHOR MODELO
            # ==================================================

            # menor loss
            min_val_loss = float(
                "inf"
            )
            min_val_loss_epoch = 0
            min_val_loss_model_state = None

            # máxima acurácia
            max_val_accuracy = -1
            best_val_loss_to_max_accuracy = float("inf")
            max_val_accuracy_epoch = 0
            max_val_accuracy_model_state = None

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

                # menor loss
                if val_loss < min_val_loss:

                    min_val_loss = (
                        val_loss
                    )


                    min_val_loss_epoch = (
                        epoch
                    )

                    min_val_loss_model_state = copy.deepcopy(
                        model.state_dict()
                    )

                #maior acurácia
                if (val_accuracy > max_val_accuracy or
                     ( val_accuracy == max_val_accuracy and val_loss < best_val_loss_to_max_accuracy)
                    ):
                    
                    max_val_accuracy = val_accuracy
                    best_val_loss_to_max_accuracy = val_loss
                    max_val_accuracy_epoch = epoch

                    max_val_accuracy_model_state = copy.deepcopy(
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


            # ==============================
            # TESTE DO MODELO DE MENOR LOSS
            # ==============================

            model_ml, test_loss_ml, test_accuracy_ml, fold_losses_ml, fold_accuracies_ml, all_labels_ml, all_predictions_ml = test_model(model, min_val_loss_model_state,
                test_loader,
                device,
                criterion,
                fold_accuracies_ml,
                fold_losses_ml,
            )

            # ===================================
            # TESTE DO MODELO DE MAIOR ACURÁCIA
            # ===================================

            model_ma, test_loss_ma, test_accuracy_ma, fold_losses_ma, fold_accuracies_ma, all_labels_ma, all_predictions_ma = test_model(model, max_val_accuracy_model_state,
                test_loader,
                device,
                criterion,
                fold_accuracies_ma,
                fold_losses_ma,
            )

            # =============================================================
            # IMPRIMINDO OS RESULTADOS DE TESTE DE CADA MODELO POR FOLD 
            # =============================================================

            print_fold_results(
                "minimum_loss", 
                fold, 
                min_val_loss_epoch, 
                min_val_loss, 
                test_loss_ml,
                test_accuracy_ml, 
                all_labels_ml, 
                all_predictions_ml )

            print_fold_results(
                "maximum_accuracy", 
                fold, 
                max_val_accuracy_epoch, 
                max_val_accuracy, 
                test_loss_ma,
                test_accuracy_ma, 
                all_labels_ma, 
                all_predictions_ma )

            

            # ==================================================
            # SALVAR O CHECKPOINT DO FOLD
            # ==================================================

            torch.save(
                { 

                    # =============================
                    # DADOS DO FOLD 
                    # =============================

                    # ------------------------------------------
                    # FOLD
                    # ------------------------------------------

                    "fold":
                        fold,

                    # ------------------------------------------
                    # NÚMERO DE AMOSTRAS DO FOLD
                    #-------------------------------------------

                    "total_training_samples": len(train_indices),

                    "total_validation_samples": len(val_indices),

                    "total_test_samples":  len(test_indices),

                    "total_samples":  len(X),


                    # ------------------------------------------
                    # NORMALIZAÇÃO
                    # ------------------------------------------

                    "power_max":
                        power_max.item(),
                  
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
                    # HISTÓRICO DAS LOSSES
                    # ------------------------------------------

                    "train_losses_epoch":
                        train_losses_epoch,

                    "val_losses_epoch":
                        val_losses_epoch,

                    "min_val_loss":
                        min_val_loss,

                    "min_val_loss_epoch":
                        min_val_loss_epoch,

                    # ------------------------------------------
                    # HISTÓRICO DAS ACURÁCIAS
                    # ------------------------------------------

                    "train_accuracies_epoch":
                        train_accuracies_epoch,

                    "val_accuracies_epoch":
                        val_accuracies_epoch,

                    "max_val_accuracy" : 
                        max_val_accuracy,

                    "max_val_accuracy_epoch": 
                        max_val_accuracy_epoch,


                    # ===================================
                    # TEST RESULTS USING MINIMUM LOSS MODEL
                    # ==================================

                    "minimum_loss_model_test_results": {

                        # ------------------------------------------
                        # MELHOR MODELO PARA O MENOR CUSTO
                        # ------------------------------------------

                        "model_state_dict":
                            model_ml.state_dict(),
                           
                        # ---------------------
                        # RESULTADOS
                        # ----------------------
                        
                        "test_loss":
                            test_loss_ml,

                        "test_accuracy":
                            test_accuracy_ml,

                        "all_predictions":
                            np.array(
                                all_predictions_ml
                            ),

                        "all_labels":
                            np.array(
                                all_labels_ml
                            ),

                        "fold_accuracies": np.array(fold_accuracies_ml),

                        "fold_losses": np.array(fold_losses_ml),
                    },

                    # ==========================================
                    # TEST RESULTS USING MAXIMUM ACURRACY MODEL
                    # ==========================================

                    "maximum_accuracy_model_test_results": {

                        # ------------------------------------------
                        # MELHOR MODELO PARA O MENOR CUSTO
                        # ------------------------------------------

                        "model_state_dict":
                            model_ma.state_dict(),

                        # ---------------------
                        # RESULTADOS
                        # ----------------------
                        
                        "test_loss":
                            test_loss_ma,

                        "test_accuracy":
                            test_accuracy_ma,

                        "all_predictions":
                            np.array(
                                all_predictions_ma
                            ),

                        "all_labels":
                            np.array(
                                all_labels_ma
                            ),

                        "fold_accuracies": np.array(fold_accuracies_ma),

                        "fold_losses": np.array(fold_losses_ma),
                    },

                },

                SUB_DIR
                / f"{branch}_fold_{fold}.pth"
            )

            
        # ==========================================================
        # RESULTADOS DOS 5 FOLDS
        # ==========================================================

        print_folds_summary("minimum_loss", subject, fold_accuracies_ml, fold_losses_ml)

        print_folds_summary("maximum_accuracy", subject, fold_accuracies_ma, fold_losses_ma)

        # =====================================
        # SAVA IMAGEM DE RESULTADOS
        # =====================================

        individual_subject_plots(branch=branch , subjects=subject)


# ==========================================================
# EXECUTAR
# ==========================================================

if __name__ == "__main__":

    # branch options:
    # space1_space2_frequency_time
    # space1_space2_time_frequency
    # time_frequency_space2_space1
    # time_frequency_space1_space2

    execute_branch(branch = "space1_space2_frequency_time", N_EPOCHS = 600, subjects = ["sub-01"])



