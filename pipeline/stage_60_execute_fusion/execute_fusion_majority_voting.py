
import numpy as np
from pathlib import Path
import sys
import torch
import pickle

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

if str(PROJECT_ROOT) not in sys.path:

    sys.path.append(
        str(PROJECT_ROOT)
    )


if str(PIPELINE_ROOT) not in sys.path:

    sys.path.append(
        str(PIPELINE_ROOT)
    )

from pipeline.stage_60_execute_fusion.tools import save_summary


from pipeline.stage_60_execute_fusion.fusion_individual_subject_plots import fusion_individual_subject_plots

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

N_FOLDS = 6
BATCH_SIZE = 8

RANDOM_STATE = 42

INPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_30_psd_array_assembly"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_60_execute_fusion"
)

OBTAIN_INDICES_STAGE = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_40_obtain_indices"
)

INPUT_PREVIOUS_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_50_execute_branch"
)


DEFAULT_SUBJECTS = [
    f"sub-{i:02d}"
    for i in range(1, 11)
]

def evaluate_majority_voting(
    model,
    data_loader,
    device
):

    model.eval()

    total_correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for (
            X_s1_s2_f_t_batch,
            X_s1_s2_t_f_batch,
            X_t_f_s2_s1_batch,
            X_t_f_s1_s2_batch,
            y_batch
        ) in data_loader:

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

            predictions = model(
                X_s1_s2_f_t_batch,
                X_s1_s2_t_f_batch,
                X_t_f_s2_s1_batch,
                X_t_f_s1_s2_batch
            )

            total_correct += (
                predictions == y_batch
            ).sum().item()

            total += y_batch.size(0)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                y_batch.cpu().tolist()
            )

    accuracy = (
        total_correct / total
    )

    return (
        accuracy,
        all_labels,
        all_predictions
    )


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


def execute_fusion_majority_voting(
    fusion,
    branch_model_state = "minimum_loss_model",
    subjects=subjects,
    condition_folder = "pronounced_speech",
    N_FOLDS=N_FOLDS,
    BATCH_SIZE=BATCH_SIZE,
    RANDOM_STATE=RANDOM_STATE,
):



    # ======================================================
    # CHECAGEM DA VARIÁVEL SUBJECTS
    # ======================================================

    if isinstance(
        subjects,
        str
    ):

        # Se o usuário passar uma única string,
        # transforma em uma lista com um elemento.

        subjects = [
            subjects
        ]


    elif subjects is None:

        subjects = (
            DEFAULT_SUBJECTS
        )

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


        # ==================================================
        # LOAD DATA
        # ==================================================

        file_path = (
            INPUT_DIR
            / f"{subject}_all_sessions.pkl"
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"\nFile not found:\n"
                f"{file_path}"
            )


        with open(file_path, "rb") as file:

            subj_file = pickle.load(file)


        psd_array = subj_file["psd_array"]
        labels = subj_file["labels"]

        # ==================================================
        # LOADING CNN MODELS
        # ==================================================

        condition = subj_file["condition"]

        if condition_folder.lower() != condition.lower() : 

            raise ValueError(
                f"Condition Folder {condition_folder.lower()} differs from condition {condition.lower()} obtained in checkpoint!"
            )             

        if condition == "PRONOUNCED_SPEECH":

            import pipeline.stage_50_execute_branch.branch_cnn_models_pronounced as cnn_models
            import pipeline.stage_60_execute_fusion.fusion_cnn_models_pronounced as fusion_cnn_models

        elif condition == "INNER_SPEECH":

            import pipeline.stage_50_execute_branch.branch_cnn_models_inner as cnn_models
            import pipeline.stage_60_execute_fusion.fusion_cnn_models_inner as fusion_cnn_models

        elif condition == "VISUALIZED_CONDITION":

            import pipeline.stage_50_execute_branch.branch_cnn_models_visualized as cnn_models
            import pipeline.stage_60_execute_fusion.fusion_cnn_models_visualized as fusion_cnn_models
        else:

            raise ValueError(
                f"Condition {condition} not found!"
            )

        print(
            f"Chosen {condition} models"
        )


        print(
            "Power:",
            psd_array.shape
        )

        print(
            "Labels:",
            labels.shape
        )

        # ======================================================
        # DIRETÓRIOS
        # ======================================================

        S1_S2_F_T_SUB_DIR = (
            INPUT_PREVIOUS_DIR
            / subject
            / condition_folder.lower()
            / f"{branch_s1_s2_f_t}_branch"
        )

        S1_S2_T_F_SUB_DIR = (
            INPUT_PREVIOUS_DIR
            / subject
            / condition_folder.lower()
            / f"{branch_s1_s2_t_f}_branch"
        )


        T_F_S2_S1_SUB_DIR = (
            INPUT_PREVIOUS_DIR
            / subject
            / condition_folder.lower()
            / f"{branch_t_f_s2_s1}_branch"
        )


        T_F_S1_S2_SUB_DIR = (
            INPUT_PREVIOUS_DIR
            / subject
            / condition_folder.lower()
            / f"{branch_t_f_s1_s2}_branch"
        )


        SUB_CONDIT_FUSION_DIR = (
            OUTPUT_DIR
            / subject
            / condition_folder.lower()
            / f"{fusion}_from_{branch_model_state}_in_branches"
        )

        SUB_CONDIT_FUSION_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        FOLDS_DIR = (
            OBTAIN_INDICES_STAGE
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

        power_array_s1_s2_f_t = np.transpose(
            psd_array,
            (0, 4, 3, 1, 2)
        )

        power_array_s1_s2_t_f = np.transpose(
            psd_array,
            (0, 3, 4, 1, 2)
        )

        power_array_t_f_s2_s1 = np.transpose(
            psd_array,
            (0, 2, 1, 3, 4)
        )

        power_array_t_f_s1_s2 = np.transpose(
            psd_array,
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

        test_fold_accuracies = []


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

    
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
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
                cnn_models                
                .Space1Space2FrequencyTimeCNN()
                .to(device)
            )

            model_s1_s2_f_t.load_state_dict(
                checkpoint_s1_s2_f_t[branch_model_state][
                    "model_state_dict"
                ]
            )

            model_s1_s2_t_f = (
                cnn_models
                .Space1Space2TimeFrequencyCNN()
                .to(device)
            )

            model_s1_s2_t_f.load_state_dict(
                checkpoint_s1_s2_t_f[branch_model_state][
                    "model_state_dict"
                ]
            )

            model_t_f_s2_s1 = (
                cnn_models
                .TimeFrequencySpace2Space1CNN()
                .to(device)
            )

            model_t_f_s2_s1.load_state_dict(
                checkpoint_t_f_s2_s1[branch_model_state][
                    "model_state_dict"
                ]
            )

            model_t_f_s1_s2 = (
                cnn_models
                .TimeFrequencySpace1Space2CNN()
                .to(device)
            )

            model_t_f_s1_s2.load_state_dict(
                checkpoint_t_f_s1_s2[branch_model_state][
                    "model_state_dict"
                ]
            )

            # ==================================================
            # FREEZE BRANCH MODELS
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

                case "majority_voting_fusion":
                
                    model = fusion_cnn_models.MajorityVotingFusion(
                        model_s1_s2_f_t,
                        model_s1_s2_t_f,
                        model_t_f_s2_s1,
                        model_t_f_s1_s2
                    ).to(device)
                    
                case _:

                    raise ValueError(
                        f"Invalid fusion {fusion}"
                    )

         
            # ===================================================
            # IMPRIMINDO O NÚMERO DE PARÂMETROS DA REDE
            # ===================================================

            if fold == 1:
            
                save_summary(
                    model,
                    fusion,
                    "Majority Voting",
                    summary_input_sizes,
                    device,
                    SUB_CONDIT_FUSION_DIR
                )

            # ================================================
            # EVALUATING MAJORITY VOTING
            # ================================================

            (
                train_accuracy,
                all_labels_train,
                all_predictions_train
            ) = evaluate_majority_voting(
                model,
                train_loader,
                device
            )

            (
                val_accuracy, 
                all_labels_val, 
                all_predictions_val
            ) = evaluate_majority_voting(
                model,
                val_loader,
                device
            )

            (
                test_accuracy,
                all_labels_test,
                all_predictions_test
            ) = evaluate_majority_voting(
                model,
                test_loader,
                device
            )   


            test_fold_accuracies.append(
                test_accuracy
            )    

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

                    "random_state":
                        RANDOM_STATE,

                   
                },

                "training_set": {

                    "accuracy": train_accuracy,

                    "labels": all_labels_train,
                     
                    "predictions": all_predictions_train

                },

                "validation_set": {
                
                    "accuracy": val_accuracy,

                    "labels": all_labels_val,
                        
                    "predictions": all_predictions_val

                },

                "test_set": {
                            
                    "accuracy": test_accuracy,

                    "labels": all_labels_test,
                        
                    "predictions": all_predictions_test

                }

            }

            torch.save(
                checkpoint,
                SUB_CONDIT_FUSION_DIR
                / f"{fusion}_fold_{fold}.pth"
            )



            # ======================================================
            # PRINT DOS RESULTADOS
            # ======================================================


            print()
            print("-" * 70)
            print(f"FOLD {fold} RESULTS")
            print("-" * 70)

            print("Training:")
            print("Accuracy:", train_accuracy)
            print("All labels:     ", all_labels_train)
            print("All predictions:", all_predictions_train)

            print()

            print("Validation:")
            print("Accuracy:", val_accuracy)
            print("All labels:     ", all_labels_val)
            print("All predictions:", all_predictions_val)

            print()

            print("Test:")
            print("Accuracy:", test_accuracy)
            print("All labels:     ", all_labels_test)
            print("All predictions:", all_predictions_test)


        # ===========================================
        # PRINTING FOLDS RESULTS
        # ===========================================

        test_fold_accuracies = np.array(
            test_fold_accuracies
        )

        print("\nMAJORITY VOTING - TEST RESULTS")
        print("=" * 70)

        for fold, accuracy in enumerate(
            test_fold_accuracies,
            start=1
        ):
            print(
                f"Fold {fold}: {accuracy:.4f}"
            )

        print()
        print(
            "Mean accuracy:",
            test_fold_accuracies.mean()
        )

        print(
            "Standard deviation:",
            test_fold_accuracies.std()
        )


        # ========================================
        # SAVING FOLDS SUMMARY 
        # ========================================
        summary_results = {

            "fusion":
                fusion,

            "subject":
                subject,

            "branch_model_state":
                branch_model_state,

            "n_folds":
                N_FOLDS,

            "test_fold_accuracies":
                test_fold_accuracies,

            "mean_test_accuracy":
                test_fold_accuracies.mean(),

            "std_test_accuracy":
                test_fold_accuracies.std(),

            "frequencies":
                subj_file["frequencies"],

            "times":
                subj_file["times"],

            "channel_names":
                subj_file["channel_names"],

            "sampling_rate":
                subj_file["sampling_rate"],

            "condition":
                subj_file["condition"],

            "start_seconds":
                subj_file["start_seconds"],

            "end_seconds":
                subj_file["end_seconds"],
        }

        torch.save(
            summary_results,

            SUB_CONDIT_FUSION_DIR
            / f"{fusion}_summary.pth"
        )


        print(
            f"\nSummary saved: "
            f"{SUB_CONDIT_FUSION_DIR / f'{fusion}_summary.pth'}"
        )

        # ======================================================
        # GRÁFICOS
        # ======================================================

        # fusion_individual_subject_plots(
        #     fusion=fusion, 
        #     branch_model_state = branch_model_state,           
        #     subjects=[subject],
        #     condition_folder = condition_folder,
        #     OUTPUT_DIR = OUTPUT_DIR
            
        # )


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

    # ======================================================
    # FUSION OPTIONS
    # ======================================================
    #
    # majority_voting_fusion
    # 
    #
    # ======================================================


    execute_fusion_majority_voting(
        fusion= "majority_voting_fusion",
        branch_model_state = "maximum_accuracy_model",
        subjects=["sub-01"],
        condition_folder = "pronounced_speech",        
    )
