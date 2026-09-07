import copy
import numpy as np
from pathlib import Path
import sys                        
import matplotlib.pyplot as plt

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
# CONFIGURAÇÕES DE CAMINHO & IMPORTS LOCAIS
# ==========================================================

current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[2]
PIPELINE_ROOT = current_file.parents[1]  # Caminho até a pasta 'pipeline'

# 2. Adiciona a pasta 'pipeline' ao sys.path para o Python achar a 'stage41'
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.append(str(PIPELINE_ROOT))

# 3. Agora o import funciona tanto no terminal quanto no debugger do VS Code
from fusion_individual_subject_plots import fusion_individual_subject_plots


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

N_FOLDS = 6
N_EPOCHS_ESTAGE_1 = 600
N_EPOCHS_STAGE_2 = 100
BATCH_SIZE = 8
#LEARNING_RATE = 0.001

LEARNING_RATE_STAGE_1 = 0.001
LEARNING_RATE_STAGE_2 = 0.0001



RANDOM_STATE = 42
RANDOM_STATE_LOADER = 42


INPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_40_cnn"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_50_fusion"
)




# ==========================================================
# DEFINIR A REDE
# ==========================================================
class Space1Space2FrequencyTimeCNN(nn.Module):

   def __init__(self):


        super().__init__()

       # (batch, 9, 65, 21, 21)

        self.conv1 = nn.Conv3d(
           in_channels=9,
           out_channels=6,
           kernel_size=3,
           padding=1
        )
        # (batch, 6, 65, 21, 21)


        self.relu1 = nn.ReLU()


        self.pool1 = nn.MaxPool3d(
           kernel_size=(2, 2, 2)
        )
        # (batch, 6, 32, 10, 10)


        self.conv2 = nn.Conv3d(
           in_channels=6,
           out_channels=4,
           kernel_size=3,
           padding=1
         )
        # (batch, 4, 32, 10, 10)

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool3d(
           kernel_size=(2, 2, 2)
        )
        # (batch, 4, 16, 5, 5)

        self.flatten = nn.Flatten()

        self.classifier = nn.Linear(
           4 * 16 * 5 * 5,
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



class Space1Space2TimeFrequencyCNN(nn.Module):

   def __init__(self):


        super().__init__()

       # (batch, 65, 9, 21, 21)

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



class TimeFrequencySpace2Space1CNN(nn.Module):

   def __init__(self):


        super().__init__()

       # (batch, 21, 21, 65, 9)

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



class TimeFrequencySpace1Space2CNN(nn.Module):

   def __init__(self):


        super().__init__()

       # (batch, 21, 21, 65, 9)

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


class FusionModel(nn.Module):

    def __init__(
        self,
        space_frequency_model,
        space_time_model
    ):

        super().__init__()

        self.space_frequency_model = space_frequency_model
        self.space_time_model = space_time_model


        # ==================================================
        # PROJEÇÕES
        # ==================================================

        # 3200 -> 128
        self.sf_projection = nn.Sequential(
            nn.Linear(3200, 128),
            nn.ReLU()
        )

        # 1800 -> 128
        self.st_projection = nn.Sequential(
            nn.Linear(1800, 128),
            nn.ReLU()
        )


        # ==================================================
        # GATE
        # ==================================================

        # Recebe:
        #
        # SF = 128
        # ST = 128
        #
        # concatenação = 256
        #
        # Produz 128 valores de gate

        self.gate = nn.Linear(
            256,
            128
        )


        # ==================================================
        # CLASSIFICADOR
        # ==================================================

        # A representação fundida possui
        # 128 características

        self.classifier = nn.Linear(
            128,
            4
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.Linear(64, 4),
        )


    def forward(
        self,
        x_space_frequency,
        x_space_time
    ):

        # ==================================================
        # EXTRAIR CARACTERÍSTICAS DOS DOIS RAMOS
        # ==================================================

        sf = (
            self.space_frequency_model
            .extract_features(
                x_space_frequency
            )
        )

        # (batch, 3200)


        st = (
            self.space_time_model
            .extract_features(
                x_space_time
            )
        )

        # (batch, 1800)


        # ==================================================
        # PROJETAR PARA 128 CARACTERÍSTICAS
        # ==================================================

        sf = self.sf_projection(sf)

        # (batch, 128)


        st = self.st_projection(st)

        # (batch, 128)


        # ==================================================
        # CONCATENAR
        # ==================================================

        combined = torch.cat(
            (sf, st),
            dim=1
        )

        # (batch, 256)


        # ==================================================
        # CALCULAR O GATE
        # ==================================================

        gate_logits = self.gate(
            combined
        )

        # (batch, 128)


        gate = torch.sigmoid(
            gate_logits
        )

        # (batch, 128)
        #
        # Cada valor está entre 0 e 1.


        # ==================================================
        # GATED FUSION
        # ==================================================

        fused = (
            gate * sf
            + (1 - gate) * st
        )

        # (batch, 128)


        # ==================================================
        # CLASSIFICAÇÃO
        # ==================================================

        outputs = self.classifier(
            fused
        )

        # (batch, 4)

        return outputs


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

    print(f"Best Epoch: {best_epoch}")


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

def execute_fusion(
        fusion_model,
        subjects = subjects,
        N_FOLDS = N_FOLDS,
        N_EPOCHS_STAGE_1 = N_EPOCHS_ESTAGE_1,
        N_EPOCHS_STAGE_2 = N_EPOCHS_STAGE_2,
        BATCH_SIZE = BATCH_SIZE,
        LEARNING_RATE_STAGE_1 = LEARNING_RATE_STAGE_1,
        LEARNING_RATE_STAGE_2 = LEARNING_RATE_STAGE_2
       
        RANDOM_STATE = RANDOM_STATE,
        RANDOM_STATE_LOADER = RANDOM_STATE_LOADER,
):

    torch.manual_seed(
        RANDOM_STATE
    )

    branch_s1_s2_f_t = "space1_space2_frequency_time"
    branch_s1_s2_t_f = "space1_space2_time_frequency"

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
        # DIRETÓRIOS DOS MODELOS
        # ======================================================

        # S1_S2_F_T:

        S1_S2_F_T_MODEL_DIR = (
            OUTPUT_DIR
            / f"{branch_s1_s2_f_t}_branch"
        )

        S1_S2_F_T_SUB_DIR = (
            S1_S2_F_T_MODEL_DIR
            / subject
        )

        # S1_S2_T_F:
        
        S1_S2_T_F_MODEL_DIR = (
            OUTPUT_DIR
            / f"{branch_s1_s2_t_f}_branch"
        )

        S1_S2_T_F_SUB_DIR = (
            S1_S2_T_F_MODEL_DIR
            / subject
        )

        # Fusion
        FUSION_MODEL_DIR = (
            OUTPUT_DIR
            / f"{fusion_model}_fusion"
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


        # print(
        #     "Min:",
        #     X.min()
        # )

        # print(
        #     "Max:",
        #     X.max()
        # )

        # print(
        #     "Mean:",
        #     X.mean()
        # )

        # print(
        #     "Std:",
        #     X.std()
        # )


        # print(
        #     "Valores diferentes de zero:",
        #     torch.count_nonzero(X).item()
        # )


        # print(
        #     "Proporção diferente de zero:",
        #     torch.count_nonzero(X).item()
        #     / X.numel()
        # )


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

        # ==========================================================
        # SALVANDO ESTRUTURA DA REDE
        # ==========================================================
        print("=== Verificando Estrutura da Rede ===")


        match fusion_model:

            case "fusion_mul":

                None

                # modelo_validador = Space1Space2FrequencyTimeCNN().to(device)
                # model_stats = summary(modelo_validador, input_size=(1, 9, 65, 21, 21), device=device, verbose=0)
                               
            # case "space1_space2_time_frequency":

            #     modelo_validador = Space1Space2TimeFrequencyCNN().to(device)
            #     model_stats = summary(modelo_validador, input_size=(1, 65, 9, 21, 21), device=device, verbose=0)

            # case "time_frequency_space2_space1":

            #     modelo_validador = TimeFrequencySpace2Space1CNN().to(device)
            #     model_stats = summary(modelo_validador, input_size=(1, 21, 21, 65, 9), device=device, verbose=0)

            # case "time_frequency_space1_space2":

            #     modelo_validador = TimeFrequencySpace1Space2CNN().to(device)
            #     model_stats = summary(modelo_validador, input_size=(1, 21, 21, 65, 9), device=device, verbose=0)
            
            case _:
                raise ValueError(f"Invalid branch {fusion_model}")

        save_summary_as_pdf(fusion_model , model_stats, save_path= MODEL_DIR)

        # Chama a função para gerar a imagem
        #print(model_stats)

        # Deleta a instância temporária para liberar memória da GPU imediatamente
        del modelo_validador 
        torch.cuda.empty_cache()

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
            val_indices = fold_data["val_indices"]
            test_indices = fold_data["test_indices"]


            # ==================================================
            # DIVIDIR A PORCENTAGEM DE DESENVOLVIMENTO EM 
            # TREINAMENTO E VALIDAÇÃO
            # ==================================================  

            # train_indices, val_indices = train_test_split(
            #     development_indices,
            #     test_size=VALIDATION_SIZE,
            #     stratify=labels[
            #         development_indices
            #     ],
            #     random_state=RANDOM_STATE
            # )

            # ======================================================
            # 8.1 SEPARAR TREINO E TESTE
            # ======================================================

            # X_train = X[train_indices]
            # y_train = y[train_indices]


            # X_test = X[test_indices]
            # y_test = y[test_indices]

            # X_val = X[val_indices]


            # --------------------------------
            # Normalização
            # --------------------------------
            # power_max = X_train.max()
            # X_train = X_train / power_max
            # X_test = X_test / power_max

            # print("Treino:", X_train.shape)
            # print("Teste:", X_test.shape)

            # ==================================================
            # MOSTRAR TAMANHOS
            # ==================================================

            # print(
            #     "Total:",
            #     len(X)
            # )

            # print(
            #     "Treinamento:",
            #     len(train_indices)
            # )

            # print(
            #     "Validação:",
            #     len(val_indices)
            # )

            # print(
            #     "Teste:",
            #     len(test_indices)
            # )


            # print(
            #     "Treinamento (%):",
            #     len(train_indices)
            #     / len(X)
            # )

            # print(
            #     "Validação (%):",
            #     len(val_indices)
            #     / len(X)
            # )

            # print(
            #     "Teste (%):",
            #     len(test_indices)
            #     / len(X)
            # )


            # ==================================================
            # SALVAR OS ÍNDICES
            # ==================================================

            # Esses são índices relativos ao conjunto ORIGINAL.
            #
            # Portanto, o ramo espaço-tempo poderá carregá-los
            # diretamente.

            # np.savez(
            #     FOLDS_DIR
            #     / f"fold_{fold}.npz",

            #     train_indices=train_indices,
            #     val_indices=val_indices,
            #     test_indices=test_indices
            # )


            # ==================================================
            # SEPARAR OS DADOS
            # ==================================================

            # X_train = X[
            #     train_indices
            # ]

            # y_train = y[
            #     train_indices
            # ]


            # X_val = X[
            #     val_indices
            # ]

            # y_val = y[
            #     val_indices
            # ]


            # X_test = X[
            #     test_indices
            # ]

            # y_test = y[
            #     test_indices
            # ]

            # ======================================================
            # 9.2 SEPARAR DADOS DE S1_S2_F_T
            # ======================================================

            X_s1_s2_f_t_train = (X_s1_s2_f_t[train_indices])
            X_s1_s2_f_t_val = (X_s1_s2_f_t[val_indices])
            X_s1_s2_f_t_test = (X_s1_s2_f_t[test_indices])

            # ======================================================
            # 9.3 SEPARAR DADOS DE S1_S2_T_F
            # ======================================================

            X_s1_s2_t_f_train = (X_s1_s2_t_f[train_indices])
            X_s1_s2_t_f_val = (X_s1_s2_t_f[val_indices])
            X_s1_s2_t_f_test = (X_s1_s2_t_f[test_indices])

            # ======================================================
            # 9.4 LABELS
            # ======================================================

            y_train = y[train_indices]
            y_val = y[val_indices]
            y_test = y[test_indices]


            # ======================================================
            # 9.5 CARREGAR CHECKPOINT S1_S2_F_T
            # ======================================================

            checkpoint_s1_s2_f_t = torch.load(
                S1_S2_F_T_SUB_DIR
                / f"space1_space2_frequency_time_fold_{fold}.pth",
                map_location=device,
                weights_only=False
            )

            power_max_s1_s2_f_t = (
                checkpoint_s1_s2_f_t["power_max"].item()
            )

            # ======================================================
            # 9.5 CARREGAR CHECKPOINT S1_S2_T_F
            # ======================================================

            checkpoint_s1_s2_t_f = torch.load(
                S1_S2_T_F_SUB_DIR
                / f"space1_space2_time_frequency_fold_{fold}.pth",
                map_location=device,
                weights_only=False
            )

            power_max_s1_s2_t_f = (
                checkpoint_s1_s2_t_f["power_max"].item()
            )



            # ==================================================
            # NORMALIZAÇÃO
            # ==================================================

            X_s1_s2_f_t_train = X_s1_s2_f_t_train / power_max_s1_s2_f_t
            X_s1_s2_f_t_val = X_s1_s2_f_t_val / power_max_s1_s2_f_t
            X_s1_s2_f_t_test = X_s1_s2_f_t_test / power_max_s1_s2_f_t

            X_s1_s2_t_f_train = X_s1_s2_t_f_train / power_max_s1_s2_t_f
            X_s1_s2_t_f_val = X_s1_s2_t_f_val / power_max_s1_s2_t_f
            X_s1_s2_t_f_test = X_s1_s2_t_f_test / power_max_s1_s2_t_f


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


            # X_s1_s2_f_t
            print(
                "X_s1_s2_f_t_train:",
                X_s1_s2_f_t_train.shape
            )

            print(
                "X_s1_s2_f_t_val:",
                X_s1_s2_f_t_val.shape
            )

            print(
                "X_s1_s2_f_t:",
                X_s1_s2_f_t_test.shape
            )


            #X_s1_s2_t_f
            print(
                "X_s1_s2_t_f_train:",
                X_s1_s2_t_f_train.shape
            )

            print(
                "X_s1_s2_t_f__val:",
                X_s1_s2_t_f_val.shape
            )

            print(
                "X_s1_s2_t_f:",
                X_s1_s2_t_f_test.shape
            )


            # ==================================================
            # DATASETS
            # ==================================================

            train_dataset = TensorDataset(
                X_s1_s2_f_t_train,
                X_s1_s2_t_f_train,
                y_train
            )


            val_dataset = TensorDataset(
                X_s1_s2_f_t_val,
                X_s1_s2_t_f_val,
                y_val
            )


            test_dataset = TensorDataset(
                X_s1_s2_f_t_test,
                X_s1_s2_t_f_test,
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
            # CRIAR UMA NOVA REDE PARA ESTE FOLD
            # ==================================================

            torch.manual_seed(
                RANDOM_STATE
            )

            # ======================================================
            # 9.8 CARREGAR MODELO S1_S2_F_T
            # ======================================================

            model_s1_s2_f_t = (
                Space1Space2FrequencyTimeCNN()
                .to(device)
            )

            model_s1_s2_f_t.load_state_dict(
                checkpoint_s1_s2_f_t[
                    "model_state_dict"
                ]
            )

            # ======================================================
            # 9.9 CARREGAR MODELO S1_S2_T_F
            # ======================================================

            model_s1_s2_t_f = (
                Space1Space2TimeFrequencyCNN()
                .to(device)
            )

            model_s1_s2_t_f.load_state_dict(
                checkpoint_s1_s2_t_f[
                    "model_state_dict"
                ]
            )

            # ======================================================
            # 9.10 CONGELAR OS RAMOS
            # ======================================================

            for parameter in model_s1_s2_f_t.parameters():

                parameter.requires_grad = False


            for parameter in model_s1_s2_t_f.parameters():

                parameter.requires_grad = False


            # ===============================================
            # SELECIONA O MODELO DE FUSÃO
            # ===============================================
            match fusion_model:

                case "fusion_mul":

                    model = FusionModel(
                        model_s1_s2_f_t, 
                        model_s1_s2_t_f
                    ).to(device)


                    
                                
                # case "space1_space2_time_frequency":

                #     model = Space1Space2TimeFrequencyCNN().to(device)

                # case "time_frequency_space2_space1":
                                        
                #     model = TimeFrequencySpace2Space1CNN().to(device)

                # case "time_frequency_space1_space2":

                #     model = TimeFrequencySpace1Space2CNN().to(device)

                case _:
                    raise ValueError(f"Invalid branch {fusion_model}")




            
            # ==================================================
            # FUNÇÃO DE PERDA
            # ==================================================

            criterion = nn.CrossEntropyLoss()

            # ======================================================
            # ESTÁGIO 1
            # TREINAR SOMENTE O CLASSIFICADOR DE FUSÃO
            # ======================================================

            # ==================================================
            # OTIMIZADOR
            # ==================================================

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LEARNING_RATE_STAGE_1
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
            min_val_loss = float("inf")
            min_val_loss_epoch = 0
            min_val_loss_model_state = None

            # máxima acurácia
            max_val_accuracy = -1
            best_val_loss_to_max_accuracy = float("inf")
            max_val_accuracy_epoch = 0
            max_val_accuracy_state = None

            # ==================================================
            # TREINAMENTO - ESTAGIO 1
            # ==================================================

            for epoch in range(N_EPOCHS_STAGE_1):


                # ==============================================
                # TREINAMENTO DA ÉPOCA
                # ==============================================

                model.train()

                # Os dois extratores permanecem congelados
                # e em modo de avaliação
                model_s1_s2_f_t.eval()
                model_s1_s2_t_f.eval()

                train_total_loss = 0.0
                train_correct = 0
                train_total = 0


                for (
                        X_s1_s2_f_t_batch,
                        X_s1_s2_t_f_batch,
                        y_batch
                    ) in train_loader:


                    X_s1_s2_f_t_batch = X_s1_s2_f_t_batch.to(
                        device
                    )

                    X_s1_s2_t_f_batch = X_s1_s2_t_f_batch.to(
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
                        X_s1_s2_f_t_batch, 
                        X_s1_s2_t_f_batch
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
                    f"Epoch {epoch + 1:03d}/{N_EPOCHS_STAGE_1} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {train_accuracy:.4f} | "
                    f"Val Acc: {val_accuracy:.4f}"
                )


            # # ======================================================
            # # ESTÁGIO 2
            # # DESCONGELAR OS DOIS RAMOS
            # # ======================================================

            # for parameter in model_s1_s2_f_t.parameters():
            #     parameter.requires_grad = True

            # for parameter in model_s1_s2_t_f.parameters():
            #     parameter.requires_grad = True


            # # ======================================================
            # # NOVO OTIMIZADOR PARA FINE-TUNING
            # # ======================================================

            # optimizer = torch.optim.Adam(
            #     fusion_model.parameters(),
            #     lr=LEARNING_RATE_STAGE_2
            # )

            # # ==================================================
            # # TREINAMENTO ESTAGIO 2 - FINE TUNNING
            # # ==================================================

            # for epoch in range(N_EPOCHS_STAGE_2):

            #     model.train()

            #     train_total_loss = 0.0
            #     train_correct = 0
            #     train_total = 0


            #     for (
            #             X_s1_s2_f_t_batch,
            #             X_s1_s2_t_f_batch,
            #             y_batch
            #         ) in train_loader:


            #         X_s1_s2_f_t_batch = X_s1_s2_f_t_batch.to(
            #             device
            #         )

            #         X_s1_s2_t_f_batch = X_s1_s2_t_f_batch.to(
            #             device
            #         )

            #         y_batch = y_batch.to(
            #             device
            #         )


            #         # ------------------------------------------
            #         # ZERAR GRADIENTES
            #         # ------------------------------------------

            #         optimizer.zero_grad()


            #         # ------------------------------------------
            #         # FORWARD
            #         # ------------------------------------------

            #         outputs = model(
            #             X_s1_s2_f_t_batch, 
            #             X_s1_s2_t_f_batch
            #         )


            #         # ------------------------------------------
            #         # LOSS
            #         # ------------------------------------------

            #         loss = criterion(
            #             outputs,
            #             y_batch
            #         )


            #         # ------------------------------------------
            #         # BACKPROPAGATION
            #         # ------------------------------------------

            #         loss.backward()


            #         # ------------------------------------------
            #         # ATUALIZAR PESOS
            #         # ------------------------------------------

            #         optimizer.step()


            #         # ------------------------------------------
            #         # ACUMULAR LOSS
            #         # ------------------------------------------

            #         train_total_loss += (
            #             loss.item()
            #         )


            #         # ------------------------------------------
            #         # PREVISÕES
            #         # ------------------------------------------

            #         predictions = outputs.argmax(
            #             dim=1
            #         )


            #         # ------------------------------------------
            #         # ACERTOS
            #         # ------------------------------------------

            #         train_correct += (
            #             predictions
            #             == y_batch
            #         ).sum().item()


            #         train_total += (
            #             y_batch.size(0)
            #         )


            #     # ==============================================
            #     # RESULTADOS DE TREINAMENTO
            #     # ==============================================

            #     train_loss = (
            #         train_total_loss
            #         / len(train_loader)
            #     )


            #     train_accuracy = (
            #         train_correct
            #         / train_total
            #     )


            #     # ==============================================
            #     # VALIDAÇÃO DA ÉPOCA
            #     # ==============================================

            #     model.eval()


            #     val_total_loss = 0.0

            #     val_correct = 0

            #     val_total = 0


            #     with torch.no_grad():


            #         for (
            #             X_batch,
            #             y_batch
            #         ) in val_loader:


            #             X_batch = X_batch.to(
            #                 device
            #             )

            #             y_batch = y_batch.to(
            #                 device
            #             )


            #             outputs = model(
            #                 X_batch
            #             )


            #             loss = criterion(
            #                 outputs,
            #                 y_batch
            #             )


            #             val_total_loss += (
            #                 loss.item()
            #             )


            #             predictions = outputs.argmax(
            #                 dim=1
            #             )


            #             val_correct += (
            #                 predictions
            #                 == y_batch
            #             ).sum().item()


            #             val_total += (
            #                 y_batch.size(0)
            #             )


            #     # ==============================================
            #     # RESULTADOS DE VALIDAÇÃO
            #     # ==============================================

            #     val_loss = (
            #         val_total_loss
            #         / len(val_loader)
            #     )


            #     val_accuracy = (
            #         val_correct
            #         / val_total
            #     )


            #     # ==============================================
            #     # GUARDAR HISTÓRICO
            #     # ==============================================

            #     train_accuracies_epoch.append(
            #         train_accuracy
            #     )


            #     val_accuracies_epoch.append(
            #         val_accuracy
            #     )


            #     train_losses_epoch.append(
            #         train_loss
            #     )


            #     val_losses_epoch.append(
            #         val_loss
            #     )


            #     # ==============================================
            #     # VERIFICAR SE É O MELHOR MODELO
            #     # ==============================================

            #     # menor loss
            #     if val_loss < min_val_loss:

            #         min_val_loss = (
            #             val_loss
            #         )


            #         min_val_loss_epoch = (
            #             epoch
            #         )

            #         min_val_loss_model_state = copy.deepcopy(
            #             model.state_dict()
            #         )

            #     #maior acurácia
            #     if (val_accuracy > max_val_accuracy or
            #          ( val_accuracy == max_val_accuracy and val_loss < best_val_loss_to_max_accuracy)
            #         ):
                    
            #         max_val_accuracy = val_accuracy
            #         best_val_loss_to_max_accuracy = val_loss
            #         max_val_accuracy_epoch = epoch

            #         max_val_accuracy_model_state = copy.deepcopy(
            #             model.state_dict()
            #         )


            #     # ==============================================
            #     # MOSTRAR RESULTADOS
            #     # ==============================================

            #     print(
            #         f"Fold {fold} | "
            #         f"Epoch {epoch + 1:03d}/{N_EPOCHS_STAGE_1} | "
            #         f"Train Loss: {train_loss:.4f} | "
            #         f"Val Loss: {val_loss:.4f} | "
            #         f"Train Acc: {train_accuracy:.4f} | "
            #         f"Val Acc: {val_accuracy:.4f}"
            #     )


            





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
                / f"{fusion_model}_fold_{fold}.pth"
            )

            
        # ==========================================================
        # RESULTADOS DOS 5 FOLDS
        # ==========================================================

        print_folds_summary("minimum_loss", subject, fold_accuracies_ml, fold_losses_ml)

        print_folds_summary("maximum_accuracy", subject, fold_accuracies_ma, fold_losses_ma)

        # =====================================
        # SAVA IMAGEM DE RESULTADOS
        # =====================================

        individual_subject_plots_fusion(branch=fusion_model , subjects=subject)


# ==========================================================
# EXECUTAR
# ==========================================================

if __name__ == "__main__":

    execute_fusion(fusion_model = "space1_space2_frequency_time", N_EPOCHS_STAGE_1 = 600, subjects = ["sub-01"])