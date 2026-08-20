import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from space_frequency import SpaceFrequencyCNN
from space_time import SpaceTimeCNN


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

N_FOLDS = 5

N_EPOCHS_FUSION = 50
N_EPOCHS_FINE_TUNING = 20

LEARNING_RATE_FUSION = 0.001
LEARNING_RATE_FINE_TUNING = 0.0001

BATCH_SIZE = 8

RANDOM_STATE = 42
RANDOM_STATE_LOADER = 42

# Inicialização dos pesos e outras operações aleatórias do PyTorch
torch.manual_seed(RANDOM_STATE)

# ==========================================================
# CAMINHOS
# ==========================================================

current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[2]

INPUT_DIR = (PROJECT_ROOT / "processed_data"/ "stage_30_array_assembly" )

OUTPUT_DIR = (PROJECT_ROOT / "processed_data" / "stage_40_cnn" )

FOLDS_DIR = OUTPUT_DIR / "folds"

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]


# ==========================================================
# 1. CARREGAR OS DADOS
# ==========================================================

subject = "sub-02"

power_array = np.load(
    INPUT_DIR / f"{subject}_power.npy"
)

labels = np.load(
    INPUT_DIR / f"{subject}_labels.npy"

)

print("Power:", power_array.shape)
print("Labels:", labels.shape)


SPACE_FREQUENCY_MODEL_DIR = (OUTPUT_DIR / "space_frequency_models" / subject )

SPACE_TIME_MODEL_DIR = ( OUTPUT_DIR / "space_time_models" / subject )

FUSION_MODEL_DIR = (OUTPUT_DIR / "fusion_models" / subject)

FUSION_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================================
# 2. CRIAR AS DUAS REPRESENTAÇÕES
# ==========================================================

# Original:
#
# (epoch, row, column, frequency, time)
#
# (80, 21, 21, 65, 9)


# ----------------------------------------------------------
# Espaço-Frequência
#
# canais = tempo
#
# (epoch, time, frequency, row, column)
# ----------------------------------------------------------

X_space_frequency = np.transpose(
    power_array,
    (0, 4, 3, 1, 2)
)


# ----------------------------------------------------------
# Espaço-Tempo
#
# canais = frequência
#
# (epoch, frequency, time, row, column)
# ----------------------------------------------------------

X_space_time = np.transpose(
    power_array,
    (0, 3, 4, 1, 2)
)

print("Power reorganizado:", X_space_frequency.shape)
print("Power reorganizado:", X_space_time.shape)



# ==========================================================
# 3. CONVERTER PARA TENSORES
# ==========================================================

X_space_frequency = torch.from_numpy(
    X_space_frequency
).float()

X_space_time = torch.from_numpy(
    X_space_time
).float()

y = torch.from_numpy(
    labels
).long()


print(
    "Space-frequency:",
    X_space_frequency.shape
)

print(
    "Space-time:",
    X_space_time.shape
)


# ==========================================================
# 4. DEFINIR A REDE DE FUSÃO 
# ==========================================================

# As redes space-frequency e space-time são importadas

class FusionModel(nn.Module):

    def __init__(
        self,
        space_frequency_model,
        space_time_model
    ):

        super().__init__()

        self.space_frequency_model = space_frequency_model
        self.space_time_model = space_time_model

        # 3200 -> 64
        self.sf_projection = nn.Sequential(
            nn.Linear(3200, 128),
            nn.ReLU()
        )

        # 1800 -> 64
        self.st_projection = nn.Sequential(
            nn.Linear(1800, 128),
            nn.ReLU()
        )

        # Recebe SF + ST:
        # 64 + 64 = 128
        #
        # Produz um gate para cada uma
        # das 64 características
        self.gate = nn.Linear(
            128+128,
            128
        )

        # A gated fusion continua tendo
        # apenas 64 características
        self.classifier = nn.Linear(
            64,
            4
        )


    def forward(
        self,
        x_space_frequency,
        x_space_time
    ):

        # ---------------------------------------------
        # Extrair características
        # ---------------------------------------------

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


        # ---------------------------------------------
        # Projetar
        # ---------------------------------------------

        sf = self.sf_projection(sf)
        # (batch, 64)

        st = self.st_projection(st)
        # (batch, 64)


        # ---------------------------------------------
        # Juntar para calcular o gate
        # ---------------------------------------------

        combined = torch.cat(
            (sf, st),
            dim=1
        )
        # (batch, 128)


        # ---------------------------------------------
        # Calcular os 64 gates
        # ---------------------------------------------

        gate = torch.sigmoid(
            self.gate(combined)
        )
        # (batch, 64)


        # ---------------------------------------------
        # Gated fusion
        # ---------------------------------------------

        fused = (
            gate * sf
            + (1 - gate) * st
        )
        # (batch, 64)


        # ---------------------------------------------
        # Classificação
        # ---------------------------------------------

        outputs = self.classifier(
            fused
        )
        # (batch, 4)

        return outputs

    
class FusionModel(nn.Module):

    def __init__(
        self,
        space_frequency_model,
        space_time_model
    ):

        super().__init__()

        self.space_frequency_model = space_frequency_model
        self.space_time_model = space_time_model

        # ----------------------------------------------
        # Projetar os dois ramos para a mesma dimensão
        # ----------------------------------------------

        self.sf_projection = nn.Sequential(
            nn.Linear(3200, 128),
            nn.ReLU()
        )

        self.st_projection = nn.Sequential(
            nn.Linear(1800, 128),
            nn.ReLU()
        )

        # ----------------------------------------------
        # Classificador
        #
        # sf = 128
        # st = 128
        # produto externo = 128 * 64 = 4096
        #
        # total = 64 + 64 + 4096 = 4224
        # ----------------------------------------------

        self.gated = nn.Sequential(


            
        )




        self.classifier = nn.Sequential(

            nn.Linear(
                128 + 128 + 128*128,
                1024
            ),

            nn.ReLU(),

            nn.Linear(
                1024,
                512
            ),
            nn.ReLU(),

            nn.Linear(
                512,
                4
            )


        )


    def forward(
        self,
        x_space_frequency,
        x_space_time
    ):

        # ==============================================
        # 1. EXTRAIR FEATURES DOS DOIS RAMOS
        # ==============================================

        sf = (
            self.space_frequency_model
            .extract_features(
                x_space_frequency
            )
        )
        # shape:
        # (batch, 3200)


        st = (
            self.space_time_model
            .extract_features(
                x_space_time
            )
        )
        # shape:
        # (batch, 1800)


        # ==============================================
        # 2. PROJETAR PARA 64 DIMENSÕES
        # ==============================================

        sf = self.sf_projection(sf)
        # (batch, 64)

        st = self.st_projection(st)
        # (batch, 64)


        # ==============================================
        # 3. PRODUTO EXTERNO
        # ==============================================

        interaction = torch.bmm(
            sf.unsqueeze(2),
            st.unsqueeze(1)
        )

        # sf.unsqueeze(2):
        # (batch, 64, 1)
        #
        # st.unsqueeze(1):
        # (batch, 1, 64)
        #
        # resultado:
        # (batch, 64, 64)


        # ==============================================
        # 4. ACHATAR A MATRIZ DE INTERAÇÃO
        # ==============================================

        interaction = interaction.flatten(
            start_dim=1
        )

        # (batch, 4096)


        # ==============================================
        # 5. CONCATENAR
        # ==============================================

        fused = torch.cat(
            (
                sf,
                st,
                interaction
            ),
            dim=1
        )

        # (batch, 4224)


        # ==============================================
        # 6. CLASSIFICAÇÃO
        # ==============================================

        outputs = self.classifier(
            fused
        )

        return outputs
    
# class FusionModel(nn.Module):

#     def __init__(
#         self,
#         space_frequency_model,
#         space_time_model
#     ):
#         super().__init__()

#         self.space_frequency_model = space_frequency_model
#         self.space_time_model = space_time_model

#         self.sf_projection = nn.Sequential(
#             nn.Linear(3200, 64),
#             nn.ReLU()
#         )

#         self.st_projection = nn.Sequential(
#             nn.Linear(1800, 64),
#             nn.ReLU()
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(4096 + 2*64, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 4)
#         )

#     def forward(
#         self,
#         x_space_frequency,
#         x_space_time
#     ):

#         sf = self.space_frequency_model.extract_features(
#             x_space_frequency
#         )

#         st = self.space_time_model.extract_features(
#             x_space_time
#         )

#         sf = self.sf_projection(sf)
#         st = self.st_projection(st)


#         # interaction_sum = sf + st

#         # interaction_mul = sf * st

#         # interaction_diff = torch.abs(
#         #     sf - st
#         # )

#         interaction = torch.bmm(
#             sf.unsqueeze(2),
#             st.unsqueeze(1)
#         )

#         interaction = interaction.flatten(
#             start_dim=1
#         )

#         fused = torch.cat(
#             (
#                 sf, 
#                 st,
#                 interaction,
                
                
#             ),
#             dim=1
#         )

#         outputs = self.classifier(
#             fused
#         )

#         return outputs

# class FusionModel(nn.Module):

#     def __init__(
#         self,
#         space_frequency_model,
#         space_time_model
#     ):

#         super().__init__()

#         self.space_frequency_model = (
#             space_frequency_model
#         )

#         self.space_time_model = (
#             space_time_model
#         )

#         self.sf_projection = nn.Sequential(
#             nn.Linear(3200, 64),
#             nn.ReLU()
#         )

#         self.st_projection = nn.Sequential(
#             nn.Linear(1800, 64),
#             nn.ReLU()
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(
#                 64 + 64 + 64,
#                 64
#             ),
#             nn.ReLU(),

#             nn.Linear(
#                 64,
#                 4
#             )
#         )


#     def forward(
#         self,
#         x_space_frequency,
#         x_space_time
#     ):

#         features_sf = (
#             self.space_frequency_model
#             .extract_features(
#                 x_space_frequency
#             )
#         )

#         features_st = (
#             self.space_time_model
#             .extract_features(
#                 x_space_time
#             )
#         )

#         features_sf = (
#             self.sf_projection(
#                 features_sf
#             )
#         )

#         features_st = (
#             self.st_projection(
#                 features_st
#             )
#         )

#         interaction = (
#             features_sf + features_st
#         )

#         features = torch.cat(
#             (
#                 features_sf,
#                 features_st,
#                 interaction
#             ),
#             dim=1
#         )

#         outputs = (
#             self.classifier(
#                 features
#             )
#         )

#         return outputs

# class FusionModel(nn.Module):

#     def __init__(
#         self,
#         space_frequency_model,
#         space_time_model
#     ):

#         super().__init__()

#         self.space_frequency_model = space_frequency_model
#         self.space_time_model = space_time_model

#          # SF:
#                 # 8 * 16 * 5 * 5 = 3200
#                 #
#                 # ST:
#                 # 8 * 9 * 5 * 5 = 1800
#                 #
#                 # total:
#                 # 5000
        
#         # ==================================================
#         # PROJEÇÕES DOS DOIS ESPAÇOS LATENTES
#         # ==================================================

#         self.sf_projection = nn.Sequential(
#             nn.Linear(3200, 512),
#             nn.ReLU()
#         )

#         self.st_projection = nn.Sequential(
#             nn.Linear(1800, 512),
#             nn.ReLU()
#         )


#         # ==================================================
#         # CLASSIFICADOR APÓS A FUSÃO
#         # ==================================================

#         self.classifier = nn.Sequential(

#             nn.Linear(
#                 512 + 512,
#                 256
#             ),

#             nn.ReLU(),

#             nn.Linear(
#                 256,
#                 64
#             ),

#             nn.ReLU(),

#             nn.Linear(
#                 64,
#                 4
#             )
#         )


#     def forward(
#         self,
#         x_space_frequency,
#         x_space_time
#     ):

#         # ==================================================
#         # EXTRAIR FEATURES DOS RAMOS
#         # ==================================================

#         features_sf = (self.space_frequency_model.extract_features(x_space_frequency))

#         features_st = (self.space_time_model.extract_features(x_space_time))


#         # ==================================================
#         # REDUZIR OS ESPAÇOS LATENTES
#         # ==================================================

#         features_sf = self.sf_projection(
#             features_sf
#         )

#         features_st = self.st_projection(
#             features_st
#         )


#         # ==================================================
#         # CONCATENAÇÃO
#         # ==================================================

#         features = torch.cat(
#             (
#                 features_sf,
#                 features_st
#             ),
#             dim=1
#         )


#         # ==================================================
#         # CLASSIFICAÇÃO
#         # ==================================================

#         outputs = self.classifier(
#             features
#         )

#         return outputs

# class FusionModel(nn.Module):

#     def __init__(
#         self,
#         space_frequency_model,
#         space_time_model
#     ):

#         super().__init__()

#         self.space_frequency_model = (space_frequency_model)
#         self.space_time_model = (space_time_model)


#         # SF:
#         # 8 * 16 * 5 * 5 = 3200
#         #
#         # ST:
#         # 8 * 9 * 5 * 5 = 1800
#         #
#         # total:
#         # 5000

#         self.classifier = nn.Sequential(

#             nn.Linear(
#                 3200 + 1800,
#                 256
#             ),

#             nn.ReLU(),

#             nn.Linear(
#                 256,
#                 64
#             ),

#             nn.ReLU(),

#             nn.Linear(
#                 64,
#                 4
#             )
#         )


#     def forward(self, x_space_frequency, x_space_time):

#         # ----------------------------------------------
#         # Extrair espaço latente SF
#         # ----------------------------------------------

#         features_sf = (self.space_frequency_model.extract_features(x_space_frequency))


#         # ----------------------------------------------
#         # Extrair espaço latente ST
#         # ----------------------------------------------

#         features_st = (self.space_time_model.extract_features(x_space_time))


#         # ----------------------------------------------
#         # Concatenar
#         # ----------------------------------------------

#         features = torch.cat(
#             (
#                 features_sf,
#                 features_st
#             ),
#             dim=1
#         )


#         # ----------------------------------------------
#         # Novo classificador
#         # ----------------------------------------------

#         outputs = self.classifier(features)

#         return outputs

    
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

FOLDS_DIR = OUTPUT_DIR / "folds"
# FOLDS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 7. ARMAZENAR RESULTADO DOS FOLDS
# ==========================================================

fold_accuracies = []


# ==========================================================
# 8. LOOP DOS 5 FOLDS
# ==========================================================
for fold in range(1, N_FOLDS + 1):

    print()
    print("=" * 60)
    print(f"FOLD {fold}/{N_FOLDS}")
    print("=" * 60)

    # ======================================================
    # CARREGAR OS ÍNDICES DESTE FOLD
    # ======================================================

    fold_data = np.load(
        FOLDS_DIR / f"fold_{fold}.npz"
    )

    train_indices = fold_data["train_indices"]
    test_indices = fold_data["test_indices"]

    # ======================================================
    # 9.2 SEPARAR SF
    # ======================================================

    X_sf_train = (X_space_frequency[train_indices])
    X_sf_test = (X_space_frequency[test_indices])


    # ======================================================
    # 9.3 SEPARAR ST
    # ======================================================

    X_st_train = (X_space_time[train_indices])
    X_st_test = (X_space_time[test_indices])


    # ======================================================
    # 9.4 LABELS
    # ======================================================

    y_train = y[train_indices]
    y_test = y[test_indices]


    # ======================================================
    # 9.5 CARREGAR CHECKPOINT SF
    # ======================================================

    checkpoint_sf = torch.load(
        SPACE_FREQUENCY_MODEL_DIR
        / f"space_frequency_fold_{fold}.pth",
        map_location=device,
        weights_only=False
    )

    power_max_sf = (
        checkpoint_sf["power_max"].item()
    )


    # ======================================================
    # 9.6 CARREGAR CHECKPOINT ST
    # ======================================================

    checkpoint_st = torch.load(
        SPACE_TIME_MODEL_DIR
        / f"space_time_fold_{fold}.pth",
        map_location=device,
        weights_only=False
    )

    power_max_st = (
        checkpoint_st["power_max"].item()
    )



    # ======================================================
    # 9.7 NORMALIZAR
    # ======================================================

    X_sf_train = (X_sf_train / power_max_sf)
    X_sf_test = (X_sf_test / power_max_sf)


    X_st_train = (X_st_train / power_max_st)
    X_st_test = ( X_st_test / power_max_st)

    # ======================================================
    # 9.8 DATASETS
    # ======================================================

    train_dataset = TensorDataset(
        X_sf_train,
        X_st_train,
        y_train
    )

    test_dataset = TensorDataset(
        X_sf_test,
        X_st_test,
        y_test
    )


    # ======================================================
    # 9.13 DATALOADERS
    # ======================================================

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


    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # ======================================================
    # 8.4 CRIAR UMA NOVA REDE PARA ESTE FOLD
    # ======================================================
   
    torch.manual_seed(42) #inicia os pesos das redes de cada fold da mesma forma

    # ======================================================
    # 9.8 CARREGAR MODELO SF
    # ======================================================

    model_sf = (
        SpaceFrequencyCNN()
        .to(device)
    )

    model_sf.load_state_dict(
        checkpoint_sf[
            "model_state_dict"
        ]
    )

    # ======================================================
    # 9.9 CARREGAR MODELO ST
    # ======================================================

    model_st = (
        SpaceTimeCNN()
        .to(device)
    )

    model_st.load_state_dict(
        checkpoint_st[
            "model_state_dict"
        ]
    )

    # ======================================================
    # 9.10 CONGELAR OS RAMOS
    # ======================================================

    for parameter in model_sf.parameters():

        parameter.requires_grad = False


    for parameter in model_st.parameters():

        parameter.requires_grad = False

    # ======================================================
    # 9.11 MODELO DE FUSÃO
    # ======================================================

    fusion_model = FusionModel(
        model_sf,
        model_st
    ).to(device)




    # ======================================================
    # 9.14 LOSS
    # ======================================================

    criterion = nn.CrossEntropyLoss()


    # ======================================================
    # ESTÁGIO 1
    # TREINAR SOMENTE O CLASSIFICADOR DE FUSÃO
    # ======================================================

    optimizer = torch.optim.Adam(
        fusion_model.classifier.parameters(),
        lr=LEARNING_RATE_FUSION
    )

    # ======================================================
    # 9.16 TREINAMENTO
    # ======================================================

    for epoch in range(N_EPOCHS_FUSION):

        fusion_model.train()

        # Os dois extratores permanecem congelados
        # e em modo de avaliação
        model_sf.eval()
        model_st.eval()

        total_loss = 0.0
        correct = 0
        total = 0


        for (X_sf_batch, X_st_batch, y_batch) in train_loader:


            X_sf_batch = (
                X_sf_batch.to(device)
            )

            X_st_batch = (
                X_st_batch.to(device)
            )

            y_batch = (
                y_batch.to(device)
            )


            # ----------------------------------------------
            # Zerar gradientes
            # ----------------------------------------------

            optimizer.zero_grad()


            # ----------------------------------------------
            # Forward
            # ----------------------------------------------

            outputs = fusion_model(
                X_sf_batch,
                X_st_batch
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
            # Atualizar parâmetros
            # ----------------------------------------------

            optimizer.step()


            total_loss += (
                loss.item()
            )


            predictions = (
                outputs.argmax(
                    dim=1
                )
            )


            correct += (
                predictions
                == y_batch
            ).sum().item()


            total += (
                y_batch.size(0)
            )


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
            f"Epoch {epoch + 1:03d}/{N_EPOCHS_FUSION} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f}"
        )


    # ======================================================
    # ESTÁGIO 2
    # DESCONGELAR OS DOIS RAMOS
    # ======================================================

    for parameter in model_sf.parameters():
        parameter.requires_grad = True

    for parameter in model_st.parameters():
        parameter.requires_grad = True


    # ======================================================
    # NOVO OTIMIZADOR PARA FINE-TUNING
    # ======================================================

    optimizer = torch.optim.Adam(
        fusion_model.parameters(),
        lr=LEARNING_RATE_FINE_TUNING
    )


    # ======================================================
    # FINE-TUNING DA REDE COMPLETA
    # ======================================================

    for epoch in range(N_EPOCHS_FINE_TUNING):

        fusion_model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for X_sf_batch, X_st_batch, y_batch in train_loader:

            X_sf_batch = X_sf_batch.to(device)
            X_st_batch = X_st_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = fusion_model(
                X_sf_batch,
                X_st_batch
            )

            loss = criterion(
                outputs,
                y_batch
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == y_batch
            ).sum().item()

            total += y_batch.size(0)

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
            f"Fine-tuning {epoch + 1}/{N_EPOCHS_FINE_TUNING} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f}"
        )


    # ======================================================
    # 9. TESTAR O FOLD
    # ======================================================

    fusion_model.eval()

    correct = 0
    total = 0

    # Guardar todas as previsões e labels do fold
    all_predictions = []
    all_labels = []


    with torch.no_grad():

        for (X_sf_batch, X_st_batch, y_batch) in test_loader:

            X_sf_batch = (X_sf_batch.to(device))
            X_st_batch = (X_st_batch.to(device))
            y_batch = (y_batch.to(device))

            outputs = fusion_model(X_sf_batch, X_st_batch)


            predictions = (
                outputs.argmax(
                    dim=1
                )
            )


            correct += (
                predictions
                == y_batch
            ).sum().item()


            total += (
                y_batch.size(0)
            )


            # ----------------------------------------------
            # Guardar previsões
            # ----------------------------------------------

            all_predictions.extend(
                predictions.cpu().numpy()
            )


            # ----------------------------------------------
            # Guardar labels verdadeiras
            # ----------------------------------------------

            all_labels.extend(
                y_batch.cpu().numpy()
            )


    # ======================================================
    # VER DISTRIBUIÇÃO DAS PREVISÕES
    # ======================================================

    print("--- Resultados do Teste ---")

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

    # ======================================================
    # 9.18 SALVAR FUSÃO
    # ======================================================

    torch.save(
        {
            "model_state_dict":
                fusion_model.state_dict(),

            "fold":
                fold,

            "accuracy":
                fold_accuracy
        },

        FUSION_MODEL_DIR
        / f"fusion_fold_{fold}.pth"
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