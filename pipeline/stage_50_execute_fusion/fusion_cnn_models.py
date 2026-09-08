
import torch
import torch.nn as nn



class GatedFusion(nn.Module):

    def __init__(
        self,
        s1_s2_f_t_model,
        s1_s2_t_f_model
    ):

        super().__init__()

        self.s1_s2_f_t_model = s1_s2_f_t_model
        self.s1_s2_t_f_model = s1_s2_t_f_model

        # ==================================================
        # PROJEÇÕES
        # ==================================================

        # 1600 -> 128
        self.s1_s2_f_t_projection = nn.Sequential(
            nn.Linear(1600, 128),
            nn.ReLU()
        )

        # 800 -> 128
        self.s1_s2_t_f_projection = nn.Sequential(
            nn.Linear(800, 128),
            nn.ReLU()
        )


        # ==================================================
        # GATE
        # ==================================================

        # Recebe:
        #
        # s1_s2_f_t = 128
        # s1_s2_t_f = 128
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

        # self.classifier = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.Linear(64, 4),
        # )


    def forward(
        self,
        x_s1_s2_f_t,
        x_s1_s2_t_f
    ):

        # ==================================================
        # EXTRAIR CARACTERÍSTICAS DOS DOIS RAMOS
        # ==================================================

        features_s1_s2_f_t = (
            self.s1_s2_f_t_model
            .extract_features(
                x_s1_s2_f_t
            )
        )

        # (batch, 1600)


        features_s1_s2_t_f = (
            self.s1_s2_t_f_model
            .extract_features(
                x_s1_s2_t_f
            )
        )

        # (batch, 800)


        # ==================================================
        # PROJETAR PARA 128 CARACTERÍSTICAS
        # ==================================================

        features_s1_s2_f_t = self.s1_s2_f_t_projection(features_s1_s2_f_t)

        # (batch, 128)


        features_s1_s2_t_f = self.s1_s2_t_f_projection(features_s1_s2_t_f)

        # (batch, 128)


        # ==================================================
        # CONCATENAR
        # ==================================================

        combined = torch.cat(
            (features_s1_s2_f_t, features_s1_s2_t_f),
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
            gate * features_s1_s2_f_t
            + (1 - gate) * features_s1_s2_t_f
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


