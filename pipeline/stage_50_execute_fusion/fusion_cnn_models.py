
import torch
import torch.nn as nn


class GatedFusion(nn.Module):

    def __init__(
        self,
        s1_s2_f_t_model,
        s1_s2_t_f_model,
        t_f_s2_s1_model,
        t_f_s1_s2_model,
    ):

        super().__init__()

        self.s1_s2_f_t_model = (
            s1_s2_f_t_model
        )

        self.s1_s2_t_f_model = (
            s1_s2_t_f_model
        )

        self.t_f_s2_s1_model = (
            t_f_s2_s1_model
        )

        self.t_f_s1_s2_model = (
            t_f_s1_s2_model
        )


        # ==================================================
        # PROJEÇÕES
        # ==================================================

        self.s1_s2_f_t_projection = nn.Sequential(
            nn.Linear(1600, 128),
            nn.ReLU()
        )

        self.s1_s2_t_f_projection = nn.Sequential(
            nn.Linear(800, 128),
            nn.ReLU()
        )

        self.t_f_s2_s1_projection = nn.Sequential(
            nn.Linear(800, 128),
            nn.ReLU()
        )

        self.t_f_s1_s2_projection = nn.Sequential(
            nn.Linear(800, 128),
            nn.ReLU()
        )


        # ==================================================
        # GATE
        #
        # Entrada:
        # 4 x 128 = 512
        #
        # Saída:
        # 4 gates para cada uma das 128 características
        #
        # 4 x 128 = 512
        # ==================================================

        self.gate = nn.Linear(
            512,
            512
        )


        # ==================================================
        # CLASSIFICADOR
        # ==================================================

        self.classifier = nn.Linear(
            128,
            4
        )


    def forward(
        self,
        x_s1_s2_f_t,
        x_s1_s2_t_f,
        x_t_f_s2_s1,
        x_t_f_s1_s2,
    ):

        # ==================================================
        # EXTRAIR FEATURES
        # ==================================================

        features_s1_s2_f_t = (
            self.s1_s2_f_t_model
            .extract_features(
                x_s1_s2_f_t
            )
        )

        features_s1_s2_t_f = (
            self.s1_s2_t_f_model
            .extract_features(
                x_s1_s2_t_f
            )
        )

        features_t_f_s2_s1 = (
            self.t_f_s2_s1_model
            .extract_features(
                x_t_f_s2_s1
            )
        )

        features_t_f_s1_s2 = (
            self.t_f_s1_s2_model
            .extract_features(
                x_t_f_s1_s2
            )
        )


        # ==================================================
        # PROJETAR TODOS PARA 128
        # ==================================================

        features_s1_s2_f_t = (
            self.s1_s2_f_t_projection(
                features_s1_s2_f_t
            )
        )

        features_s1_s2_t_f = (
            self.s1_s2_t_f_projection(
                features_s1_s2_t_f
            )
        )

        features_t_f_s2_s1 = (
            self.t_f_s2_s1_projection(
                features_t_f_s2_s1
            )
        )

        features_t_f_s1_s2 = (
            self.t_f_s1_s2_projection(
                features_t_f_s1_s2
            )
        )


        # ==================================================
        # CONCATENAÇÃO PARA CALCULAR OS GATES
        # ==================================================

        combined = torch.cat(
            (
                features_s1_s2_f_t,
                features_s1_s2_t_f,
                features_t_f_s2_s1,
                features_t_f_s1_s2
            ),
            dim=1
        )

        # (batch, 512)


        # ==================================================
        # CALCULAR GATES
        # ==================================================

        gate_logits = self.gate(
            combined
        )

        # (batch, 512)


        gate_logits = gate_logits.view(
            -1,
            4,
            128
        )

        # (batch, 4, 128)


        gate = torch.softmax(
            gate_logits,
            dim=1
        )

        # (batch, 4, 128)


        # ==================================================
        # EMPILHAR FEATURES
        # ==================================================

        features = torch.stack(
            (
                features_s1_s2_f_t,
                features_s1_s2_t_f,
                features_t_f_s2_s1,
                features_t_f_s1_s2
            ),
            dim=1
        )

        # (batch, 4, 128)


        # ==================================================
        # GATED FUSION
        # ==================================================

        fused = (
            gate * features
        ).sum(
            dim=1
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