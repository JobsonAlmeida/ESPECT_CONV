
# ==========================================================
# 6. ATTENTION FUSION
#
# Each branch becomes one token with 128 features.
#
# Input to attention:
# (batch, 4, 128)
#
# Self-attention operates across the 4 branches.
# ==========================================================

class AttentionFusion(nn.Module):

    def __init__(
        self,
        s1_s2_f_t_model,
        s1_s2_t_f_model,
        t_f_s2_s1_model,
        t_f_s1_s2_model
    ):

        super().__init__()

        # Branch models
        self.s1_s2_f_t_model = s1_s2_f_t_model
        self.s1_s2_t_f_model = s1_s2_t_f_model
        self.t_f_s2_s1_model = t_f_s2_s1_model
        self.t_f_s1_s2_model = t_f_s1_s2_model

        # Feature projections
        self.s1_s2_f_t_projection = nn.Sequential(
            nn.Linear(5346, 128),
            nn.ReLU()
        )

        self.s1_s2_t_f_projection = nn.Sequential(
            nn.Linear(810, 128),
            nn.ReLU()
        )

        self.t_f_s2_s1_projection = nn.Sequential(
            nn.Linear(2970, 128),
            nn.ReLU()
        )

        self.t_f_s1_s2_projection = nn.Sequential(
            nn.Linear(2970, 128),
            nn.ReLU()
        )

        # Self-attention across the four branches
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )

        self.classifier = nn.Linear(
            128,
            4
        )


    def forward(
        self,
        x_s1_s2_f_t,
        x_s1_s2_t_f,
        x_t_f_s2_s1,
        x_t_f_s1_s2
    ):

        f1 = self.s1_s2_f_t_model.extract_features(
            x_s1_s2_f_t
        )

        f2 = self.s1_s2_t_f_model.extract_features(
            x_s1_s2_t_f
        )

        f3 = self.t_f_s2_s1_model.extract_features(
            x_t_f_s2_s1
        )

        f4 = self.t_f_s1_s2_model.extract_features(
            x_t_f_s1_s2
        )

        # Projection to 128 features
        f1 = self.s1_s2_f_t_projection(f1)
        f2 = self.s1_s2_t_f_projection(f2)
        f3 = self.t_f_s2_s1_projection(f3)
        f4 = self.t_f_s1_s2_projection(f4)

        # Four branch tokens
        features = torch.stack(
            (f1, f2, f3, f4),
            dim=1
        )

        # (batch, 4, 128)

        attended_features, attention_weights = (
            self.attention(
                features,
                features,
                features
            )
        )

        # attended_features:
        # (batch, 4, 128)

        # Aggregate the four attended branch representations
        fused = attended_features.mean(
            dim=1
        )

        # (batch, 128)

        output = self.classifier(fused)

        return output


# ==========================================================
# 7. BILINEAR FUSION
#
# Pairwise bilinear interactions between branches.
#
# There are six pairs:
#
# f1-f2
# f1-f3
# f1-f4
# f2-f3
# f2-f4
# f3-f4
#
# Each bilinear layer:
# 128 x 128 -> 128
#
# The six outputs are averaged.
# ==========================================================

class BilinearFusion(nn.Module):

    def __init__(
        self,
        s1_s2_f_t_model,
        s1_s2_t_f_model,
        t_f_s2_s1_model,
        t_f_s1_s2_model
    ):

        super().__init__()

        # Branch models
        self.s1_s2_f_t_model = s1_s2_f_t_model
        self.s1_s2_t_f_model = s1_s2_t_f_model
        self.t_f_s2_s1_model = t_f_s2_s1_model
        self.t_f_s1_s2_model = t_f_s1_s2_model

        # Feature projections
        self.s1_s2_f_t_projection = nn.Sequential(
            nn.Linear(5346, 128),
            nn.ReLU()
        )

        self.s1_s2_t_f_projection = nn.Sequential(
            nn.Linear(810, 128),
            nn.ReLU()
        )

        self.t_f_s2_s1_projection = nn.Sequential(
            nn.Linear(2970, 128),
            nn.ReLU()
        )

        self.t_f_s1_s2_projection = nn.Sequential(
            nn.Linear(2970, 128),
            nn.ReLU()
        )

        # Pairwise bilinear interactions
        self.bilinear_12 = nn.Bilinear(
            128,
            128,
            128
        )

        self.bilinear_13 = nn.Bilinear(
            128,
            128,
            128
        )

        self.bilinear_14 = nn.Bilinear(
            128,
            128,
            128
        )

        self.bilinear_23 = nn.Bilinear(
            128,
            128,
            128
        )

        self.bilinear_24 = nn.Bilinear(
            128,
            128,
            128
        )

        self.bilinear_34 = nn.Bilinear(
            128,
            128,
            128
        )

        self.relu = nn.ReLU()

        self.classifier = nn.Linear(
            128,
            4
        )


    def forward(
        self,
        x_s1_s2_f_t,
        x_s1_s2_t_f,
        x_t_f_s2_s1,
        x_t_f_s1_s2
    ):

        f1 = self.s1_s2_f_t_model.extract_features(
            x_s1_s2_f_t
        )

        f2 = self.s1_s2_t_f_model.extract_features(
            x_s1_s2_t_f
        )

        f3 = self.t_f_s2_s1_model.extract_features(
            x_t_f_s2_s1
        )

        f4 = self.t_f_s1_s2_model.extract_features(
            x_t_f_s1_s2
        )

        # Projection to 128 features
        f1 = self.s1_s2_f_t_projection(f1)
        f2 = self.s1_s2_t_f_projection(f2)
        f3 = self.t_f_s2_s1_projection(f3)
        f4 = self.t_f_s1_s2_projection(f4)

        # Pairwise bilinear interactions
        b12 = self.relu(
            self.bilinear_12(f1, f2)
        )

        b13 = self.relu(
            self.bilinear_13(f1, f3)
        )

        b14 = self.relu(
            self.bilinear_14(f1, f4)
        )

        b23 = self.relu(
            self.bilinear_23(f2, f3)
        )

        b24 = self.relu(
            self.bilinear_24(f2, f4)
        )

        b34 = self.relu(
            self.bilinear_34(f3, f4)
        )

        # Mean of the six pairwise interactions
        fused = (
            b12 +
            b13 +
            b14 +
            b23 +
            b24 +
            b34
        ) / 6.0

        # (batch, 128)

        output = self.classifier(fused)

        return output