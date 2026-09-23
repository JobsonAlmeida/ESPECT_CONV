import torch
import torch.nn as nn

import torch.nn as nn


# ==========================================================
# MEAN LATE FUSION
# ==========================================================

class MeanFusion(nn.Module):

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

    def forward(
        self,
        x_s1_s2_f_t,
        x_s1_s2_t_f,
        x_t_f_s2_s1,
        x_t_f_s1_s2
    ):

        # =====================================================
        # LOGITS FROM EACH BRANCH
        # =====================================================

        logits_1 = self.s1_s2_f_t_model(
            x_s1_s2_f_t
        )
        # (batch, 4)

        logits_2 = self.s1_s2_t_f_model(
            x_s1_s2_t_f
        )
        # (batch, 4)

        logits_3 = self.t_f_s2_s1_model(
            x_t_f_s2_s1
        )
        # (batch, 4)

        logits_4 = self.t_f_s1_s2_model(
            x_t_f_s1_s2
        )
        # (batch, 4)


        # =====================================================
        # MEAN LATE FUSION
        # =====================================================

        logits = (
            logits_1
            + logits_2
            + logits_3
            + logits_4
        ) / 4.0

        # (batch, 4)

        return logits



# ==========================================================
# MULTIPLICATION FUSION
# ==========================================================

class MultiplicationFusion(nn.Module):

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

    def forward(
        self,
        x_s1_s2_f_t,
        x_s1_s2_t_f,
        x_t_f_s2_s1,
        x_t_f_s1_s2
    ):

        # Branch logits
        logits_1 = self.s1_s2_f_t_model(
            x_s1_s2_f_t
        )

        logits_2 = self.s1_s2_t_f_model(
            x_s1_s2_t_f
        )

        logits_3 = self.t_f_s2_s1_model(
            x_t_f_s2_s1
        )

        logits_4 = self.t_f_s1_s2_model(
            x_t_f_s1_s2
        )

        # Element-wise multiplication of branch logits
        # (batch, 4) -> (batch, 4)
        output_logits = (
            logits_1
            * logits_2
            * logits_3
            * logits_4
        )

        return output_logits


# ========================================================
# Multiplication Log Softmax Fusion
# ========================================================


class MultiplicationLogSoftmaxFusion(nn.Module):

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

    def forward(
        self,
        x_s1_s2_f_t,
        x_s1_s2_t_f,
        x_t_f_s2_s1,
        x_t_f_s1_s2
    ):

        # Branch logits
        logits_1 = self.s1_s2_f_t_model(
            x_s1_s2_f_t
        )

        logits_2 = self.s1_s2_t_f_model(
            x_s1_s2_t_f
        )

        logits_3 = self.t_f_s2_s1_model(
            x_t_f_s2_s1
        )

        logits_4 = self.t_f_s1_s2_model(
            x_t_f_s1_s2
        )

        # Convert logits to log-probabilities
        log_probabilities_1 = torch.log_softmax(
            logits_1,
            dim=1
        )

        log_probabilities_2 = torch.log_softmax(
            logits_2,
            dim=1
        )

        log_probabilities_3 = torch.log_softmax(
            logits_3,
            dim=1
        )

        log_probabilities_4 = torch.log_softmax(
            logits_4,
            dim=1
        )

        # Equivalent to element-wise multiplication
        # of branch probabilities in log-space
        output_logits = (
            log_probabilities_1
            + log_probabilities_2
            + log_probabilities_3
            + log_probabilities_4
        )

        return output_logits


    
# ==========================================================
# CONCATENATION FUSION
# ==========================================================

class ConcatenationFusion(nn.Module):

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

        # 4 branches x 4 logits = 16
        self.linear = nn.Linear(
            16,
            4
        )

    def forward(
        self,
        x_s1_s2_f_t,
        x_s1_s2_t_f,
        x_t_f_s2_s1,
        x_t_f_s1_s2
    ):

        # Branch logits
        logits_1 = self.s1_s2_f_t_model(
            x_s1_s2_f_t
        )

        logits_2 = self.s1_s2_t_f_model(
            x_s1_s2_t_f
        )

        logits_3 = self.t_f_s2_s1_model(
            x_t_f_s2_s1
        )

        logits_4 = self.t_f_s1_s2_model(
            x_t_f_s1_s2
        )

        # Concatenate branch logits
        # (batch, 4) x 4 -> (batch, 16)
        concatenated_logits = torch.cat(
            (
                logits_1,
                logits_2,
                logits_3,
                logits_4
            ),
            dim=1
        )

        # (batch, 16) -> (batch, 4)
        output_logits = self.linear(
            concatenated_logits
        )

        return output_logits


# ==========================================================
# 3. WEIGHTED SUM FUSION
#
# One global weight per branch.
# The weights are learned during training.
# ==========================================================

class WeightedSumFusion(nn.Module):

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

        # Four learnable global weights
        self.branch_logits = nn.Parameter(
            torch.zeros(4)
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

        # Global branch weights
        weights = torch.softmax(
            self.branch_logits,
            dim=0
        )

        # Weighted sum
        fused = (
            weights[0] * f1 +
            weights[1] * f2 +
            weights[2] * f3 +
            weights[3] * f4
        )

        output = self.classifier(fused)

        return output


# ==========================================================
# 4. SCALAR GATED FUSION
#
# One weight per branch AND per sample.
#
# Gate input:
# 4 x 128 = 512
#
# Gate output:
# 4 scalar weights
# ==========================================================

class ScalarGatedFusion(nn.Module):

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

        # 512 concatenated features -> 4 branch weights
        self.gate = nn.Linear(
            512,
            4
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

        # Concatenate features for gate
        combined = torch.cat(
            (f1, f2, f3, f4),
            dim=1
        )
        # (batch, 512)

        # One scalar weight per branch
        gate_logits = self.gate(combined)
        # (batch, 4)

        gate = torch.softmax(
            gate_logits,
            dim=1
        )
        # (batch, 4)

        # Weighted sum
        fused = (
            gate[:, 0:1] * f1 +
            gate[:, 1:2] * f2 +
            gate[:, 2:3] * f3 +
            gate[:, 3:4] * f4
        )

        # (batch, 128)

        output = self.classifier(fused)

        return output


# ==========================================================
# 5. FEATURE-WISE GATED FUSION
#
# One weight per:
# sample x branch x feature
#
# Gate:
# 512 -> 512
#
# Then reshape:
# (batch, 512)
#       ->
# (batch, 4, 128)
# ==========================================================

class FeatureGatedFusion(nn.Module):

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

        # 512 -> 4 x 128
        self.gate = nn.Linear(
            512,
            512
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

        # Concatenated representation
        combined = torch.cat(
            (f1, f2, f3, f4),
            dim=1
        )
        # (batch, 512)

        # Generate 4 x 128 gate logits
        gate_logits = self.gate(combined)
        # (batch, 512)

        gate_logits = gate_logits.view(
            -1,
            4,
            128
        )
        # (batch, 4, 128)

        # For each feature, branch weights sum to 1
        gate = torch.softmax(
            gate_logits,
            dim=1
        )
        # (batch, 4, 128)

        # Stack branch features
        features = torch.stack(
            (f1, f2, f3, f4),
            dim=1
        )
        # (batch, 4, 128)

        # Feature-wise weighted sum
        fused = (
            gate * features
        ).sum(dim=1)

        # (batch, 128)

        output = self.classifier(fused)

        return output


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