import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# MODELS FOR PRONOUNCED SPEECH
# =========================================================

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
# WEIGHTED SUM FUSION
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

        # Four learnable global weights
        self.branch_logits = nn.Parameter(
            torch.zeros(4)
        )


    def forward(
        self,
        x_s1_s2_f_t,
        x_s1_s2_t_f,
        x_t_f_s2_s1,
        x_t_f_s1_s2
    ):

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


        # Global branch weights
        weights = torch.softmax(
            self.branch_logits,
            dim=0
        )

        # Weighted sum
        output_logits = (
            weights[0] * logits_1 +
            weights[1] * logits_2 +
            weights[2] * logits_3 +
            weights[3] * logits_4
        )

        return output_logits


# ==========================================================
# SCALAR GATED FUSION
#
# One weight per branch AND per sample.
#
# Gate input:
# 4 x 4 = 16
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

        # 16 concatenated features -> 4 branch weights
        self.gate = nn.Linear(
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

        # Concatenate branch logits for gate
        combined = torch.cat(
            (logits_1, logits_2, logits_3, logits_4),
            dim=1
        )
        # (batch, 16)

        # One scalar weight per branch
        gate_logits = self.gate(combined)
        # (batch, 4)

        gate = torch.softmax(
            gate_logits,
            dim=1
        )
        # (batch, 4)

        # Weighted sum of branch logits
        fused_logits = (
            gate[:, 0:1] * logits_1 +
            gate[:, 1:2] * logits_2 +
            gate[:, 2:3] * logits_3 +
            gate[:, 3:4] * logits_4
        )
        # (batch, 4)

        return fused_logits


# ==========================================================
# CLASS-WISE GATED FUSION
#
# One weight per:
# sample x branch x class
#
# Gate input:
# 4 branches x 4 classes = 16
#
# Gate output:
# 4 branches x 4 classes = 16
# ==========================================================

class ClassWiseGatedFusion(nn.Module):

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

        # 16 concatenated branch logits
        # ->
        # 4 branches x 4 classes
        self.gate = nn.Linear(
            16,
            16
        )


    def forward(
        self,
        x_s1_s2_f_t,
        x_s1_s2_t_f,
        x_t_f_s2_s1,
        x_t_f_s1_s2
    ):

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
        combined = torch.cat(
            (logits_1, logits_2, logits_3, logits_4),
            dim=1
        )
        # (batch, 16)

        # Generate gate logits
        gate_logits = self.gate(combined)
        # (batch, 16)

        # Organize by branch and class
        gate_logits = gate_logits.view(
            -1,
            4,
            4
        )
        # (batch, branch, class)

        # For each class, branch weights sum to 1
        gate = torch.softmax(
            gate_logits,
            dim=1
        )
        # (batch, branch, class)

        # Organize branch logits
        branch_logits = torch.stack(
            (logits_1, logits_2, logits_3, logits_4),
            dim=1
        )
        # (batch, branch, class)

        # Class-wise weighted sum across branches
        fused_logits = (
            gate * branch_logits
        ).sum(dim=1)
        # (batch, class)
        # (batch, 4)

        return fused_logits


class MajorityVotingFusion(nn.Module):

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

        # ==========================================================
        # BRANCH LOGITS
        # ==========================================================

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


        # ==========================================================
        # BRANCH PROBABILITIES
        # ==========================================================

        probabilities_1 = torch.softmax(
            logits_1,
            dim=1
        )

        probabilities_2 = torch.softmax(
            logits_2,
            dim=1
        )

        probabilities_3 = torch.softmax(
            logits_3,
            dim=1
        )

        probabilities_4 = torch.softmax(
            logits_4,
            dim=1
        )


        # ==========================================================
        # BRANCH PREDICTIONS
        # ==========================================================

        prediction_1 = torch.argmax(
            probabilities_1,
            dim=1
        )

        prediction_2 = torch.argmax(
            probabilities_2,
            dim=1
        )

        prediction_3 = torch.argmax(
            probabilities_3,
            dim=1
        )

        prediction_4 = torch.argmax(
            probabilities_4,
            dim=1
        )


        # ==========================================================
        # MAJORITY VOTING
        # ==========================================================

        predictions = torch.stack(
            (
                prediction_1,
                prediction_2,
                prediction_3,
                prediction_4
            ),
            dim=1
        )
        # (batch, branch)
        # (batch, 4)


        one_hot_predictions = F.one_hot(
            predictions,
            num_classes=4
        )
        # (batch, branch, class)
        # (batch, 4, 4)


        votes = one_hot_predictions.sum(
            dim=1
        )
        # (batch, class)
        # (batch, 4)


        # ==========================================================
        # BRANCH PROBABILITIES STACK
        # ==========================================================

        probabilities = torch.stack(
            (
                probabilities_1,
                probabilities_2,
                probabilities_3,
                probabilities_4
            ),
            dim=1
        )
        # (batch, branch, class)
        # (batch, 4, 4)


        # ==========================================================
        # KEEP ONLY THE PROBABILITY OF THE CLASS
        # CHOSEN BY EACH BRANCH
        # ==========================================================

        voted_probabilities = (
            probabilities * one_hot_predictions
        )
        # (batch, branch, class)
        # (batch, 4, 4)


        # ==========================================================
        # FIRST CRITERION:
        # NUMBER OF VOTES
        # ==========================================================

        max_votes = votes.max(
            dim=1,
            keepdim=True
        ).values
        # (batch, 1)


        classes_with_max_votes = (
            votes == max_votes
        )
        # (batch, class)
        # (batch, 4)


        # ==========================================================
        # SECOND CRITERION:
        # SUM OF PROBABILITIES FROM BRANCHES THAT VOTED
        # FOR EACH CLASS
        # ==========================================================

        probability_sum = voted_probabilities.sum(
            dim=1
        )
        # (batch, class)
        # (batch, 4)


        probability_sum_masked = probability_sum.masked_fill(
            ~classes_with_max_votes,
            float("-inf")
        )
        # (batch, class)


        max_probability_sum = probability_sum_masked.max(
            dim=1,
            keepdim=True
        ).values
        # (batch, 1)


        classes_with_max_probability_sum = (
            probability_sum_masked
            == max_probability_sum
        )
        # (batch, class)
        # (batch, 4)


        # ==========================================================
        # THIRD CRITERION:
        # HIGHEST INDIVIDUAL PROBABILITY AMONG THE BRANCHES
        # THAT VOTED FOR EACH CLASS
        # ==========================================================

        max_individual_probability = voted_probabilities.max(
            dim=1
        ).values
        # (batch, class)
        # (batch, 4)


        max_individual_probability_masked = (
            max_individual_probability.masked_fill(
                ~classes_with_max_probability_sum,
                float("-inf")
            )
        )
        # (batch, class)
        # (batch, 4)


        # ==========================================================
        # FINAL PREDICTION
        # ==========================================================

        final_prediction = torch.argmax(
            max_individual_probability_masked,
            dim=1
        )
        # (batch,)


        return final_prediction

    