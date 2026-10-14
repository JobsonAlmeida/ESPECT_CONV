import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# MODELS FOR PRONOUNCED SPEECH
#
# This models use the probabilities to produce the fusions
#
# =========================================================

# ==========================================================
# MAJORITY VOTING
# ==========================================================

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
        # SUM OF PROBABILITIES FROM ALL FOUR BRANCHES
        # ==========================================================

        probability_sum = probabilities.sum(
            dim=1
        )
        # (batch, class)
        # (batch, 4)


        probability_sum_masked = probability_sum.masked_fill(
            ~classes_with_max_votes,
            float("-inf")
        )
        # (batch, class)
        # (batch, 4)


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
        # HIGHEST INDIVIDUAL PROBABILITY AMONG ALL FOUR BRANCHES
        # ==========================================================

        max_individual_probability = probabilities.max(
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



# ==========================================================
# MEAN LATE FUSION
#
# Mean of probabilities after Softmax.
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
        # BRANCH PROBABILITIES
        # =====================================================

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


        # =====================================================
        # MEAN LATE FUSION
        # =====================================================

        output_probabilities = (
            probabilities_1
            + probabilities_2
            + probabilities_3
            + probabilities_4
        ) / 4.0
        # (batch, 4)


        # =====================================================
        # CONVERT PROBABILITIES TO LOG-PROBABILITIES
        # =====================================================

        output_log_probabilities = torch.log(
            output_probabilities.clamp_min(1e-12)
        )
        # (batch, 4)

        return output_log_probabilities

# ==========================================================
# CONCATENATION FUSION
#
# Concatenation of probabilities after Softmax.
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

        # 4 branches x 4 probabilities = 16
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

        # ==================================================
        # BRANCH LOGITS
        # ==================================================

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


        # ==================================================
        # BRANCH PROBABILITIES
        # ==================================================

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


        # ==================================================
        # CONCATENATE BRANCH PROBABILITIES
        # ==================================================

        concatenated_probabilities = torch.cat(
            (
                probabilities_1,
                probabilities_2,
                probabilities_3,
                probabilities_4
            ),
            dim=1
        )
        # (batch, 4) x 4 -> (batch, 16)


        # ==================================================
        # LINEAR CLASSIFIER
        # ==================================================

        output_logits = self.linear(
            concatenated_probabilities
        )
        # (batch, 16) -> (batch, 4)


        return output_logits
 


# ==========================================================
# WEIGHTED SUM FUSION
#
# One global weight per branch.
# The weights are learned during training.
# Weighted sum of probabilities after Softmax.
# ==========================================================

class BranchWiseWeightedSumFusion(nn.Module):

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

        # ==================================================
        # BRANCH LOGITS
        # ==================================================

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


        # ==================================================
        # BRANCH PROBABILITIES
        # ==================================================

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


        # ==================================================
        # GLOBAL BRANCH WEIGHTS
        # ==================================================

        weights = torch.softmax(
            self.branch_logits,
            dim=0
        )
        # (4,)


        # ==================================================
        # WEIGHTED SUM OF PROBABILITIES
        # ==================================================

        output_probabilities = (
            weights[0] * probabilities_1 +
            weights[1] * probabilities_2 +
            weights[2] * probabilities_3 +
            weights[3] * probabilities_4
        )
        # (batch, class)
        # (batch, 4)


        # ==================================================
        # CONVERT PROBABILITIES TO LOG-PROBABILITIES
        # ==================================================

        output_log_probabilities = torch.log(
            output_probabilities.clamp_min(1e-12)
        )
        # (batch, class)
        # (batch, 4)


        return output_log_probabilities



class BranchAndClassWiseWeightedSumFusion(nn.Module):


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

        # One learnable weight per branch and class
        self.branch_logits = nn.Parameter(
            torch.zeros(4, 4)
        )


    def forward(
        self,
        x_s1_s2_f_t,
        x_s1_s2_t_f,
        x_t_f_s2_s1,
        x_t_f_s1_s2
    ):

        # ==================================================
        # BRANCH LOGITS
        # ==================================================

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


        # ==================================================
        # BRANCH PROBABILITIES
        # ==================================================

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


        # ==================================================
        # CLASS-WISE BRANCH WEIGHTS
        # ==================================================

        weights = torch.softmax(
            self.branch_logits,
            dim=0
        )
        # (branch, class)
        # (4, 4)


        # ==================================================
        # WEIGHTED PROBABILITY SUM
        # ==================================================

        output_scores = (
            weights[0] * probabilities_1 +
            weights[1] * probabilities_2 +
            weights[2] * probabilities_3 +
            weights[3] * probabilities_4
        )
        # (batch, class)
        # (batch, 4)


        # Normalize weighted scores
        output_probabilities = (
            output_scores /
            output_scores.sum(
                dim=1,
                keepdim=True
            ).clamp_min(1e-12)
        )

        # Convert probabilities to log-probabilities
        output_log_probabilities = torch.log(
            output_probabilities.clamp_min(1e-12)
        )

        return output_log_probabilities
    

# ==========================================================
# BRANCH-WISE SCALAR GATED FUSION
#
# One weight per branch AND per sample.
#
# Gate input:
# 4 x 4 = 16 probabilities
#
# Gate output:
# 4 scalar weights
# ==========================================================

class BranchWiseScalarGatedFusion(nn.Module):

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

        # 16 concatenated probabilities -> 4 branch weights
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

        # ==================================================
        # BRANCH LOGITS
        # ==================================================

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


        # ==================================================
        # BRANCH PROBABILITIES
        # ==================================================

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


        # ==================================================
        # CONCATENATE BRANCH PROBABILITIES
        # ==================================================

        combined = torch.cat(
            (
                probabilities_1,
                probabilities_2,
                probabilities_3,
                probabilities_4
            ),
            dim=1
        )
        # (batch, 16)


        # ==================================================
        # GATE WEIGHTS
        # ==================================================

        gate_logits = self.gate(
            combined
        )
        # (batch, 4)

        gate = torch.softmax(
            gate_logits,
            dim=1
        )
        # (batch, 4)


        # ==================================================
        # WEIGHTED SUM OF PROBABILITIES
        # ==================================================

        output_probabilities = (
            gate[:, 0:1] * probabilities_1 +
            gate[:, 1:2] * probabilities_2 +
            gate[:, 2:3] * probabilities_3 +
            gate[:, 3:4] * probabilities_4
        )
        # (batch, 4)


        # ==================================================
        # CONVERT PROBABILITIES TO LOG-PROBABILITIES
        # ==================================================

        output_log_probabilities = torch.log(
            output_probabilities.clamp_min(1e-12)
        )
        # (batch, 4)


        return output_log_probabilities


# ==========================================================
# BRANCH-AND-CLASS-WISE SCALAR GATED FUSION
# 
# One weight per:
# sample x branch x class
#
# Gate input:
# 4 branches x 4 classes = 16 probabilities
#
# Gate output:
# 4 branches x 4 classes = 16 weights
# ==========================================================

class BranchAndClassWiseScalarGatedFusion(nn.Module):

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

        # 16 concatenated branch probabilities
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

        # ==================================================
        # BRANCH LOGITS
        # ==================================================

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


        # ==================================================
        # BRANCH PROBABILITIES
        # ==================================================

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


        # ==================================================
        # CONCATENATE BRANCH PROBABILITIES
        # ==================================================

        combined = torch.cat(
            (
                probabilities_1,
                probabilities_2,
                probabilities_3,
                probabilities_4
            ),
            dim=1
        )
        # (batch, 16)


        # ==================================================
        # GENERATE GATE WEIGHTS
        # ==================================================

        gate_logits = self.gate(
            combined
        )
        # (batch, 16)


        # ==================================================
        # ORGANIZE BY BRANCH AND CLASS
        # ==================================================

        gate_logits = gate_logits.view(
            -1,
            4,
            4
        )
        # (batch, branch, class)


        # ==================================================
        # NORMALIZE GATE WEIGHTS ACROSS BRANCHES
        # ==================================================

        gate = torch.softmax(
            gate_logits,
            dim=1
        )
        # (batch, branch, class)


        # ==================================================
        # ORGANIZE BRANCH PROBABILITIES
        # ==================================================

        branch_probabilities = torch.stack(
            (
                probabilities_1,
                probabilities_2,
                probabilities_3,
                probabilities_4
            ),
            dim=1
        )
        # (batch, branch, class)


        # ==================================================
        # CLASS-WISE WEIGHTED SUM ACROSS BRANCHES
        # ==================================================

        output_scores = (
            gate * branch_probabilities
        ).sum(dim=1)
        # (batch, 4)


        # ==================================================
        # NORMALIZE OUTPUT SCORES
        # ==================================================

        output_probabilities = (
            output_scores /
            output_scores.sum(
                dim=1,
                keepdim=True
            ).clamp_min(1e-12)
        )
        # (batch, 4)


        # ==================================================
        # CONVERT PROBABILITIES TO LOG-PROBABILITIES
        # ==================================================

        output_log_probabilities = torch.log(
            output_probabilities.clamp_min(1e-12)
        )
        # (batch, 4)


        return output_log_probabilities
