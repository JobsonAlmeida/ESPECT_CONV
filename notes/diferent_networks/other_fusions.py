
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

