from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# CAMINHOS
# ==========================================================

current_file = Path(__file__).resolve()

PROJECT_ROOT = current_file.parents[2]


INPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_50_execute_fusion"
)


OUTPUT_DIR = INPUT_DIR


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


DEFAULT_SUBJECTS = [
    f"sub-{i:02d}"
    for i in range(1, 11)
]


# ==========================================================
# CONFIGURAÇÃO DOS TRÊS PONTOS DO TREINAMENTO
# ==========================================================

STAGES = {

    "stage_1": {
        "checkpoint_key":
            "stage_1",

        "title":
            "Stage 1 - Frozen Branches",

        "file_suffix":
            "stage_1",
    },

    "stage_2_from_minimum_loss": {
        "checkpoint_key":
            "stage_2_from_stage_1_minimum_loss",

        "title":
            (
                "Stage 2 - Starting from "
                "Stage 1 Minimum Loss"
            ),

        "file_suffix":
            "stage_2_from_stage_1_minimum_loss",
    },

    "stage_2_from_maximum_accuracy": {
        "checkpoint_key":
            "stage_2_from_stage_1_maximum_accuracy",

        "title":
            (
                "Stage 2 - Starting from "
                "Stage 1 Maximum Accuracy"
            ),

        "file_suffix":
            "stage_2_from_stage_1_maximum_accuracy",
    },
}


# ==========================================================
# FUNÇÃO QUE PLOTA UMA DAS TRÊS FIGURAS
# ==========================================================

def plot_fusion_stage(
    fold_files,
    checkpoint_key,
    stage_title,
    file_suffix,
    fusion,
    subject,
    model_dir,
):

    n_folds = len(
        fold_files
    )


    # ======================================================
    # CRIAR FIGURA
    #
    # 3 linhas:
    #
    # 0 -> loss
    # 1 -> accuracy
    # 2 -> tabelas dos folds
    #
    # n_folds + 1 colunas:
    #
    # folds + coluna final de resumo
    # ======================================================

    fig, axs = plt.subplots(
        3,
        n_folds + 1,
        figsize=(22, 10)
    )


    fig.suptitle(
        (
            f"Gated Fusion / "
            f"{subject.capitalize()}\n"
            f"{stage_title}"
        ),
        fontsize=12,
        fontweight="bold"
    )


    # ======================================================
    # RESULTADOS DOS FOLDS
    # ======================================================

    fold_accuracies_ml = []

    fold_losses_ml = []

    fold_accuracies_ma = []

    fold_losses_ma = []


    # ======================================================
    # NÚMERO DE AMOSTRAS
    # ======================================================

    n_training_samples = []

    n_validation_samples = []

    n_test_samples = []

    n_total_samples = []


    # ======================================================
    # LOOP DOS FOLDS
    # ======================================================

    for fold_idx, file_path in enumerate(
        fold_files
    ):

        # --------------------------------------------------
        # Número real do fold
        # --------------------------------------------------

        fold_num = int(
            file_path.stem
            .split("_fold_")[-1]
        )


        # ==================================================
        # CARREGAR CHECKPOINT
        # ==================================================

        checkpoint = torch.load(
            file_path,
            map_location="cpu",
            weights_only=False
        )


        # ==================================================
        # SELECIONAR O ESTÁGIO DESEJADO
        # ==================================================

        stage = checkpoint[
            checkpoint_key
        ]


        training_history = stage[
            "training_history"
        ]


        minimum_loss_model = stage[
            "minimum_loss"
        ]


        maximum_accuracy_model = stage[
            "maximum_accuracy"
        ]


        # ==================================================
        # HISTÓRICO DE TREINAMENTO / VALIDAÇÃO
        # ==================================================

        train_losses_epoch = np.asarray(
            training_history[
                "train_losses_epoch"
            ],
            dtype=np.float64
        )


        val_losses_epoch = np.asarray(
            training_history[
                "val_losses_epoch"
            ],
            dtype=np.float64
        )


        train_accuracies_epoch = np.asarray(
            training_history[
                "train_accuracies_epoch"
            ],
            dtype=np.float64
        )


        val_accuracies_epoch = np.asarray(
            training_history[
                "val_accuracies_epoch"
            ],
            dtype=np.float64
        )


        # ==================================================
        # MODELO DE MENOR VALIDATION LOSS
        # ==================================================

        min_val_loss_epoch = int(
            minimum_loss_model[
                "validation"
            ][
                "epoch"
            ]
        )


        min_val_loss = float(
            minimum_loss_model[
                "validation"
            ][
                "value"
            ]
        )


        val_accuracy_at_min_loss = float(
            val_accuracies_epoch[
                min_val_loss_epoch
            ]
        )


        test_loss_ml = float(
            minimum_loss_model[
                "test"
            ][
                "loss"
            ]
        )


        test_accuracy_ml = float(
            minimum_loss_model[
                "test"
            ][
                "accuracy"
            ]
        )


        all_labels_ml = np.asarray(
            minimum_loss_model[
                "test"
            ][
                "all_labels"
            ],
            dtype=np.int64
        )


        all_predictions_ml = np.asarray(
            minimum_loss_model[
                "test"
            ][
                "all_predictions"
            ],
            dtype=np.int64
        )


        # ==================================================
        # MODELO DE MAIOR VALIDATION ACCURACY
        # ==================================================

        max_val_accuracy_epoch = int(
            maximum_accuracy_model[
                "validation"
            ][
                "epoch"
            ]
        )


        max_val_accuracy = float(
            maximum_accuracy_model[
                "validation"
            ][
                "value"
            ]
        )


        val_loss_at_max_accuracy = float(
            val_losses_epoch[
                max_val_accuracy_epoch
            ]
        )


        test_loss_ma = float(
            maximum_accuracy_model[
                "test"
            ][
                "loss"
            ]
        )


        test_accuracy_ma = float(
            maximum_accuracy_model[
                "test"
            ][
                "accuracy"
            ]
        )


        all_labels_ma = np.asarray(
            maximum_accuracy_model[
                "test"
            ][
                "all_labels"
            ],
            dtype=np.int64
        )


        all_predictions_ma = np.asarray(
            maximum_accuracy_model[
                "test"
            ][
                "all_predictions"
            ],
            dtype=np.int64
        )


        # ==================================================
        # GUARDAR RESULTADOS DO FOLD PARA A TABELA FINAL
        # ==================================================

        fold_accuracies_ml.append(
            test_accuracy_ml
        )


        fold_losses_ml.append(
            test_loss_ml
        )


        fold_accuracies_ma.append(
            test_accuracy_ma
        )


        fold_losses_ma.append(
            test_loss_ma
        )


        # ==================================================
        # NÚMERO DE AMOSTRAS
        # ==================================================

        dataset_info = checkpoint[
            "dataset"
        ]


        n_training_samples.append(
            dataset_info[
                "total_training_samples"
            ]
        )


        n_validation_samples.append(
            dataset_info[
                "total_validation_samples"
            ]
        )


        n_test_samples.append(
            dataset_info[
                "total_test_samples"
            ]
        )


        n_total_samples.append(
            dataset_info[
                "total_samples"
            ]
        )


        # ==================================================
        # ÉPOCAS
        # ==================================================

        epochs = range(
            1,
            len(
                train_losses_epoch
            ) + 1
        )


        # ==================================================
        # GRÁFICO DE LOSS
        # LINHA 0
        # ==================================================

        ax_loss = axs[
            0,
            fold_idx
        ]


        ax_loss.plot(
            epochs,
            train_losses_epoch,
            color="sandybrown",
            label="Train"
        )


        ax_loss.plot(
            epochs,
            val_losses_epoch,
            color="royalblue",
            label="Validation"
        )


        ax_loss.axvline(
            x=min_val_loss_epoch + 1,
            color="forestgreen",
            linestyle=":",
            linewidth=2,
            label="Min Val Loss"
        )


        ax_loss.axvline(
            x=max_val_accuracy_epoch + 1,
            color="violet",
            linestyle=":",
            linewidth=2,
            label="Max Val Acc"
        )


        ax_loss.set_title(
            f"Fold {fold_num} - Loss",
            fontsize=11,
            fontweight="bold"
        )


        ax_loss.grid(
            True,
            linestyle=":",
            alpha=0.5
        )


        ax_loss.tick_params(
            axis="x",
            labelbottom=False
        )


        if fold_idx == 0:

            ax_loss.set_ylabel(
                "Loss",
                fontsize=12
            )


            ax_loss.legend(
                loc="best"
            )


        # ==================================================
        # GRÁFICO DE ACCURACY
        # LINHA 1
        # ==================================================

        ax_acc = axs[
            1,
            fold_idx
        ]


        ax_acc.plot(
            epochs,
            train_accuracies_epoch,
            color="sandybrown",
            label="Train"
        )


        ax_acc.plot(
            epochs,
            val_accuracies_epoch,
            color="royalblue",
            label="Validation"
        )


        ax_acc.axvline(
            x=min_val_loss_epoch + 1,
            color="forestgreen",
            linestyle=":",
            linewidth=2,
            label="Min Val Loss"
        )


        ax_acc.axvline(
            x=max_val_accuracy_epoch + 1,
            color="violet",
            linestyle=":",
            linewidth=2,
            label="Max Val Acc"
        )


        ax_acc.set_title(
            f"Fold {fold_num} - Accuracy",
            fontsize=11,
            fontweight="bold"
        )


        ax_acc.set_xlabel(
            "Epochs"
        )


        ax_acc.grid(
            True,
            linestyle=":",
            alpha=0.5
        )


        if fold_idx == 0:

            ax_acc.set_ylabel(
                "Accuracy",
                fontsize=12
            )


            ax_acc.legend(
                loc="best"
            )


        # ==================================================
        # MINI-TABELA DO FOLD
        # LINHA 2
        # ==================================================

        ax_table = axs[
            2,
            fold_idx
        ]


        ax_table.axis(
            "off"
        )


        cell_text = [

            [
                str(
                    min_val_loss_epoch + 1
                ),

                str(
                    max_val_accuracy_epoch + 1
                ),
            ],

            [
                f"{min_val_loss:.4f}",

                f"{val_loss_at_max_accuracy:.4f}",
            ],

            [
                f"{val_accuracy_at_min_loss:.4f}",

                f"{max_val_accuracy:.4f}",
            ],

            [
                f"{test_loss_ml:.4f}",

                f"{test_loss_ma:.4f}",
            ],

            [
                f"{test_accuracy_ml:.4f}",

                f"{test_accuracy_ma:.4f}",
            ],

            [
                np.bincount(
                    all_labels_ml,
                    minlength=4
                ),

                np.bincount(
                    all_labels_ma,
                    minlength=4
                ),
            ],

            [
                np.bincount(
                    all_predictions_ml,
                    minlength=4
                ),

                np.bincount(
                    all_predictions_ma,
                    minlength=4
                ),
            ],
        ]


        if fold_idx == 0:

            row_labels = [

                "Best Epoch",

                "Val Loss",

                "Val Acc",

                "Test Loss",

                "Test Acc",

                "Test Labels",

                "Test Predictions",
            ]

        else:

            row_labels = None


        mini_table = ax_table.table(

            cellText=cell_text,

            rowLabels=row_labels,

            colLabels=[
                "Min\nVal Loss",
                "Max\nVal Acc"
            ],

            loc="center",

            cellLoc="center",

            colWidths=[
                0.45,
                0.45
            ]
        )


        mini_table.auto_set_font_size(
            False
        )


        mini_table.set_fontsize(
            9
        )


        mini_table.scale(
            1.0,
            1.4
        )


        for col_idx in range(
            2
        ):

            mini_table[
                0,
                col_idx
            ].set_height(
                0.18
            )


        if fold_idx > 0:

            mini_table.scale(
                1.0,
                1.0
            )


    # ======================================================
    # CONVERTER LISTAS EM ARRAYS
    # ======================================================

    fold_accuracies_ml = np.asarray(
        fold_accuracies_ml,
        dtype=np.float64
    )


    fold_losses_ml = np.asarray(
        fold_losses_ml,
        dtype=np.float64
    )


    fold_accuracies_ma = np.asarray(
        fold_accuracies_ma,
        dtype=np.float64
    )


    fold_losses_ma = np.asarray(
        fold_losses_ma,
        dtype=np.float64
    )


    n_training_samples = np.asarray(
        n_training_samples,
        dtype=np.int64
    )


    n_validation_samples = np.asarray(
        n_validation_samples,
        dtype=np.int64
    )


    n_test_samples = np.asarray(
        n_test_samples,
        dtype=np.int64
    )


    n_total_samples = np.asarray(
        n_total_samples,
        dtype=np.int64
    )


    # ======================================================
    # TABELA FINAL DE RESULTADOS DE TESTE
    # LINHA 2 / ÚLTIMA COLUNA
    # ======================================================

    ax_table = axs[
        2,
        n_folds
    ]


    ax_table.axis(
        "off"
    )


    cell_text = [

        [
            f"{fold_accuracies_ml[fold_idx]:.4f}",

            f"{fold_accuracies_ma[fold_idx]:.4f}",
        ]

        for fold_idx in range(
            n_folds
        )
    ]


    fold_accuracies_mean_ml = (
        fold_accuracies_ml.mean()
    )


    fold_accuracies_mean_ma = (
        fold_accuracies_ma.mean()
    )


    fold_accuracies_std_ml = (
        fold_accuracies_ml.std()
    )


    fold_accuracies_std_ma = (
        fold_accuracies_ma.std()
    )


    cell_text.append(
        [
            f"{fold_accuracies_mean_ml:.4f}",

            f"{fold_accuracies_mean_ma:.4f}",
        ]
    )


    cell_text.append(
        [
            f"{fold_accuracies_std_ml:.4f}",

            f"{fold_accuracies_std_ma:.4f}",
        ]
    )


    row_labels = [

        f"Fold {fold_idx + 1}"

        for fold_idx in range(
            n_folds
        )
    ]


    row_labels.append(
        "Mean"
    )


    row_labels.append(
        "Std"
    )


    mini_table = ax_table.table(

        cellText=cell_text,

        rowLabels=row_labels,

        colLabels=[
            "Min\nVal Loss",
            "Max\nVal Acc"
        ],

        loc="center",

        cellLoc="center",

        colWidths=[
            0.45,
            0.45
        ]
    )


    mini_table.auto_set_font_size(
        False
    )


    mini_table.set_fontsize(
        9
    )


    mini_table.scale(
        1.0,
        1.4
    )


    num_rows = len(
        cell_text
    )


    for col_idx in range(
        2
    ):

        mini_table[
            0,
            col_idx
        ].set_height(
            0.15
        )


    mean_row = (
        num_rows - 1
    )


    std_row = (
        num_rows
    )


    for column in [
        0,
        1
    ]:

        mini_table[
            mean_row,
            column
        ].get_text().set_weight(
            "bold"
        )


        mini_table[
            std_row,
            column
        ].get_text().set_weight(
            "bold"
        )


        mini_table[
            mean_row,
            column
        ].set_facecolor(
            "#e6f2ff"
        )


        mini_table[
            std_row,
            column
        ].set_facecolor(
            "#e6f2ff"
        )


    mini_table[
        mean_row,
        -1
    ].get_text().set_weight(
        "bold"
    )


    mini_table[
        std_row,
        -1
    ].get_text().set_weight(
        "bold"
    )


    # ======================================================
    # TABELA DE NÚMERO DE AMOSTRAS
    # LINHA 1 / ÚLTIMA COLUNA
    # ======================================================

    ax_table = axs[
        1,
        n_folds
    ]


    ax_table.axis(
        "off"
    )


    cell_text = [

        [
            f"{n_training_samples[fold_idx]:d}",

            f"{n_validation_samples[fold_idx]:d}",

            f"{n_test_samples[fold_idx]:d}",

            f"{n_total_samples[fold_idx]:d}",
        ]

        for fold_idx in range(
            n_folds
        )
    ]


    row_labels = [

        f"Fold {fold_idx + 1}"

        for fold_idx in range(
            n_folds
        )
    ]


    mini_table = ax_table.table(

        cellText=cell_text,

        rowLabels=row_labels,

        colLabels=[
            "Train",
            "Val",
            "Test",
            "Total"
        ],

        loc="center",

        cellLoc="center",

        colWidths=[
            0.24,
            0.24,
            0.24,
            0.24
        ]
    )


    mini_table.auto_set_font_size(
        False
    )


    mini_table.set_fontsize(
        9
    )


    mini_table.scale(
        1.0,
        1.4
    )


    for col_idx in range(
        4
    ):

        mini_table[
            0,
            col_idx
        ].set_height(
            0.15
        )


    # ======================================================
    # TABELA EXTRA
    # LINHA 0 / ÚLTIMA COLUNA
    #
    # Mostra as losses médias dos folds.
    # ======================================================

    ax_table = axs[
        0,
        n_folds
    ]


    ax_table.axis(
        "off"
    )


    cell_text = [

        [
            f"{fold_losses_ml.mean():.4f}",
            f"{fold_losses_ma.mean():.4f}",
        ],

        [
            f"{fold_losses_ml.std():.4f}",
            f"{fold_losses_ma.std():.4f}",
        ],
    ]


    mini_table = ax_table.table(

        cellText=cell_text,

        rowLabels=[
            "Mean Test Loss",
            "Std Test Loss",
        ],

        colLabels=[
            "Min\nVal Loss",
            "Max\nVal Acc"
        ],

        loc="center",

        cellLoc="center",

        colWidths=[
            0.45,
            0.45
        ]
    )


    mini_table.auto_set_font_size(
        False
    )


    mini_table.set_fontsize(
        9
    )


    mini_table.scale(
        1.0,
        1.4
    )


    # ======================================================
    # LAYOUT
    # ======================================================

    plt.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
            0.95
        ]
    )


    # ======================================================
    # SALVAR
    # ======================================================

    save_dir = (
        model_dir
        / "plots"
    )


    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    figure_path = (
        save_dir
        / (
            f"{fusion}_{subject}_"
            f"{file_suffix}.svg"
        )
    )


    plt.savefig(
        figure_path,
        bbox_inches="tight"
    )


    print(
        f"Figure saved: "
        f"{figure_path}"
    )


    plt.close(
        fig
    )


# ==========================================================
# FUNÇÃO PRINCIPAL
# ==========================================================

def fusion_individual_subject_plots(
    MODEL_DIR,
    fusion="gated_fusion",
    subjects=None,
    
):

    # ======================================================
    # CHECAGEM DE SUBJECTS
    # ======================================================

    if isinstance(
        subjects,
        str
    ):

        subjects = [
            subjects
        ]


    elif subjects is None:

        subjects = (
            DEFAULT_SUBJECTS
        )


    # ======================================================
    # LOOP DOS SUJEITOS
    # ======================================================

    for subject in subjects:

        # SUB_DIR = (
        #     INPUT_DIR
        #     / f"{fusion}_fusion"
        #     / subject
        # )

        SUB_DIR = (
            MODEL_DIR
            / subject
        )
       


        # ==================================================
        # LOCALIZAR CHECKPOINTS
        # ==================================================

        fold_files = list(
            SUB_DIR.glob(
                f"{fusion}_fold_*.pth"
            )
        )


        fold_files = sorted(
            fold_files,
            key=lambda path: int(
                path.stem
                .split("_fold_")[-1]
            )
        )


        if len(
            fold_files
        ) == 0:

            print(
                f"No fold files found for "
                f"{subject} / {fusion}"
            )

            continue


        # ==================================================
        # CRIAR AS TRÊS FIGURAS
        # ==================================================

        for stage_configuration in (
            STAGES.values()
        ):

            plot_fusion_stage(

                fold_files=fold_files,

                checkpoint_key=(
                    stage_configuration[
                        "checkpoint_key"
                    ]
                ),

                stage_title=(
                    stage_configuration[
                        "title"
                    ]
                ),

                file_suffix=(
                    stage_configuration[
                        "file_suffix"
                    ]
                ),

                fusion=fusion,

                subject=subject,

                model_dir=SUB_DIR,
            )


# ==========================================================
# EXECUTAR
# ==========================================================

if __name__ == "__main__":

    fusion_individual_subject_plots(
        fusion="gated_fusion",
        subjects=[
            "sub-01"
        ]
    )
