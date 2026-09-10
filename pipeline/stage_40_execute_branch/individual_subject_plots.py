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
    / "stage_40_execute_branch"
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
# FUNÇÃO PRINCIPAL
# ==========================================================

def individual_subject_plots(
    branch="space1_space2_frequency_time",
    subjects=None,
):

    # ======================================================
    # CHECAGEM DA VARIÁVEL SUBJECTS
    # ======================================================

    if isinstance(
        subjects,
        str
    ):

        # Se o usuário passar uma única string,
        # transforma em uma lista com um elemento.

        subjects = [
            subjects
        ]


    elif subjects is None:

        subjects = (
            DEFAULT_SUBJECTS
        )


    # ======================================================
    # LOOP PARA CADA SUJEITO
    # ======================================================

    for subject in subjects:

        MODEL_DIR = (
            INPUT_DIR
            / f"{branch}_branch"
            / subject
        )


        # ==================================================
        # CONFIGURAÇÕES ESPECÍFICAS DE CADA RAMO
        # ==================================================

        match branch:

            case "space1_space2_frequency_time":

                title = (
                    "Space1 x Space2 x Frequency "
                    "- Channel: Time"
                )


            case "space1_space2_time_frequency":

                title = (
                    "Space1 x Space2 x Time "
                    "- Channel: Frequency"
                )


            case "time_frequency_space2_space1":

                title = (
                    "Time x Frequency x Space2 "
                    "- Channel: Space1"
                )


            case "time_frequency_space1_space2":

                title = (
                    "Time x Frequency x Space1 "
                    "- Channel: Space2"
                )


            case _:

                raise ValueError(
                    f"Invalid branch: {branch}"
                )


        # ==================================================
        # LOCALIZAR CHECKPOINTS DOS FOLDS
        # ==================================================

        fold_files = list(
            MODEL_DIR.glob(
                f"{branch}_fold_*.pth"
            )
        )


        # --------------------------------------------------
        # Ordenação numérica do fold
        #
        # Evita:
        #
        # fold_1
        # fold_10
        # fold_2
        # ...
        # --------------------------------------------------

        fold_files = sorted(
            fold_files,
            key=lambda path: int(
                path.stem
                .split("_fold_")[-1]
            )
        )


        n_folds = len(
            fold_files
        )


        if n_folds == 0:

            print(
                f"No fold files found for "
                f"{subject} / {branch}"
            )

            continue


        # ==================================================
        # ARQUIVO DE RESUMO
        # ==================================================

        summary_file = (
            MODEL_DIR
            / f"{branch}_summary.pth"
        )


        if not summary_file.exists():

            raise FileNotFoundError(
                f"Summary file not found:\n"
                f"{summary_file}"
            )


        # ==================================================
        # CARREGAR RESUMO DOS FOLDS
        # ==================================================

        summary_checkpoint = torch.load(
            summary_file,
            map_location="cpu",
            weights_only=False
        )


        fold_accuracies_ml = np.asarray(
            summary_checkpoint[
                "minimum_loss_model"
            ][
                "fold_accuracies"
            ],
            dtype=np.float64
        )


        fold_losses_ml = np.asarray(
            summary_checkpoint[
                "minimum_loss_model"
            ][
                "fold_losses"
            ],
            dtype=np.float64
        )


        fold_accuracies_ma = np.asarray(
            summary_checkpoint[
                "maximum_accuracy_model"
            ][
                "fold_accuracies"
            ],
            dtype=np.float64
        )


        fold_losses_ma = np.asarray(
            summary_checkpoint[
                "maximum_accuracy_model"
            ][
                "fold_losses"
            ],
            dtype=np.float64
        )


        # ==================================================
        # VERIFICAÇÃO DO RESUMO
        # ==================================================

        if len(
            fold_accuracies_ml
        ) != n_folds:

            raise ValueError(
                "Number of minimum-loss accuracies "
                "in summary differs from number "
                "of fold files."
            )


        if len(
            fold_accuracies_ma
        ) != n_folds:

            raise ValueError(
                "Number of maximum-accuracy accuracies "
                "in summary differs from number "
                "of fold files."
            )


        # ==================================================
        # CRIAR FIGURA
        #
        # 3 linhas
        #
        # n_folds colunas para os folds
        # +
        # 1 coluna para informações gerais
        # ==================================================

        fig, axs = plt.subplots(
            3,
            n_folds + 1,
            figsize=(22, 10)
        )


        fig.suptitle(
            f"{title} / {subject.capitalize()}",
            fontsize=12,
            fontweight="bold"
        )


        # ==================================================
        # ARRAYS PARA NÚMERO DE AMOSTRAS
        # ==================================================

        n_training_samples = []

        n_validation_samples = []

        n_test_samples = []

        n_total_samples = []


        # ==================================================
        # LOOP DOS FOLDS
        # ==================================================

        for fold_idx, file_path in enumerate(
            fold_files
        ):

            # --------------------------------------------------
            # Extrair número real do fold do nome do arquivo
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
            # HISTÓRICO DE TREINAMENTO / VALIDAÇÃO
            # ==================================================

            train_losses_epoch = np.asarray(
                checkpoint[
                    "train_losses_epoch"
                ],
                dtype=np.float64
            )


            val_losses_epoch = np.asarray(
                checkpoint[
                    "val_losses_epoch"
                ],
                dtype=np.float64
            )


            train_accuracies_epoch = np.asarray(
                checkpoint[
                    "train_accuracies_epoch"
                ],
                dtype=np.float64
            )


            val_accuracies_epoch = np.asarray(
                checkpoint[
                    "val_accuracies_epoch"
                ],
                dtype=np.float64
            )


            # ==================================================
            # MODELO DE MENOR VALIDATION LOSS
            # ==================================================

            minimum_loss_model = checkpoint[
                "minimum_loss_model"
            ]


            min_val_loss_epoch = int(
                minimum_loss_model[
                    "epoch"
                ]
            )


            min_val_loss = float(
                minimum_loss_model[
                    "validation_loss"
                ]
            )


            val_accuracy_at_min_loss = float(
                minimum_loss_model[
                    "validation_accuracy"
                ]
            )


            test_loss_ml = float(
                minimum_loss_model[
                    "test_loss"
                ]
            )


            test_accuracy_ml = float(
                minimum_loss_model[
                    "test_accuracy"
                ]
            )


            all_labels_ml = np.asarray(
                minimum_loss_model[
                    "all_labels"
                ],
                dtype=np.int64
            )


            all_predictions_ml = np.asarray(
                minimum_loss_model[
                    "all_predictions"
                ],
                dtype=np.int64
            )


            # ==================================================
            # MODELO DE MAIOR VALIDATION ACCURACY
            # ==================================================

            maximum_accuracy_model = checkpoint[
                "maximum_accuracy_model"
            ]


            max_val_accuracy_epoch = int(
                maximum_accuracy_model[
                    "epoch"
                ]
            )


            max_val_accuracy = float(
                maximum_accuracy_model[
                    "validation_accuracy"
                ]
            )


            val_loss_at_max_accuracy = float(
                maximum_accuracy_model[
                    "validation_loss"
                ]
            )


            test_loss_ma = float(
                maximum_accuracy_model[
                    "test_loss"
                ]
            )


            test_accuracy_ma = float(
                maximum_accuracy_model[
                    "test_accuracy"
                ]
            )


            all_labels_ma = np.asarray(
                maximum_accuracy_model[
                    "all_labels"
                ],
                dtype=np.int64
            )


            all_predictions_ma = np.asarray(
                maximum_accuracy_model[
                    "all_predictions"
                ],
                dtype=np.int64
            )


            # ==================================================
            # NÚMERO DE AMOSTRAS
            #
            # Agora usamos diretamente as informações salvas
            # no checkpoint.
            #
            # Não é necessário abrir novamente fold_X.npz.
            # ==================================================

            n_training_samples.append(
                checkpoint[
                    "total_training_samples"
                ]
            )


            n_validation_samples.append(
                checkpoint[
                    "total_validation_samples"
                ]
            )


            n_test_samples.append(
                checkpoint[
                    "total_test_samples"
                ]
            )


            n_total_samples.append(
                checkpoint[
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


            # --------------------------------------------------
            # Época de menor validation loss
            # --------------------------------------------------

            ax_loss.axvline(
                x=min_val_loss_epoch + 1,
                color="forestgreen",
                linestyle=":",
                linewidth=2,
                label="Min Val Loss"
            )


            # --------------------------------------------------
            # Época de maior validation accuracy
            # --------------------------------------------------

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
            # GRÁFICO DE ACURÁCIA
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


            # --------------------------------------------------
            # Época de menor loss
            # --------------------------------------------------

            ax_acc.axvline(
                x=min_val_loss_epoch + 1,
                color="forestgreen",
                linestyle=":",
                linewidth=2,
                label="Min Val Loss"
            )


            # --------------------------------------------------
            # Época de maior accuracy
            # --------------------------------------------------

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


            # --------------------------------------------------
            # OBSERVAÇÃO
            #
            # Estes valores agora são lidos diretamente do
            # checkpoint do melhor modelo.
            #
            # Não precisamos recalculá-los usando os arrays.
            # --------------------------------------------------

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


            # --------------------------------------------------
            # Aumentar cabeçalho
            # --------------------------------------------------

            for col_idx in range(
                2
            ):

                mini_table[
                    0,
                    col_idx
                ].set_height(
                    0.18
                )


            # --------------------------------------------------
            # Nos folds sem rowLabels não precisamos reservar
            # o mesmo espaço lateral.
            # --------------------------------------------------

            if fold_idx > 0:

                mini_table.scale(
                    1.0,
                    1.0
                )


        # ==================================================
        # TRANSFORMAR NÚMERO DE AMOSTRAS EM ARRAYS
        # ==================================================

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


        # ==================================================
        # TABELA FINAL DE RESULTADOS DE TESTE
        #
        # LINHA 2 / ÚLTIMA COLUNA
        #
        # Agora os resultados vêm do *_summary.pth
        # ==================================================

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


        # --------------------------------------------------
        # Média
        # --------------------------------------------------

        fold_accuracies_mean_ml = (
            fold_accuracies_ml.mean()
        )


        fold_accuracies_mean_ma = (
            fold_accuracies_ma.mean()
        )


        # --------------------------------------------------
        # Desvio padrão
        # --------------------------------------------------

        fold_accuracies_std_ml = (
            fold_accuracies_ml.std()
        )


        fold_accuracies_std_ma = (
            fold_accuracies_ma.std()
        )


        # --------------------------------------------------
        # Adicionar média e desvio padrão
        # --------------------------------------------------

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


        # --------------------------------------------------
        # AUMENTAR CABEÇALHO
        # --------------------------------------------------

        for col_idx in range(
            2
        ):

            mini_table[
                0,
                col_idx
            ].set_height(
                0.15
            )


        # --------------------------------------------------
        # DESTACAR MEAN E STD
        #
        # No matplotlib:
        #
        # linha 0 = cabeçalho
        # linhas 1..n = dados
        #
        # Portanto:
        #
        # Mean = num_rows - 1
        # Std  = num_rows
        # --------------------------------------------------

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


        # --------------------------------------------------
        # DESTACAR ROW LABELS
        # --------------------------------------------------

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


        # ==================================================
        # TABELA DE NÚMERO DE AMOSTRAS
        #
        # LINHA 1 / ÚLTIMA COLUNA
        # ==================================================

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


        # --------------------------------------------------
        # Aumentar cabeçalho
        # --------------------------------------------------

        for col_idx in range(
            4
        ):

            mini_table[
                0,
                col_idx
            ].set_height(
                0.15
            )


        # ==================================================
        # TABELA EXTRA
        #
        # LINHA 0 / ÚLTIMA COLUNA
        #
        # Mantida vazia como no código original.
        # ==================================================

        ax_table = axs[
            0,
            n_folds
        ]


        ax_table.axis(
            "off"
        )


        # ==================================================
        # AJUSTAR LAYOUT
        # ==================================================

        plt.tight_layout()


        # ==================================================
        # DIRETÓRIO DA FIGURA
        # ==================================================

        SAVE_FIG = (
            OUTPUT_DIR
            / f"{branch}_branch"
            / subject
        )


        SAVE_FIG.mkdir(
            parents=True,
            exist_ok=True
        )


        # ==================================================
        # SALVAR FIGURA
        # ==================================================

        figure_path = (
            SAVE_FIG
            / f"{branch}_{subject}.svg"
        )


        plt.savefig(
            figure_path,
            bbox_inches="tight"
        )


        print(
            f"Figure saved: "
            f"{figure_path}"
        )


        # ==================================================
        # FECHAR FIGURA
        #
        # Importante quando vários sujeitos são processados.
        # Evita manter figuras anteriores na memória.
        # ==================================================


        plt.show()
        
        plt.close(
            fig
        )


# ==========================================================
# EXECUTAR
# ==========================================================

if __name__ == "__main__":

    individual_subject_plots(
        branch="space1_space2_frequency_time",
        subjects=[
            "sub-01"
        ]
    )