import numpy as np
from pathlib import Path
import pickle

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split
)


# ==========================================================
# CONFIGURATION
# ==========================================================

N_FOLDS = 6
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42


current_file = Path(__file__).resolve()

PROJECT_ROOT = current_file.parents[2]


INPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_30_psd_array_assembly"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_40_obtain_indices"
)


subjects = [
    f"sub-{i:02d}"
    for i in range(1, 11)
]


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def obtain_indices(
    subjects=subjects,
    N_FOLDS=N_FOLDS,
    VALIDATION_SIZE=VALIDATION_SIZE,
    RANDOM_STATE=RANDOM_STATE,
):

    # ======================================================
    # SUBJECT LOOP
    # ======================================================

    for subject in subjects:

        print()
        print("#" * 70)
        print(f"SUBJECT: {subject}")
        print("#" * 70)


        # ==================================================
        # LOAD DATA
        # ==================================================

        file_path = (
            INPUT_DIR
            / f"{subject}_all_sessions.pkl"
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"\nFile not found:\n"
                f"{file_path}"
            )


        with open(file_path, "rb") as file:

            subj_file = pickle.load(file)


        psd_array = subj_file["psd_array"]

        labels = np.asarray(
            subj_file["labels"]
        ).reshape(-1)


        print(
            "PSD array shape:",
            psd_array.shape
        )

        print(
            "Labels shape:",
            labels.shape
        )


        # ==================================================
        # CHECK NUMBER OF SAMPLES
        # ==================================================

        if len(psd_array) != len(labels):

            raise ValueError(
                f"Number of PSD samples ({len(psd_array)}) "
                f"does not match number of labels "
                f"({len(labels)})."
            )


        # ==================================================
        # DISPLAY DATA STATISTICS
        # ==================================================

        print(
            "Min:",
            psd_array.min()
        )

        print(
            "Max:",
            psd_array.max()
        )

        print(
            "Mean:",
            psd_array.mean()
        )

        print(
            "Std:",
            psd_array.std()
        )

        nonzero_values = np.count_nonzero(
            psd_array
        )

        print(
            "Non-zero values:",
            nonzero_values
        )

        print(
            "Non-zero proportion:",
            nonzero_values
            / psd_array.size
        )


        # ==================================================
        # FOLD DIRECTORY
        # ==================================================

        FOLDS_DIR = (
            OUTPUT_DIR
            / "fold_indices"
            / subject
        )


        FOLDS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        # ==================================================
        # CREATE OUTER STRATIFIED FOLDS
        # ==================================================

        skf = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE
        )


        # ==================================================
        # FOLD LOOP
        # ==================================================

        for fold, (
            development_indices,
            test_indices
        ) in enumerate(
            skf.split(
                psd_array,
                labels
            ),
            start=1
        ):

            print()
            print("=" * 70)
            print(
                f"FOLD {fold}/{N_FOLDS}"
            )
            print("=" * 70)


            # ==============================================
            # SPLIT DEVELOPMENT SET INTO
            # TRAINING AND VALIDATION SETS
            # ==============================================

            train_indices, val_indices = train_test_split(
                development_indices,
                test_size=VALIDATION_SIZE,
                stratify=labels[
                    development_indices
                ],
                random_state=RANDOM_STATE
            )


            # ==============================================
            # CHECK FOR DATA LEAKAGE
            # ==============================================

            assert len(
                np.intersect1d(
                    train_indices,
                    val_indices
                )
            ) == 0


            assert len(
                np.intersect1d(
                    train_indices,
                    test_indices
                )
            ) == 0


            assert len(
                np.intersect1d(
                    val_indices,
                    test_indices
                )
            ) == 0


            all_indices = np.concatenate(
                [
                    train_indices,
                    val_indices,
                    test_indices
                ]
            )


            assert (
                len(
                    np.unique(
                        all_indices
                    )
                )
                == len(labels)
            )


            # ==============================================
            # DISPLAY SET SIZES
            # ==============================================

            total_samples = len(labels)


            print(
                "Total:",
                total_samples
            )

            print(
                "Training:",
                len(train_indices)
            )

            print(
                "Validation:",
                len(val_indices)
            )

            print(
                "Test:",
                len(test_indices)
            )


            print(
                "Training (%):",
                len(train_indices)
                / total_samples
            )

            print(
                "Validation (%):",
                len(val_indices)
                / total_samples
            )

            print(
                "Test (%):",
                len(test_indices)
                / total_samples
            )


            # ==============================================
            # SAVE INDICES
            # ==============================================

            # All indices are relative to the original
            # PSD array and labels array.

            np.savez(
                FOLDS_DIR
                / f"fold_{fold}.npz",

                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices
            )


            print(
                f"Indices saved: fold_{fold}.npz"
            )


# ==========================================================
# EXECUTE
# ==========================================================

if __name__ == "__main__":

    obtain_indices()