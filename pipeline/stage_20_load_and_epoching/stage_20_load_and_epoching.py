from pathlib import Path
import sys
import pickle

import mne


# ==========================================================
# PATH CONFIGURATION & LOCAL IMPORTS
# ==========================================================

current_file = Path(__file__).resolve()

PROJECT_ROOT = current_file.parents[2]

PIPELINE_ROOT = current_file.parents[1]

if str(PIPELINE_ROOT) not in sys.path:

    sys.path.append(
        str(PIPELINE_ROOT)
    )


INPUT_DIR = (
    PROJECT_ROOT
    / "thinking_outloud_dataset"
    / "derivatives"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_20_load_and_epoching"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# CONFIGURATION
# ==========================================================

EPOCH_START_SECONDS = 1.0
EPOCH_END_SECONDS = 3.5


# ==========================================================
# CONDITIONS
# ==========================================================

CONDITION_MAP = {

    "PRONOUNCED_SPEECH": 0,
    "INNER_SPEECH": 1,
    "VISUALIZED_CONDITION": 2,

}


# ==========================================================
# LOAD AND EPOCHING
# ==========================================================

def load_and_epoching(
        condition,
        crop_start_time=EPOCH_START_SECONDS,
        crop_end_time=EPOCH_END_SECONDS):

    """
    Load epoched EEG files, select one experimental condition,
    crop each epoch to the specified time interval, and save
    the processed session data.

    Parameters
    ----------
    CONDITION : str
        Experimental condition.

        Options:
            "PRONOUNCED_SPEECH"
            "INNER_SPEECH"
            "VISUALIZED_CONDITION"

    EPOCH_START_SECONDS : float
        Start time of the selected epoch window, in seconds.

    EPOCH_END_SECONDS : float
        End time of the selected epoch window, in seconds.
    """

    # ======================================================
    # CONDITION VALIDATION
    # ======================================================

    if condition not in CONDITION_MAP:

        raise ValueError(
            f"\nInvalid condition: {condition}\n"
            f"Available conditions: "
            f"{list(CONDITION_MAP.keys())}"
        )

    # ======================================================
    # TIME WINDOW VALIDATION
    # ======================================================

    if crop_start_time >= crop_end_time:

        raise ValueError(
            "\nEPOCH_START_SECONDS must be smaller than "
            "EPOCH_END_SECONDS."
        )

    # ======================================================
    # FINDING EPOCHED EEG FILES
    # ======================================================

    epoch_files = sorted(
        INPUT_DIR.glob(
            "sub-*/ses-*/*_eeg-epo.fif"
        )
    )

    if len(epoch_files) == 0:

        raise FileNotFoundError(
            f"\nNo *_eeg-epo.fif files found in:\n"
            f"{INPUT_DIR}"
        )

    print(
        f"\nSession files found: "
        f"{len(epoch_files)}"
    )

    for path in epoch_files:

        print(
            " -",
            path.relative_to(PROJECT_ROOT)
        )

    # ======================================================
    # PROCESSING INFORMATION
    # ======================================================

    print(
        "\n"
        "===============================================\n"
        "PROCESSING EPOCHS\n"
        "===============================================\n"
        f"CONDITION: {condition}\n"
        f"START TIME: {crop_start_time} s\n"
        f"END TIME: {crop_end_time} s\n"
    )

    # ======================================================
    # PROCESSING EACH SESSION
    # ======================================================

    for epochs_path in epoch_files:

        (
            sub_ses,
            data,
            labels,
            channel_names,
            sampling_rate

        ) = process_session(

            epochs_path=epochs_path,
            condition=condition,
            crop_start_time=crop_start_time,
            crop_end_time=crop_end_time

        )

        # ==================================================
        # SAVED DATA STRUCTURE
        # ==================================================

        processed_session = {

            "sub_ses": sub_ses,

            "data": data,

            "labels": labels,

            "channel_names": channel_names,

            "sampling_rate": sampling_rate,

            "condition": condition,

            "start_seconds": crop_start_time,

            "end_seconds": crop_end_time,

        }

        # ==================================================
        # SAVING PROCESSED SESSION
        # ==================================================

        output_path = (
            OUTPUT_DIR
            / f"{sub_ses}.pkl"
        )

        with output_path.open("wb") as file:

            pickle.dump(
                processed_session,
                file
            )

        # ==================================================
        # SESSION INFORMATION
        # ==================================================

        print(
            f"\n{sub_ses}"
        )

        print(
            f"  data shape: {data.shape}"
        )

        print(
            f"  labels shape: {labels.shape}"
        )

        print(
            f"  sampling rate: {sampling_rate} Hz"
        )

        print(
            f"  channels: {len(channel_names)}"
        )

        print(
            f"  saved to: {output_path.relative_to(PROJECT_ROOT)}"
        )

    print(
        "\n"
        "===============================================\n"
        "STAGE 20 COMPLETED\n"
        "==============================================="
    )


# ==========================================================
# PROCESS SESSION
# ==========================================================

def process_session(
        epochs_path,
        condition,
        crop_start_time,
        crop_end_time):

    """
    Process one epoched EEG session.

    Only epochs belonging to the selected experimental
    condition and within the specified time interval
    are returned.
    """

    # ======================================================
    # SESSION NAME
    # ======================================================

    session_name = (
        epochs_path.name
        .replace("_eeg-epo.fif", "")
    )

    # ======================================================
    # EVENTS FILE
    # ======================================================

    events_path = epochs_path.with_name(
        f"{session_name}_events.dat"
    )

    if not events_path.exists():

        raise FileNotFoundError(
            f"\nEvents file not found:\n"
            f"{events_path}"
        )

    # ======================================================
    # LOADING EPOCHED EEG
    # ======================================================

    epochs = mne.read_epochs(
        epochs_path,
        preload=True,
        verbose=False
    )

    # ======================================================
    # LOADING EVENTS
    # ======================================================

    with events_path.open("rb") as file:

        events = pickle.load(file)

    # ======================================================
    # EVENTS TABLE
    # ======================================================
    #
    # Each events.dat file contains a matrix with four
    # columns, where each row corresponds to one trial.
    #
    # Column 0:
    #     Sample number at which the event occurred.
    #
    # Column 1:
    #     Word class / direction.
    #
    #     0 = Up / Arriba
    #     1 = Down / Abajo
    #     2 = Right / Derecha
    #     3 = Left / Izquierda
    #
    # Column 2:
    #     Experimental condition.
    #
    #     0 = Pronounced Speech
    #     1 = Inner Speech
    #     2 = Visualized Condition
    #
    # Column 3:
    #     Session number.
    #
    #     1 = Session 1
    #     2 = Session 2
    #     3 = Session 3
    #
    # ======================================================

    # ======================================================
    # EVENTS FORMAT VALIDATION
    # ======================================================

    if events.ndim != 2:

        raise ValueError(
            f"\n{session_name}: events must have "
            f"two dimensions.\n"
            f"Shape found: {events.shape}"
        )

    if events.shape[1] < 4:

        raise ValueError(
            f"\n{session_name}: events must have "
            f"at least 4 columns.\n"
            f"Shape found: {events.shape}"
        )

    # ======================================================
    # EVENTS ↔ EPOCHS ALIGNMENT CHECK
    # ======================================================

    if len(events) != len(epochs):

        raise ValueError(
            f"\n{session_name}: number of events differs "
            f"from number of epochs.\n"
            f"Events: {len(events)}\n"
            f"Epochs: {len(epochs)}\n"
            "\nThis may indicate a loss of alignment between "
            "the labels and the epochs."
        )

    # ======================================================
    # TIME WINDOW VALIDATION
    # ======================================================

    if crop_start_time < epochs.tmin:

        raise ValueError(
            f"\n{session_name}: EPOCH_START_SECONDS "
            f"({crop_start_time} s) is before the beginning "
            f"of the epochs ({epochs.tmin} s)."
        )

    if crop_end_time > epochs.tmax:

        raise ValueError(
            f"\n{session_name}: EPOCH_END_SECONDS "
            f"({crop_end_time} s) is after the end "
            f"of the epochs ({epochs.tmax} s)."
        )

    # ======================================================
    # CONDITION SELECTION
    # ======================================================

    condition_id = CONDITION_MAP[
        condition
    ]

    condition_mask = (
        events[:, 2]
        == condition_id
    )

    # ======================================================
    # CHECKING WHETHER THE CONDITION HAS TRIALS
    # ======================================================

    number_of_trials = condition_mask.sum()

    if number_of_trials == 0:

        raise ValueError(
            f"\n{session_name}: no epochs found "
            f"for condition {condition}."
        )

    # ======================================================
    # SELECTING EPOCHS
    # ======================================================

    condition_epochs = epochs[
        condition_mask
    ]

    # ======================================================
    # LABELS
    # ======================================================

    labels = events[
        condition_mask,
        1
    ].astype(int)

    # ======================================================
    # SAMPLING RATE
    # ======================================================

    sampling_rate = float(
        condition_epochs.info["sfreq"]
    )

    # ======================================================
    # TIME → SAMPLE INDEX CONVERSION
    # ======================================================
    #
    # time_as_index() is preferred over manually computing:
    #
    #     round(time * sampling_rate)
    #
    # because MNE correctly takes the initial epoch time
    # (epochs.tmin) into account.
    #
    # Example:
    #
    # fs = 256 Hz
    # selected window = 1.0 to 3.5 s
    #
    # duration = 2.5 s
    #
    # number of samples:
    #
    #     2.5 * 256 = 640
    #
    # The slicing below uses:
    #
    #     [start_sample:end_sample]
    #
    # therefore end_sample is not included.
    #
    # ======================================================

    start_sample, end_sample = (
        condition_epochs.time_as_index(
            [
                crop_start_time,
                crop_end_time
            ]
        )
    )

    # ======================================================
    # EXTRACTING DATA
    # ======================================================

    data = condition_epochs.get_data()

    data = data[
        :,
        :,
        start_sample:end_sample
    ]

    # ======================================================
    # CHANNEL NAMES
    # ======================================================

    channel_names = (
        condition_epochs.ch_names
    )

    # ======================================================
    # FINAL LABELS ↔ DATA CHECK
    # ======================================================

    if data.shape[0] != labels.shape[0]:

        raise RuntimeError(
            f"\n{session_name}: number of trials in data "
            f"does not match number of labels.\n"
            f"Data: {data.shape[0]}\n"
            f"Labels: {labels.shape[0]}"
        )

    # ======================================================
    # FINAL SAMPLE COUNT CHECK
    # ======================================================

    expected_samples = (
        end_sample
        - start_sample
    )

    if data.shape[2] != expected_samples:

        raise RuntimeError(
            f"\n{session_name}: unexpected number "
            f"of samples.\n"
            f"Expected: {expected_samples}\n"
            f"Obtained: {data.shape[2]}"
        )

    # ======================================================
    # RETURN
    # ======================================================

    return (
        session_name,
        data,
        labels,
        channel_names,
        sampling_rate,
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    # ======================================================
    # CONDITION OPTIONS
    # ======================================================
    #
    # "PRONOUNCED_SPEECH"
    #
    # "INNER_SPEECH"
    #
    # "VISUALIZED_CONDITION"
    #
    # ======================================================

    load_and_epoching(
        condition="PRONOUNCED_SPEECH"
    )