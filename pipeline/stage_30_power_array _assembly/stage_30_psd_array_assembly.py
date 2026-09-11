
from pathlib import Path
import sys
import numpy as np
import os

import pickle
from scipy.signal import spectrogram



# ==========================================================
# PATH CONFIGURATION & LOCAL IMPORTS
# ==========================================================

current_file = Path(__file__).resolve()

PROJECT_ROOT = current_file.parents[2]

PIPELINE_ROOT = current_file.parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.append(
        str(PROJECT_ROOT)
    )


if str(PIPELINE_ROOT) not in sys.path:

    sys.path.append(
        str(PIPELINE_ROOT)
    )

INPUT_DIR = (
    PROJECT_ROOT 
    / "processed_data" 
    / "stage_20_load_and_epoching"
)

OUTPUT_DIR = (
   PROJECT_ROOT 
   / "processed_data" 
   / "stage_30_psd_array_assembly"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

from channels_2D_distribution import ORIGINAL_2D_DISTRIBUTION

from config.channel_mapping import CHANNELS_INDICES_MAPPING


subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]


def obtain_spectrogram(data, sampling_rate):


    # Janela de 0,5 segundo.
    WINDOW_DURATION = 0.5

    # Deslocamento de 0,25 segundo.
    STEP_DURATION = 0.25

    nperseg = int(WINDOW_DURATION * sampling_rate)
    step_samples = int(STEP_DURATION * sampling_rate)
    noverlap = nperseg - step_samples

    frequencies, times, spectrogram_data = spectrogram(
        data,
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend="constant",
        scaling="density",
        mode="psd",
        axis=-1
    )

    print("  Frequencies:", frequencies.shape)
    print("  Times:", times.shape)
    print("  Espectrograms:", spectrogram_data.shape)

    return frequencies, times, spectrogram_data


def create_power_array(data, Channels_Distribution, sampling_rate, session_data):

    s2_quantity = 21
    s1_quantity = 21

    #spectrogram_data.shape: [epoch, channel, frequency, time_window]
    frequencies, times, spectrogram_data = obtain_spectrogram(data, sampling_rate)
                                
    # psd_array.shape:
    # [epoch, s2, s1, frequency, time_window]
    psd_array = np.zeros([data.shape[0], s2_quantity, s1_quantity, frequencies.shape[0], times.shape[0]])

    used_positions = set()

    for channel, channel_index in CHANNELS_INDICES_MAPPING.items():

        # =============================================================
        # Checking if CHANNELS_INDICES_MAPPING corresponds to 
        # session_data["channel_names"][channel_index]
        # =============================================================
        actual_channel = session_data["channel_names"][channel_index]

        if actual_channel != channel:
            raise ValueError(
                f"Channel mapping mismatch: "
                f"index {channel_index} should contain {channel}, "
                f"but contains {actual_channel}."
            )

       
        s2 = Channels_Distribution.CHANNELS_POSITIONS[channel]['s2']
        s1 = Channels_Distribution.CHANNELS_POSITIONS[channel]['s1']

        # =============================================================
        # Checking
        # 0 <= s1 < 21
        # 0 <= s2 < 21
        # no electrode occupies the same position as another electrode
        # =============================================================

        if not (0 <= s1 < 21 and 0 <= s2 < 21):
            raise ValueError(
                f"Invalid position for {channel}: "
                f"s1={s1}, s2={s2}"
            )

        position = (s2, s1)

        if position in used_positions:
            raise ValueError(
                f"Duplicated spatial position: "
                f"{position} for channel {channel}"
            )

        used_positions.add(position)

        # =============================================================
        # Building PSD Array
        # =============================================================

        psd_array[ : , s2, s1, :, : ] = spectrogram_data[:, channel_index, :, : ]

    return frequencies, times, psd_array

def power_array_assembly(
        Channels_Distribution = ORIGINAL_2D_DISTRIBUTION
):

    """Build the PSD array for all sessions."""




    # ======================================================
    # CREATING FIGURE
    # ======================================================
    Channels_Distribution.plot_channels_distribution(OUTPUT_DIR)

    # ======================================================
    # FINDING SESSIONS FILES
    # ======================================================

    session_files = sorted(
        INPUT_DIR.glob(
            "sub-*ses-*.pkl"
        )
    )

    if len(session_files) == 0:

        raise FileNotFoundError(
            f"\nNo sub-*/ses-*/*.pkl files found in:\n"
            f"{INPUT_DIR}"
        )

    print(
        f"\nSession files found: "
        f"{len(session_files)}"
    )

    for path in session_files:

        print(
            " -",
            path.relative_to(PROJECT_ROOT)
        )

    # ======================================================
    # CREATING POWER ARRAY
    # ======================================================

    for data_path in session_files:

        session_file = data_path.name
        session_name = data_path.stem


        session_dir = INPUT_DIR/session_file

        with open(session_dir, 'rb') as file:
            session_data = pickle.load(file)

        data = session_data["data"]
        labels = session_data["labels"]

        frequencies, times, psd_array = create_power_array(
            data,
            Channels_Distribution,
            session_data["sampling_rate"],
            session_data
        )

        if len(psd_array) != len(labels):
            raise ValueError(f"Incompatible sessions and labels in {session_file}")


        processed_session = {

            "sub_ses": session_data["sub_ses"],

            "psd_array": psd_array,

            "frequencies": frequencies,

            "times": times,

            "labels": labels,

            "channel_names": session_data["channel_names"],

            "sampling_rate": session_data["sampling_rate"],

            "condition": session_data["condition"],

            "start_seconds": session_data["start_seconds"],

            "end_seconds": session_data["end_seconds"],

        }


        # ==================================================
        # SAVING PROCESSED SESSION
        # ==================================================

        output_path = (
            OUTPUT_DIR
            / f"{session_name}.pkl"
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
            f"\n{session_name}"
        )

        print(
            f"  data shape: {psd_array.shape}"
        )

        print(
            f"  labels shape: {labels.shape}"
        )

        print(
            f"  saved to: {output_path.relative_to(PROJECT_ROOT)}"
        )


    # ===========================================================
    # CONCATENATING THE ARRAYS
    # ===========================================================

    for subject in subjects:

        print(f"\n{subject}")

        psd_sessions = []
        label_sessions = []
        session_ids = []

        for session in sessions:

            file_path = os.path.join(
                OUTPUT_DIR,
                f"{subject}_{session}.pkl"
            )

            if not file_path.exists():
                raise FileNotFoundError(
                    f"\nFile not found:\n"
                    f"{file_path}"
                )
    
            with open(file_path, 'rb') as file:
                session_data = pickle.load(file)

            psd_array = session_data["psd_array"]
            labels = session_data["labels"]

            psd_sessions.append(psd_array)
            label_sessions.append(labels)

            session_ids.append(
                np.full(
                    len(labels),
                    session
                )
            )
            
            print(f"Shape of section data: {psd_array.shape}  \nShape of section labels: {labels.shape}")

            



    
        # =============================================
        # CONCATENATING THE SESSIONS FOR EACH SUBJECT
        # =============================================

        if len(psd_sessions) != len(sessions):
            raise ValueError(f"Required {len(sessions)} sessions. But just found {len(psd_sessions)} for subject{subject}")

        psd_array_all_sessions = np.concatenate(psd_sessions, axis=0)
        labels_all_sessions = np.concatenate(label_sessions, axis=0)
        
        session_ids_all_sessions = np.concatenate(
            session_ids,
            axis=0
        )

        if not (
            psd_array_all_sessions.shape[0]
            == labels_all_sessions.shape[0]
            == session_ids_all_sessions.shape[0]
        ):
            raise ValueError(
                f"{subject}: number of epochs, labels, and session IDs differ."
            )
        

        processed_subject = {

            "subject": subject,

            "psd_array": psd_array_all_sessions,

            "frequencies": frequencies,

            "times": times,

            "labels": labels_all_sessions,

            "session_ids": session_ids_all_sessions,

            "channel_names": session_data["channel_names"],

            "sampling_rate": session_data["sampling_rate"],

            "condition": session_data["condition"],

            "start_seconds": session_data["start_seconds"],

            "end_seconds": session_data["end_seconds"],

        }

        # ==================================================
        # SAVING PROCESSED SUBJECT
        # ==================================================

        output_path = (
            OUTPUT_DIR
            / f"{subject}_all_sessions.pkl"
        )

        with output_path.open("wb") as file:

            pickle.dump(
                processed_subject,
                file
            )

        # ==================================================
        # SUBJECT INFORMATION
        # ==================================================

        print(
            f"\n{subject}"
        )

        print(
            f"  data shape: {psd_array_all_sessions.shape}"
        )

        print(
            f"  labels shape: {labels_all_sessions.shape}"
        )

        print(
            f"  saved to: {output_path.relative_to(PROJECT_ROOT)}"
        )

    print(
        "\n"
        "===============================================\n"
        "STAGE 30 COMPLETED\n"
        "==============================================="
    )
        

if __name__ == "__main__":

    power_array_assembly()