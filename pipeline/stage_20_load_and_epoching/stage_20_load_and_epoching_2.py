from pathlib import Path
import sys
import pickle

import mne


# ==========================================================
# CONFIGURAÇÕES DE CAMINHO & IMPORTS LOCAIS
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
# CONFIGURAÇÕES
# ==========================================================

EPOCH_START_SECONDS = 1.0
EPOCH_END_SECONDS = 3.5


# ==========================================================
# CONDIÇÕES
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
        CONDITION,
        EPOCH_START_SECONDS=EPOCH_START_SECONDS,
        EPOCH_END_SECONDS=EPOCH_END_SECONDS):

    """
    Carrega os arquivos EEG epocados, seleciona uma condição
    experimental e recorta temporalmente cada época.

    Parameters
    ----------
    CONDITION : str
        Condição experimental.

        Opções:
            "PRONOUNCED_SPEECH"
            "INNER_SPEECH"
            "VISUALIZED_CONDITION"

    EPOCH_START_SECONDS : float
        Tempo inicial da janela utilizada.

    EPOCH_END_SECONDS : float
        Tempo final da janela utilizada.
    """

    # ======================================================
    # VERIFICAÇÃO DA CONDIÇÃO
    # ======================================================

    if CONDITION not in CONDITION_MAP:

        raise ValueError(
            f"\nCondição inválida: {CONDITION}\n"
            f"Condições disponíveis: "
            f"{list(CONDITION_MAP.keys())}"
        )

    # ======================================================
    # VERIFICAÇÃO DA JANELA TEMPORAL
    # ======================================================

    if EPOCH_START_SECONDS >= EPOCH_END_SECONDS:

        raise ValueError(
            "\nEPOCH_START_SECONDS deve ser menor que "
            "EPOCH_END_SECONDS."
        )

    # ======================================================
    # ENCONTRANDO OS ARQUIVOS DE EEG EPOCADO
    # ======================================================

    epoch_files = sorted(
        INPUT_DIR.glob(
            "sub-*/ses-*/*_eeg-epo.fif"
        )
    )

    if len(epoch_files) == 0:

        raise FileNotFoundError(
            f"\nNenhum arquivo *_eeg-epo.fif encontrado em:\n"
            f"{INPUT_DIR}"
        )

    print(
        f"\nArquivos das sessões encontrados: "
        f"{len(epoch_files)}"
    )

    for path in epoch_files:

        print(
            " -",
            path.relative_to(PROJECT_ROOT)
        )

    # ======================================================
    # INFORMAÇÕES DO PROCESSAMENTO
    # ======================================================

    print(
        "\n"
        "===============================================\n"
        "PROCESSING EPOCHS\n"
        "===============================================\n"
        f"CONDITION: {CONDITION}\n"
        f"START TIME: {EPOCH_START_SECONDS} s\n"
        f"END TIME: {EPOCH_END_SECONDS} s\n"
    )

    # ======================================================
    # PROCESSANDO CADA SESSÃO
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
            CONDITION=CONDITION,
            EPOCH_START_SECONDS=EPOCH_START_SECONDS,
            EPOCH_END_SECONDS=EPOCH_END_SECONDS

        )

        # ==================================================
        # ESTRUTURA SALVA
        # ==================================================

        processed_session = {

            "sub_ses": sub_ses,

            "data": data,

            "labels": labels,

            "channel_names": channel_names,

            "sampling_rate": sampling_rate,

            "condition": CONDITION,

            "start_seconds": EPOCH_START_SECONDS,

            "end_seconds": EPOCH_END_SECONDS,

        }

        # ==================================================
        # SALVANDO
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
        # INFORMAÇÕES DA SESSÃO
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
            f"  saved: {output_path.relative_to(PROJECT_ROOT)}"
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
        CONDITION,
        EPOCH_START_SECONDS,
        EPOCH_END_SECONDS):

    """
    Processa uma sessão EEG epocada.

    Retorna apenas as épocas pertencentes à condição desejada
    e dentro da janela temporal especificada.
    """

    # ======================================================
    # NOME DA SESSÃO
    # ======================================================

    session_name = (
        epochs_path.name
        .replace("_eeg-epo.fif", "")
    )

    # ======================================================
    # ARQUIVO DE EVENTOS
    # ======================================================

    events_path = epochs_path.with_name(
        f"{session_name}_events.dat"
    )

    if not events_path.exists():

        raise FileNotFoundError(
            f"\nArquivo de eventos não encontrado:\n"
            f"{events_path}"
        )

    # ======================================================
    # CARREGANDO EEG EPOCADO
    # ======================================================

    epochs = mne.read_epochs(
        epochs_path,
        preload=True,
        verbose=False
    )

    # ======================================================
    # CARREGANDO EVENTOS
    # ======================================================

    with events_path.open("rb") as file:

        events = pickle.load(file)

    # ======================================================
    # TABELA DE EVENTOS
    # ======================================================
    #
    # Cada arquivo events.dat contém uma matriz de quatro
    # colunas, onde cada linha corresponde a uma tentativa.
    #
    # Coluna 0:
    #     número da amostra em que o evento ocorreu.
    #
    # Coluna 1:
    #     classe / direção da palavra.
    #
    #     0 = Up / Arriba
    #     1 = Down / Abajo
    #     2 = Right / Derecha
    #     3 = Left / Izquierda
    #
    # Coluna 2:
    #     condição experimental.
    #
    #     0 = Pronounced Speech
    #     1 = Inner Speech
    #     2 = Visualized Condition
    #
    # Coluna 3:
    #     número da sessão.
    #
    #     1 = Sessão 1
    #     2 = Sessão 2
    #     3 = Sessão 3
    #
    # ======================================================

    # ======================================================
    # VERIFICAÇÃO DO FORMATO DOS EVENTOS
    # ======================================================

    if events.ndim != 2:

        raise ValueError(
            f"\n{session_name}: events deveria possuir "
            f"duas dimensões.\n"
            f"Shape encontrado: {events.shape}"
        )

    if events.shape[1] < 4:

        raise ValueError(
            f"\n{session_name}: events deveria possuir "
            f"pelo menos 4 colunas.\n"
            f"Shape encontrado: {events.shape}"
        )

    # ======================================================
    # VERIFICAÇÃO EVENTS ↔ EPOCHS
    # ======================================================

    if len(events) != len(epochs):

        raise ValueError(
            f"\n{session_name}: número de eventos diferente "
            f"do número de épocas.\n"
            f"Events: {len(events)}\n"
            f"Epochs: {len(epochs)}\n"
            "\nIsso pode indicar perda do alinhamento entre "
            "os labels e as épocas."
        )

    # ======================================================
    # VERIFICAÇÃO DA JANELA TEMPORAL
    # ======================================================

    if EPOCH_START_SECONDS < epochs.tmin:

        raise ValueError(
            f"\n{session_name}: EPOCH_START_SECONDS "
            f"({EPOCH_START_SECONDS} s) está antes do início "
            f"das épocas ({epochs.tmin} s)."
        )

    if EPOCH_END_SECONDS > epochs.tmax:

        raise ValueError(
            f"\n{session_name}: EPOCH_END_SECONDS "
            f"({EPOCH_END_SECONDS} s) está depois do final "
            f"das épocas ({epochs.tmax} s)."
        )

    # ======================================================
    # SELEÇÃO DA CONDIÇÃO
    # ======================================================

    condition_id = CONDITION_MAP[
        CONDITION
    ]

    condition_mask = (
        events[:, 2]
        == condition_id
    )

    # ======================================================
    # VERIFICANDO SE EXISTEM TRIALS DA CONDIÇÃO
    # ======================================================

    number_of_trials = condition_mask.sum()

    if number_of_trials == 0:

        raise ValueError(
            f"\n{session_name}: nenhuma época encontrada "
            f"para a condição {CONDITION}."
        )

    # ======================================================
    # SELECIONANDO AS ÉPOCAS
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
    # CONVERSÃO TEMPO → ÍNDICES
    # ======================================================
    #
    # É preferível utilizar time_as_index() ao invés de:
    #
    #     round(time * sampling_rate)
    #
    # porque o MNE considera corretamente o tempo inicial
    # da época (epochs.tmin).
    #
    # Exemplo:
    #
    # fs = 256 Hz
    # janela = 1.0 até 3.5 s
    #
    # duração = 2.5 s
    #
    # número de amostras:
    #
    #     2.5 * 256 = 640
    #
    # O slicing abaixo utiliza:
    #
    #     [start_sample:end_sample]
    #
    # portanto end_sample não é incluído.
    #
    # ======================================================

    start_sample, end_sample = (
        condition_epochs.time_as_index(
            [
                EPOCH_START_SECONDS,
                EPOCH_END_SECONDS
            ]
        )
    )

    # ======================================================
    # EXTRAINDO OS DADOS
    # ======================================================

    data = condition_epochs.get_data()

    data = data[
        :,
        :,
        start_sample:end_sample
    ]

    # ======================================================
    # NOMES DOS CANAIS
    # ======================================================

    channel_names = (
        condition_epochs.ch_names
    )

    # ======================================================
    # VERIFICAÇÃO FINAL LABELS ↔ DATA
    # ======================================================

    if data.shape[0] != labels.shape[0]:

        raise RuntimeError(
            f"\n{session_name}: número de trials nos dados "
            f"não corresponde ao número de labels.\n"
            f"Data: {data.shape[0]}\n"
            f"Labels: {labels.shape[0]}"
        )

    # ======================================================
    # INFORMAÇÕES DA JANELA
    # ======================================================

    expected_samples = (
        end_sample
        - start_sample
    )

    if data.shape[2] != expected_samples:

        raise RuntimeError(
            f"\n{session_name}: número inesperado "
            f"de amostras.\n"
            f"Esperado: {expected_samples}\n"
            f"Obtido: {data.shape[2]}"
        )

    # ======================================================
    # RETORNO
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
        CONDITION="INNER_SPEECH"
    )