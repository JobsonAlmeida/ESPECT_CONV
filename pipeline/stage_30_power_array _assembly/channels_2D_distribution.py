import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


class ORIGINAL_2D_DISTRIBUTION:

  CHANNELS_POSITIONS = {

  "A1": {
    "s1": 10,
    "s2": 10
  },
  "A2": {
    "s1": 10,
    "s2": 11
  },
  "A3": {
    "s1": 10,
    "s2": 12
  },
  "A4": {
    "s1": 10,
    "s2": 13
  },
  "A5": {
    "s1": 8,
    "s2": 14
  },
  "A6": {
    "s1": 6,
    "s2": 14
  },
  "A7": {
    "s1": 5,
    "s2": 15
  },
  "A8": {
    "s1": 6,
    "s2": 16
  },
  "A9": {
    "s1": 5,
    "s2": 17
  },
  "A10": {
    "s1": 4,
    "s2": 18
  },
  "A11": {
    "s1": 3,
    "s2": 19
  },
  "A12": {
    "s1": 2,
    "s2": 20
  },
  "A13": {
    "s1": 6,
    "s2": 20
  },
  "A14": {
    "s1": 7,
    "s2": 19
  },
  "A15": {
    "s1": 7,
    "s2": 18
  },
  "A16": {
    "s1": 8,
    "s2": 17
  },
  "A17": {
    "s1": 8,
    "s2": 16
  },
  "A18": {
    "s1": 8,
    "s2": 15
  },
  "A19": {
    "s1": 10,
    "s2": 14
  },
  "A20": {
    "s1": 10,
    "s2": 15
  },
  "A21": {
    "s1": 10,
    "s2": 16
  },
  "A22": {
    "s1": 10,
    "s2": 17
  },
  "A23": {
    "s1": 10,
    "s2": 18
  },
  "A24": {
    "s1": 10,
    "s2": 19
  },
  "A25": {
    "s1": 10,
    "s2": 20
  },
  "A26": {
    "s1": 14,
    "s2": 20
  },
  "A27": {
    "s1": 13,
    "s2": 19
  },
  "A28": {
    "s1": 13,
    "s2": 18
  },
  "A29": {
    "s1": 12,
    "s2": 17
  },
  "A30": {
    "s1": 12,
    "s2": 16
  },
  "A31": {
    "s1": 12,
    "s2": 15
  },
  "A32": {
    "s1": 12,
    "s2": 14
  },
  "B1": {
    "s1": 11,
    "s2": 11
  },
  "B2": {
    "s1": 12,
    "s2": 12
  },
  "B3": {
    "s1": 14,
    "s2": 14
  },
  "B4": {
    "s1": 15,
    "s2": 15
  },
  "B5": {
    "s1": 14,
    "s2": 16
  },
  "B6": {
    "s1": 15,
    "s2": 17
  },
  "B7": {
    "s1": 16,
    "s2": 18
  },
  "B8": {
    "s1": 17,
    "s2": 19
  },
  "B9": {
    "s1": 18,
    "s2": 20
  },
  "B10": {
    "s1": 19,
    "s2": 17
  },
  "B11": {
    "s1": 18,
    "s2": 16
  },
  "B12": {
    "s1": 17,
    "s2": 15
  },
  "B13": {
    "s1": 16,
    "s2": 14
  },
  "B14": {
    "s1": 18,
    "s2": 13
  },
  "B15": {
    "s1": 17,
    "s2": 12
  },
  "B16": {
    "s1": 16,
    "s2": 12
  },
  "B17": {
    "s1": 15,
    "s2": 12
  },
  "B18": {
    "s1": 14,
    "s2": 12
  },
  "B19": {
    "s1": 13,
    "s2": 12
  },
  "B20": {
    "s1": 12,
    "s2": 10
  },
  "B21": {
    "s1": 13,
    "s2": 10
  },
  "B22": {
    "s1": 14,
    "s2": 10
  },
  "B23": {
    "s1": 15,
    "s2": 10
  },
  "B24": {
    "s1": 16,
    "s2": 10
  },
  "B25": {
    "s1": 17,
    "s2": 10
  },
  "B26": {
    "s1": 18,
    "s2": 10
  },
  "B27": {
    "s1": 18,
    "s2": 7
  },
  "B28": {
    "s1": 17,
    "s2": 8
  },
  "B29": {
    "s1": 16,
    "s2": 8
  },
  "B30": {
    "s1": 15,
    "s2": 8
  },
  "B31": {
    "s1": 14,
    "s2": 8
  },
  "B32": {
    "s1": 13,
    "s2": 8
  },
  "C1": {
    "s1": 11,
    "s2": 9
  },
  "C2": {
    "s1": 12,
    "s2": 8
  },
  "C3": {
    "s1": 14,
    "s2": 6
  },
  "C4": {
    "s1": 15,
    "s2": 5
  },
  "C5": {
    "s1": 16,
    "s2": 6
  },
  "C6": {
    "s1": 17,
    "s2": 5
  },
  "C7": {
    "s1": 18,
    "s2": 4
  },
  "C8": {
    "s1": 16,
    "s2": 2
  },
  "C9": {
    "s1": 15,
    "s2": 3
  },
  "C10": {
    "s1": 14,
    "s2": 4
  },
  "C11": {
    "s1": 12,
    "s2": 7
  },
  "C12": {
    "s1": 12,
    "s2": 6
  },
  "C13": {
    "s1": 12,
    "s2": 5
  },
  "C14": {
    "s1": 12,
    "s2": 4
  },
  "C15": {
    "s1": 12,
    "s2": 3
  },
  "C16": {
    "s1": 13,
    "s2": 2
  },
  "C17": {
    "s1": 10,
    "s2": 2
  },
  "C18": {
    "s1": 10,
    "s2": 3
  },
  "C19": {
    "s1": 10,
    "s2": 4
  },
  "C20": {
    "s1": 10,
    "s2": 5
  },
  "C21": {
    "s1": 10,
    "s2": 6
  },
  "C22": {
    "s1": 10,
    "s2": 7
  },
  "C23": {
    "s1": 10,
    "s2": 8
  },
  "C24": {
    "s1": 8,
    "s2": 7
  },
  "C25": {
    "s1": 8,
    "s2": 6
  },
  "C26": {
    "s1": 8,
    "s2": 5
  },
  "C27": {
    "s1": 8,
    "s2": 4
  },
  "C28": {
    "s1": 8,
    "s2": 3
  },
  "C29": {
    "s1": 7,
    "s2": 2
  },
  "C30": {
    "s1": 4,
    "s2": 2
  },
  "C31": {
    "s1": 5,
    "s2": 3
  },
  "C32": {
    "s1": 6,
    "s2": 4
  },
  "D1": {
    "s1": 9,
    "s2": 9
  },
  "D2": {
    "s1": 8,
    "s2": 8
  },
  "D3": {
    "s1": 6,
    "s2": 6
  },
  "D4": {
    "s1": 5,
    "s2": 5
  },
  "D5": {
    "s1": 4,
    "s2": 6
  },
  "D6": {
    "s1": 3,
    "s2": 5
  },
  "D7": {
    "s1": 2,
    "s2": 4
  },
  "D8": {
    "s1": 2,
    "s2": 7
  },
  "D9": {
    "s1": 3,
    "s2": 8
  },
  "D10": {
    "s1": 4,
    "s2": 8
  },
  "D11": {
    "s1": 5,
    "s2": 8
  },
  "D12": {
    "s1": 6,
    "s2": 8
  },
  "D13": {
    "s1": 7,
    "s2": 8
  },
  "D14": {
    "s1": 8,
    "s2": 10
  },
  "D15": {
    "s1": 9,
    "s2": 11
  },
  "D16": {
    "s1": 8,
    "s2": 12
  },
  "D17": {
    "s1": 7,
    "s2": 12
  },
  "D18": {
    "s1": 7,
    "s2": 10
  },
  "D19": {
    "s1": 6,
    "s2": 10
  },
  "D20": {
    "s1": 5,
    "s2": 10
  },
  "D21": {
    "s1": 4,
    "s2": 10
  },
  "D22": {
    "s1": 3,
    "s2": 10
  },
  "D23": {
    "s1": 2,
    "s2": 10
  },
  "D24": {
    "s1": 2,
    "s2": 13
  },
  "D25": {
    "s1": 3,
    "s2": 12
  },
  "D26": {
    "s1": 4,
    "s2": 12
  },
  "D27": {
    "s1": 5,
    "s2": 12
  },
  "D28": {
    "s1": 6,
    "s2": 12
  },
  "D29": {
    "s1": 4,
    "s2": 14
  },
  "D30": {
    "s1": 3,
    "s2": 15
  },
  "D31": {
    "s1": 2,
    "s2": 16
  },
  "D32": {
    "s1": 1,
    "s2": 17
  }
}

  @classmethod
  def plot_channels_distribution(
    cls,
    OUTPUT_DIR = None,
    figsize=(9, 9),
    font_size=11,
    
):
    """
    Desenha a distribuição bidimensional dos canais de EEG.
    """

    max_s2 = 20
    max_s1 = 20

    n_rows = max_s2 + 1
    n_columns = max_s1 + 1

    row_indices, column_indices = np.indices(
        (n_rows, n_columns)
    )

    distance_to_border = np.minimum.reduce([
        row_indices,
        column_indices,
        n_rows - 1 - row_indices,
        n_columns - 1 - column_indices
    ])

    quadrangular_rings = distance_to_border % 2
    
    # Rosa claro e azul claro.
    color_map = ListedColormap([
        "#f4cccc",
        "#cfe2f3"
    ])

    figure, axis = plt.subplots(figsize=figsize)

    axis.imshow(
        quadrangular_rings,
        cmap=color_map,
        interpolation="none",
        origin="upper",
        aspect="equal"
    )

    # Adiciona o nome de cada canal.
    for channel, position in cls.CHANNELS_POSITIONS.items():
        s2 = position["s2"]
        s1 = position["s1"]

        axis.text(
            s1,
            s2,
            channel,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=font_size
        )

    # Linhas da grade.
    axis.set_xticks(
        np.arange(-0.5, n_columns, 1),
      
    )

    axis.set_yticks(
        np.arange(-0.5, n_rows, 1),
        
    )

    axis.grid(
        which="major",
        linewidth=1.0,
        alpha=1.0
    )

    # Remove números dos eixos.
    axis.tick_params(
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False
    )

    axis.set_xlim(-0.5, n_columns - 0.5)
    axis.set_ylim(n_rows - 0.5, -0.5)

    axis.set_title(
        "2D distribution of EEG channels ",
        fontsize=16,
        pad=15
    )

    axis.set_xlabel("s1")
    axis.set_ylabel("s2")

    plt.tight_layout()

    # plt.show()

    # ======================================================
    # SALVAR
    # ======================================================

    save_dir = (
        OUTPUT_DIR
        / "figures"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    figure_path = (
        save_dir
        / (
            f"channels_2D_distribution.svg"
        )
    )

    figure.savefig(
        figure_path,
        bbox_inches="tight"
    )

    print(
        f"EEG 2D Distribution figure saved: "
        f"{figure_path}"
    )