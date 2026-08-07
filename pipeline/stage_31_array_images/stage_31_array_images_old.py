# Imports
from pathlib import Path
import pickle
import mne
import numpy as np
from scipy.signal import spectrogram
import sys
import os
from pathlib import Path

#Variables
current_dir = Path.cwd()

if "ESPECT_CONV" in current_dir.parts:
    idx = current_dir.parts.index("ESPECT_CONV")
    PROJECT_ROOT = Path(*current_dir.parts[:idx + 1])
else:
    PROJECT_ROOT = current_dir

INPUT_DIR = PROJECT_ROOT / "processed_data" / "stage_30_array_assembly"
OUTPUT_DIR = PROJECT_ROOT / "processed_data" / "stage_31_array_images"


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Raiz do Projeto:", PROJECT_ROOT)
print("Entrada:", INPUT_DIR)
print("Saída:", OUTPUT_DIR)


#Reading data
epoch_files = sorted(INPUT_DIR.glob("*_power.npy"))

print(f"Sessões encontradas: {len(epoch_files)}")
for path in epoch_files[:6]:
    print(" -", path.relative_to(PROJECT_ROOT))

# Loading data from one sample

data_path = epoch_files[0]

power_array = np.load(data_path)
print(power_array.shape)

frequencies_path = INPUT_DIR/"frequencies.npy"
frequencies = np.load(frequencies_path)

times_path = INPUT_DIR/"times.npy"
times = np.load(times_path)

epoch = 0
time_window = 0
cube = power_array[
    epoch,
    :,
    :,
    :,
    time_window
]

print(cube.shape)


# import numpy as np
# import pyvista as pv

# # cube.shape = (21,21,65)

# volume = np.transpose(
#     cube,
#     (1, 0, 2)
# ).astype(np.float32)

# # Cria uma grade estruturada
# grid = pv.ImageData()

# grid.dimensions = volume.shape

# grid.spacing = (1, 1, 1)

# grid.origin = (0, 0, 0)

# # Coloca os valores de potência na grade
# grid.point_data["power"] = volume.ravel(order="F")

# plotter = pv.Plotter()

# plotter.add_volume(
#     grid,
#     scalars="power",
#     cmap="hot",
#     opacity="sigmoid",
#     shade=False
# )

# plotter.add_axes()

# plotter.show()

#----------------

import matplotlib.pyplot as plt

figure, axis = plt.subplots(figsize=(6,6))

for frequency in range(cube.shape[2]):

    axis.clear()

    image = axis.imshow(
        cube[:, :, frequency],
        cmap="hot",
        origin="upper"
    )

    axis.set_title(f"Frequency {frequency}")

    plt.pause(0.2)

plt.show()


# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np

# rows, columns, frequencies = cube.shape

# row_grid, column_grid, frequency_grid = np.indices(
#     cube.shape
# )

# figure = plt.figure(figsize=(10,8))

# axis = figure.add_subplot(
#     projection="3d"
# )

# values = cube.flatten()

# mask = values > 0

# scatter = axis.scatter(
#     column_grid.flatten()[mask],
#     row_grid.flatten()[mask],
#     frequency_grid.flatten()[mask],
#     c=values[mask],
#     cmap="hot",
#     s=8
# )

# figure.colorbar(scatter)

# axis.set_xlabel("Column")
# axis.set_ylabel("Row")
# axis.set_zlabel("Frequency")

# plt.show()

#------------

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
# import numpy as np

# rows, columns, frequencies = cube.shape

# row_grid, column_grid, frequency_grid = np.indices(
#     cube.shape
# )

# figure = plt.figure(figsize=(10, 8))

# axis = figure.add_subplot(
#     projection="3d"
# )

# values = cube.flatten()

# # Mantém inclusive os zeros.
# mask = values >= 0

# # Colormap:
# # branco -> amarelo -> laranja -> vermelho -> preto
# power_cmap = LinearSegmentedColormap.from_list(
#     "power_cmap",
#     [
#         "white",
#         "yellow",
#         "orange",
#         "red",
#         "black"
#     ]
# )

# scatter = axis.scatter(
#     column_grid.flatten()[mask],
#     row_grid.flatten()[mask],
#     frequency_grid.flatten()[mask],
#     c=values[mask],
#     cmap=power_cmap,
#     s=8,
#     vmin=0,
#     vmax=values.max()
# )

# figure.colorbar(
#     scatter,
#     ax=axis,
#     label="Power"
# )

# axis.set_xlabel("Column")
# axis.set_ylabel("Row")
# axis.set_zlabel("Frequency")

# plt.show()


# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import numpy as np

# rows, columns, frequencies = cube.shape

# row_grid, column_grid, frequency_grid = np.indices(cube.shape)

# values = cube.flatten()

# # Normaliza os valores entre 0 e 1
# norm = colors.Normalize(
#     vmin=values.min(),
#     vmax=values.max()
# )

# normalized_values = norm(values)

# # Obtém as cores do mapa "hot"
# cmap = plt.colormaps["hot"]
# point_colors = cmap(normalized_values)

# # A transparência será igual ao valor normalizado
# point_colors[:, 3] = normalized_values

# # alpha = np.sqrt(normalized_values)

# # point_colors[:, 3] = alpha


# figure = plt.figure(figsize=(10, 8))

# axis = figure.add_subplot(
#     projection="3d"
# )

# scatter = axis.scatter(
#     column_grid.flatten(),
#     row_grid.flatten(),
#     frequency_grid.flatten(),
#     c=point_colors,
#     s=8
# )

# mappable = plt.cm.ScalarMappable(
#     norm=norm,
#     cmap=cmap
# )

# figure.colorbar(
#     mappable,
#     ax=axis,
#     label="Power"
# )

# axis.set_xlabel("Column")
# axis.set_ylabel("Row")
# axis.set_zlabel("Frequency")

# plt.show()