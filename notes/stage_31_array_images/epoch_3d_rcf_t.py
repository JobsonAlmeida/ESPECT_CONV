import os
import numpy as np
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import plotly.graph_objects as go

import plotly.graph_objects as go
import pyvista as pv

current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[2]

INPUT_DIR = PROJECT_ROOT / "processed_data" / "stage_30_array_assembly"
OUTPUT_DIR = PROJECT_ROOT / "processed_data" / current_file.parents[0].name

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]

def rcf_at_time(cube):

    row_grid, column_grid, frequency_grid = np.indices(cube.shape)

    points = np.column_stack([
        column_grid.flatten(),
        frequency_grid.flatten(),
        row_grid.flatten()
    ]).astype(np.float32)

    values = cube.flatten()

    # Normaliza entre 0 e 1
    normalized = (values - values.min() ) / (values.max() - values.min())
    #opacity = normalized
    #opacity = normalized ** 2
    opacity = np.sqrt(normalized)

    cloud = pv.PolyData(points)

    cloud["Power"] = values

    plotter = pv.Plotter()

    plotter.add_mesh(
        cloud,
        scalars="Power",
        cmap="hot",
        point_size=5,
        render_points_as_spheres=True,
        opacity=opacity
    )

    plotter.add_axes(
        xlabel="Column",
        ylabel="Frequency",
        zlabel="Row"
    )

    # plotter.view_isometric()
    # plotter.view_xy()
    plotter.view_yz()

    plotter.camera_position = [   
        (101.1200683690373, -29.727658718647923, 29.15551154305676),
        (10.0, 32.0, 10.0),
        (-0.1425853115967611, 0.09524766550092825, 0.9851889722959133)
    ]

    plotter.show()

    # camera_position = plotter.show(
    #     return_cpos=True
    # )

    #print(camera_position)


def show_rcf_at_time(subject, session, epoch, time_window):

    file_path = os.path.join(
        INPUT_DIR,
        f"{subject}_{session}_power.npy"
    )

    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        return

    power_array = np.load(file_path)

    # Seleciona uma época e uma janela temporal
    # cube.shape:(rows, columns, frequencies)
    cube = power_array[
        epoch,
        :,
        :,
        :,
        time_window
    ]

    print("Cube shape:", cube.shape)

    rcf_at_time(cube)


# def show_rcf_through_time(subject, session, epoch):

#     file_path = os.path.join(
#         INPUT_DIR,
#         f"{subject}_{session}_power.npy"
#     )

#     if not os.path.exists(file_path):
#         print(f"Arquivo não encontrado: {file_path}")
#         return

#     power_array = np.load(file_path)

#     print(power_array.shape)

#     n_epochs, n_y_pos, n_y_pos, n_frequencies, n_time_windows = power_array.shape

#     for time_window in range(n_time_windows):

#         # Seleciona uma época e uma janela temporal
#         # cube.shape:(rows, columns, frequencies)
#         cube = power_array[
#             epoch,
#             :,
#             :,
#             :,
#             time_window
#         ]

#         print("Cube shape:", cube.shape)

#         rcf_at_time(cube)




# def show_rcf_through_time(subject, session, epoch):

#     file_path = os.path.join(
#         INPUT_DIR,
#         f"{subject}_{session}_power.npy"
#     )

#     if not os.path.exists(file_path):
#         print(f"Arquivo não encontrado: {file_path}")
#         return

#     power_array = np.load(file_path)

#     n_epochs, n_y_pos, n_y_pos, n_frequencies, n_time_windows = power_array.shape
#     print(power_array.shape)


#     plotter = pv.Plotter(off_screen=True)

#     cloud = pv.PolyData(points)
#     cloud["Power"] = first_cube.flatten()

#     plotter.add_mesh(
#         cloud,
#         scalars="Power",
#         cmap="hot",
#         point_size=5,
#         render_points_as_spheres=True,
#         clim=[
#             power_array[epoch].min(),
#             power_array[epoch].max()
#         ]
#     )

#     plotter.camera_position = [
#         (101.1200683690373, -29.727658718647923, 29.15551154305676),
#         (10.0, 32.0, 10.0),
#         (-0.1425853115967611, 0.09524766550092825, 0.9851889722959133)
#     ]

#     plotter.open_gif(
#         "evolution.gif",
#         fps=4
#     )

#     for time_window in range(n_time_windows):

#         cube = power_array[
#             epoch,
#             :,
#             :,
#             :,
#             time_window
#         ]

#         cloud["Power"] = cube.flatten()

#         cloud.modified()

#         plotter.write_frame()

#     plotter.close()


 



# def epoch_3d_rcf_through_time(subject, session, epoch):


#     file_path = os.path.join(
#         INPUT_DIR,
#         f"{subject}_{session}_power.npy"
#     )

#     if not os.path.exists(file_path):
#         print(f"Arquivo não encontrado: {file_path}")
#         return

#     power_array = np.load(file_path)

#     # Seleciona uma época e uma janela temporal
#     #
#     # cube.shape:
#     # (rows, columns, frequencies)
#     cube = power_array[
#         epoch,
#         :,
#         :,
#         :,
#         time_window
#     ]

#     print("Cube shape:", cube.shape)


# import os
# import numpy as np
# import pyvista as pv


# def show_rcf_through_time(
#     subject,
#     session,
#     epoch,
#     fps=4
# ):
#     file_path = os.path.join(
#         INPUT_DIR,
#         f"{subject}_{session}_power.npy"
#     )

#     if not os.path.exists(file_path):
#         print(f"Arquivo não encontrado: {file_path}")
#         return

#     # --------------------------------------------------
#     # Carrega os dados
#     # --------------------------------------------------
#     power_array = np.load(file_path)

#     print("Power array shape:", power_array.shape)

#     ( n_epochs, n_rows, n_columns, n_frequencies, n_time_windows) = power_array.shape

#     if epoch < 0 or epoch >= n_epochs:
#         raise ValueError(
#             f"Epoch inválida: {epoch}. "
#             f"Intervalo válido: 0 até {n_epochs - 1}"
#         )

#     # --------------------------------------------------
#     # Seleciona o primeiro instante
#     #
#     # first_cube.shape:
#     # (rows, columns, frequencies)
#     # --------------------------------------------------

#     first_cube = power_array[epoch, :, :, :, 0 ]

#     print("Cube shape:", first_cube.shape)

#     # --------------------------------------------------
#     # Cria as coordenadas 3D
#     #
#     # Isso só precisa ser feito UMA vez, porque
#     # row, column e frequency não mudam no tempo.
#     # --------------------------------------------------

#     row_grid, column_grid, frequency_grid = np.indices(first_cube.shape)

#     points = np.column_stack([
#         column_grid.flatten(),     # X
#         frequency_grid.flatten(),  # Y
#         row_grid.flatten()         # Z
#     ]).astype(np.float32)

#     # --------------------------------------------------
#     # Cria a nuvem de pontos
#     # --------------------------------------------------

#     cloud = pv.PolyData(points)

#     first_values = first_cube.flatten().astype(np.float32)

#     cloud["Power"] = first_values

#     # ------------------------------------------------------------------------------------------
#     # Define uma escala de potência fixa para TODO # o vídeo.
#     #
#     # Isso é importante para que uma mesma cor represente uma mesma potência em todos os frames.
#     # -------------------------------------------------------------------------------------------

#     epoch_data = power_array[ epoch, :, :, :, : ]

#     power_min = epoch_data.min()
#     power_max = epoch_data.max()

#     print("Power min:", power_min)
#     print("Power max:", power_max)

#     # --------------------------------------------------
#     # Cria o Plotter
#     #
#     # off_screen=True significa que não precisamos
#     # abrir uma janela durante a geração do GIF.
#     # --------------------------------------------------

#     plotter = pv.Plotter(
#         off_screen=True,
#         window_size=(1000, 800)
#     )

#     # --------------------------------------------------
#     # Adiciona a nuvem de pontos
#     # --------------------------------------------------

#     plotter.add_mesh(
#         cloud,
#         scalars="Power",
#         cmap="hot",
#         point_size=5,
#         render_points_as_spheres=True,
#         clim=[
#             power_min,
#             power_max
#         ],
#         scalar_bar_args={
#             "title": "Power"
#         }
#     )

#     # --------------------------------------------------
#     # Adiciona os eixos
#     # --------------------------------------------------

#     plotter.add_axes(
#         xlabel="Column",
#         ylabel="Frequency",
#         zlabel="Row"
#     )

#     # --------------------------------------------------
#     # Define a câmera
#     # --------------------------------------------------

#     plotter.camera_position = [
#         (
#             101.1200683690373,
#             -29.727658718647923,
#             29.15551154305676
#         ),
#         (
#             10.0,
#             32.0,
#             10.0
#         ),
#         (
#             -0.1425853115967611,
#             0.09524766550092825,
#             0.9851889722959133
#         )
#     ]

#     # --------------------------------------------------
#     # Caminho do GIF
#     # --------------------------------------------------

#     output_file = os.path.join(
#         OUTPUT_DIR,
#         f"{subject}_{session}_epoch_{epoch}_rcf_time.gif"
#     )

#     # --------------------------------------------------
#     # Inicia a gravação
#     # --------------------------------------------------

#     plotter.open_gif(
#         output_file,
#         fps=fps
#     )

#     # --------------------------------------------------
#     # Percorre todas as janelas temporais
#     # --------------------------------------------------

#     for time_window in range(n_time_windows):

#         cube = power_array[
#             epoch,
#             :,
#             :,
#             :,
#             time_window
#         ]

#         values = cube.flatten().astype(np.float32)

#         # Apenas os valores mudam.
#         # As coordenadas continuam iguais.
#         cloud["Power"] = values

#         # Informa ao VTK/PyVista que os dados mudaram
#         #cloud.modified()

#         # Atualiza a cena
#         plotter.render()

#         # Grava o frame atual no GIF
#         plotter.write_frame()

#         print(
#             f"Frame {time_window + 1}/{n_time_windows} "
#             f"- time_window={time_window}"
#         )

#     # --------------------------------------------------
#     # Finaliza o GIF
#     # --------------------------------------------------

#     plotter.close()

#     print()
#     print(f"GIF salvo em:")
#     print(output_file)



import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyvista as pv


def show_rcf_through_time(
    subject,
    session,
    epoch,
    fps=4
):

    file_path = os.path.join(
        INPUT_DIR,
        f"{subject}_{session}_power.npy"
    )

    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        return

    power_array = np.load(file_path)

    print("Power array shape:", power_array.shape)

    (n_epochs, n_rows, n_columns, n_frequencies, n_time_windows ) = power_array.shape

    if epoch < 0 or epoch >= n_epochs:
        raise ValueError(
            f"Epoch inválida: {epoch}"
        )

    # --------------------------------------------------
    # Pasta onde os frames PNG serão salvos
    # --------------------------------------------------

    frames_dir = Path(OUTPUT_DIR) / (
        f"{subject}_{session}_epoch_{epoch}_frames"
    )

    frames_dir.mkdir( parents=True, exist_ok=True
    )

    # --------------------------------------------------
    # Calcula a escala global de potência
    # para todos os time windows dessa época
    # --------------------------------------------------

    epoch_data = power_array[ epoch, :, :, :, : ]

    power_min = epoch_data.min()
    power_max = epoch_data.max()

    print("Power min:", power_min)
    print("Power max:", power_max)

    # --------------------------------------------------
    # Cria as coordenadas.
    #
    # Elas são iguais para todos os time windows.
    # --------------------------------------------------

    first_cube = power_array[ epoch, :, :, :, 0 ]

    row_grid, column_grid, frequency_grid = np.indices( first_cube.shape )

    points = np.column_stack([
        column_grid.flatten(),     #X
        frequency_grid.flatten(),  #Y
        row_grid.flatten()         #Z
    ]).astype(np.float32)

    # --------------------------------------------------
    # Câmera fixa
    # --------------------------------------------------

    camera_position = [
        (
            101.1200683690373,
            -29.727658718647923,
            29.15551154305676
        ),
        (
            10.0,
            32.0,
            10.0
        ),
        (
            -0.1425853115967611,
            0.09524766550092825,
            0.9851889722959133
        )
    ]



    # --------------------------------------------------
    # Gera um PNG para cada time window
    # --------------------------------------------------

    frame_paths = []

    for time_window in range(n_time_windows):

        cube = power_array[
            epoch,
            :,
            :,
            :,
            time_window
        ]

        values = cube.flatten().astype(np.float32)

        # ----------------------------------------------
        # Normalização da opacidade
        # usando a escala global da época
        # ----------------------------------------------

        if power_max > power_min:
            normalized = ( values - power_min) / ( power_max - power_min)
        else:
            normalized = np.zeros_like(values)

        normalized = np.clip(
            normalized,
            0.0,
            1.0
        )

        opacity = np.sqrt(normalized)

        # ----------------------------------------------
        # Cria a nuvem deste frame
        # ----------------------------------------------

        cloud = pv.PolyData(points)

        cloud["Power"] = values

        # ----------------------------------------------
        # Cria um Plotter novo para este frame
        # ----------------------------------------------

        plotter = pv.Plotter(
            off_screen=True,
            window_size=(1000, 800)
        )

        plotter.add_mesh(
            cloud,
            scalars="Power",
            cmap="hot",
            point_size=5,
            render_points_as_spheres=True,
            opacity=opacity,
            clim=[
                power_min,
                power_max
            ],
            scalar_bar_args={
                "title": "Power"
            }
        )

        plotter.add_axes(
            xlabel="Column",
            ylabel="Frequency",
            zlabel="Row"
        )

        plotter.camera_position = camera_position

        # ----------------------------------------------
        # Nome do PNG
        # ----------------------------------------------

        frame_path = frames_dir / (
            f"frame_{time_window:03d}.png"
        )

        # ----------------------------------------------
        # Salva o frame
        # ----------------------------------------------


        STEP_DURATION = 0.25
        WINDOW_DURATION = 0.5

        start_time = time_window * STEP_DURATION
        end_time = start_time + WINDOW_DURATION

        plotter.add_text(
            f"Time = {start_time:.2f}-{end_time:.2f} s",
            position="upper_left",
            font_size=18,
            color="black"
        )

        plotter.screenshot(
            str(frame_path)
        )

        plotter.close()

        frame_paths.append(frame_path)

        print(
            f"Frame {time_window + 1}/"
            f"{n_time_windows} salvo em "
            f"{frame_path}"
        )

    # --------------------------------------------------
    # Junta os PNGs em um GIF
    # --------------------------------------------------

    gif_path = Path(OUTPUT_DIR) / (
        f"{subject}_{session}_epoch_{epoch}_rcf_time.gif"
    )

    frames = []

    for frame_path in frame_paths:

        frame = imageio.imread(
            frame_path
        )

        frames.append(frame)

    imageio.mimsave(
        gif_path,
        frames,
        fps=fps,
        loop=0
    )

    print()
    print("GIF criado:")
    print(gif_path)