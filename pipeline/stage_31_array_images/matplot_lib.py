# def epoch_3d_rcf_t(subject, session, epoch, time_window):

#     file_path = os.path.join(
#         INPUT_DIR,
#         f"{subject}_{session}_power.npy"
#     )

#     if not os.path.exists(file_path):
#         print(f"Arquivo não encontrado: {file_path}")

#     power_array = np.load(file_path)

#     cube = power_array[
#         epoch,
#         :,
#         :,
#         :,
#         time_window
#     ]

#     print(cube.shape)

#     row_grid, column_grid, frequency_grid = np.indices(cube.shape)

#     values = cube.flatten()

#     # Normaliza os valores entre 0 e 1
#     norm = colors.Normalize(
#         vmin=values.min(),
#         vmax=values.max()
#     )

#     normalized_values = norm(values)

#     # Obtém as cores do mapa "hot"
#     cmap = plt.colormaps["hot"]
#     point_colors = cmap(normalized_values)

#     # A transparência será igual ao valor normalizado
#     point_colors[:, 3] = normalized_values

#     # alpha = np.sqrt(normalized_values)
#     # point_colors[:, 3] = alpha

#     figure = plt.figure(figsize=(10, 8))

#     axis = figure.add_subplot(
#         projection="3d"
#     )

#     scatter = axis.scatter(
#         column_grid.flatten(),     # X
#         frequency_grid.flatten(),  # Y
#         row_grid.flatten(),        # Z
#         c=point_colors,
#         s=8
#     )

#     axis.view_init(elev=20, azim=-100) #azim = 120 coloca o row à esquerda

#     mappable = plt.cm.ScalarMappable(
#         norm=norm,
#         cmap=cmap
#     )

#     figure.colorbar(
#         mappable,
#         ax=axis,
#         label="Power"
#     )

#     axis.set_xlabel("Column")
#     axis.set_ylabel("Frequency")
#     axis.set_zlabel("Row")

#     plt.show()

#     return