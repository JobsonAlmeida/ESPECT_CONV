import sys
from pathlib import Path
from PIL import Image
import io
import cairosvg  # <--- Nova biblioteca para converter SVG

current_file = Path(__file__).resolve()
PROCESSED_DATA = current_file.parents[0]

if str(PROCESSED_DATA) not in sys.path:
    sys.path.append(str(PROCESSED_DATA))

# 1. Lista com o caminho das suas imagens SVG
caminhos_imagens = [
    f"{PROCESSED_DATA}/1_cnn_1_out_relu_pool_1_linear/space1_space2_frequency_time_sub-01.svg", 
    f"{PROCESSED_DATA}/1_cnn_2_out_relu_pool_1_linear/space1_space2_frequency_time_sub-01.svg",
    f"{PROCESSED_DATA}/1_cnn_4_out_relu_pool_1_linear/space1_space2_frequency_time_sub-01.svg",
    f"{PROCESSED_DATA}/1_cnn_8_out_relu_pool_1_linear/space1_space2_frequency_time_sub-01.svg",
    f"{PROCESSED_DATA}/1_cnn_16_out_relu_pool_1_linear/space1_space2_frequency_time_sub-01.svg",
    f"{PROCESSED_DATA}/1_cnn_32_out_relu_pool_1_linear/space1_space2_frequency_time_sub-01.svg",
    f"{PROCESSED_DATA}/1_cnn_64_out_relu_pool_1_linear/space1_space2_frequency_time_sub-01.svg",
    f"{PROCESSED_DATA}/1_cnn_128_out_relu_pool_1_linear/space1_space2_frequency_time_sub-01.svg",
    f"{PROCESSED_DATA}/1_cnn_256_out_relu_pool_out_1_linear/space1_space2_frequency_time_sub-01.svg",
]

# 2. Converte SVG para PNG em memória e carrega no Pillow
imagens = []
for caminho in caminhos_imagens:
    # Converte o SVG para bytes PNG
    png_bytes = cairosvg.svg2png(url=caminho)
    # Transforma os bytes em um objeto que o Pillow consegue ler
    img = Image.open(io.BytesIO(png_bytes))
    imagens.append(img)

# 3. Salva a primeira imagem configurando as outras como quadros do GIF
imagens[0].save(
    "meu_animação.gif",
    save_all=True,
    append_images=imagens[1:], 
    duration=500,              # Aumentei para 500ms (0.5s) para dar tempo de analisar os gráficos da CNN
    loop=0                     
)

print("GIF criado com sucesso a partir dos SVGs!")
