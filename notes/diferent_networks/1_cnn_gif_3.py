import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont  # <--- Adicionado ImageDraw e ImageFont
import io
import cairosvg 

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

# Títulos correspondentes para cada um dos 9 frames
titulos = [
    "CNN: 1 Saída", "CNN: 2 Saídas", "CNN: 4 Saídas", 
    "CNN: 8 Saídas", "CNN: 16 Saídas", "CNN: 32 Saídas", 
    "CNN: 64 Saídas", "CNN: 128 Saídas", "CNN: 256 Saídas"
]

# 2. Converte SVG para PNG em memória e adiciona o título
imagens = []
for caminho, titulo in zip(caminhos_imagens, titulos):
    # Converte o SVG para bytes PNG
    png_bytes = cairosvg.svg2png(url=caminho)
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA") # Garante compatibilidade de cores
    
    # Criar um objeto para desenhar na imagem
    draw = ImageDraw.Draw(img)
    
    # --- NOVO BLOCO DE FONTE AUMENTADA AQUI ---
    try:
        font = ImageFont.truetype("arial.ttf", 120) # Alterado de 24 para 40
    except IOError:
        font = ImageFont.load_default() 
    # ------------------------------------------
        
    # Desenha o texto na imagem (Ajustei a posição X=40, Y=40 para acomodar o texto maior)
    draw.text((40, 40), titulo, fill=(0, 0, 0), font=font)
    
    imagens.append(img)

# 3. Salva a animação final
imagens[0].save(
    "gif_3.gif",
    save_all=True,
    append_images=imagens[1:], 
    duration=600,              
    loop=0                     
)

print("GIF animado com títulos gerado com sucesso!")
