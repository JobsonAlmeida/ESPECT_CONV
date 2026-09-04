

import numpy as np
import matplotlib.pyplot as plt

# 1. Criamos o array com as 3 classes de números
vetor = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# 2. Definimos as cores para cada classe
# Classe 1 (1-3): Vermelho | Classe 2 (4-6): Verde | Classe 3 (7-9): Azul
cores_classes = {
    1: 'tomato',
    2: 'mediumseagreen',
    3: 'dodgerblue'
}

# 3. Mapeamos cada elemento do vetor para sua respectiva classe e cor
cores_vetor = []
for num in vetor:
    if num <= 3:
        cores_vetor.append(cores_classes[1])
    elif num <= 6:
        cores_vetor.append(cores_classes[2])
    else:
        cores_vetor.append(cores_classes[3])

# 4. Configuração do gráfico para desenhar o vetor
fig, ax = plt.subplots(figsize=(10, 2))

# Desenha cada posição do vetor como um bloco colorido
for i, (num, cor) in enumerate(zip(vetor, cores_vetor)):
    # Desenha o quadrado (bloco da posição)
    rect = plt.Rectangle((i, 0), 0.9, 0.9, facecolor=cor, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    
    # Adiciona o número do vetor dentro do bloco
    ax.text(i + 0.45, 0.45, str(num), color='white', weight='bold', 
            fontsize=14, ha='center', va='center')
    
    # Adiciona o índice da posição (0, 1, 2...) abaixo do bloco
    ax.text(i + 0.45, -0.2, f"pos {i}", color='gray', fontsize=10, ha='center')

# Ajustes de exibição do Matplotlib
ax.set_xlim(-0.2, len(vetor))
ax.set_ylim(-0.4, 1.1)
ax.axis('off')  # Esconde as bordas e eixos padrão do gráfico
plt.title("Visualização do Vetor por Classes", fontsize=14, weight='bold', pad=20)

# Exibe a imagem na tela
plt.show()
