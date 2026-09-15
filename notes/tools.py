



def save_all_summaries_in_one_image(model_stats, save_path="model_summary.png"):
    # Converte o relatório para string
    text = str(model_stats)
    
    # Conta as linhas para ajustar a altura da imagem dinamicamente
    lines = text.split('\n')
    num_lines = len(lines)
    
    # Configura o tamanho da imagem (Largura, Altura) proporcional ao texto
    fig, axs = plt.subplots(2, 2, figsize=(16, (num_lines * 0.25) * 2))

    # LOOP PARA CONFIGURAR AS BORDAS VISÍVEIS EM TODOS OS 4 QUADRADOS
    for ax in axs.flat:
        # Remove os números dos eixos X e Y
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Garante que as 4 linhas da borda fiquem visíveis (esquerda, direita, topo, fundo)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')       # Cor da linha do retângulo
            spine.set_linewidth(1.0)       # Espessura da linha

    fig.suptitle(f"Model Summaries", fontsize=12, fontweight='bold')


    # CONFIGURAÇÃO DO PRIMEIRO QUADRANTE (Top-Left)
    ax_model_A = axs[0, 0]
    ax_model_A.set_title("A - Space x Space x Frequency - Channel: Time", fontsize=11, fontweight='bold')

    # Desenha o texto dentro do retângulo A
    ax_model_A.text(0.01, 0.95, text, fontsize=9, fontfamily='monospace', 
                    verticalalignment='top', horizontalalignment='left')
    
    # -------------------------------------------------------------------------
    # DICA: Nos próximos passos, você usará os outros quadrantes assim:
    # ax_model_B = axs[0, 1] -> Top-Right
    # ax_model_C = axs[1, 0] -> Bottom-Left
    # ax_model_D = axs[1, 1] -> Bottom-Right
    # -------------------------------------------------------------------------

    # Salva com margens ajustadas
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Summary successfully saved to: {save_path}")
