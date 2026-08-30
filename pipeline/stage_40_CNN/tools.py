
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

import matplotlib as plt
from pathlib import Path



def save_summary_as_pdf(branch, model_stats, save_path):

    # Garante que save_path seja um objeto Path para usar o .mkdir()
    save_path = Path(save_path)

    # Cria o diretório (MODEL_DIR) caso ele não exista
    save_path.mkdir(
        parents=True,
        exist_ok=True
    )

    arquivo_pdf = save_path / f"{branch}.pdf"
    
    # Converte o caminho final completo para string (exigência do ReportLab)
    final_path_str = str(arquivo_pdf)

    # Converte o relatório para string
    text = str(model_stats)
    
    # CORREÇÃO: Substitui os símbolos Unicode por caracteres ASCII universais
    text = text.replace("├─", "|-")
    text = text.replace("└─", "+-")
    text = text.replace("│",  "|")
    
    # Cria o documento PDF usando o caminho do arquivo com a extensão .pdf
    doc = SimpleDocTemplate( final_path_str, pagesize=letter,
                            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    
    # Configura os estilos de texto
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, leading=20, spaceAfter=12)
    text_style = ParagraphStyle('TextStyle', fontName='Courier', fontSize=9, leading=11)
    
    # Adiciona o título e o conteúdo tratado
    story.append(Paragraph(f"Branch: {branch}".title(), title_style))
    story.append(Spacer(1, 10))
    story.append(Preformatted(text, text_style))  # Usa o texto corrigido aqui
    
    # Constrói o PDF
    doc.build(story)
    print(f"Summary successfully saved to PDF without glitches: {final_path_str}")

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


