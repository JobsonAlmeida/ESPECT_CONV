from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

import matplotlib as plt
from pathlib import Path

from torchinfo import summary

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

def save_summary(
    model,
    model_name,
    input_sizes,
    device,
    MODEL_DIR
):

    model_stats = summary(
        model,
        input_size=input_sizes,
        device=device,
        verbose=0,
        mode="eval"
    )

    save_summary_as_pdf(
        model_name,
        model_stats,
        save_path=MODEL_DIR
    )
