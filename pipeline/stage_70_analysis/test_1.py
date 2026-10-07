import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.pagesizes import A4, landscape

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def criar_pdf_vetor(nome_arquivo):
    # Configuração do documento
    doc = SimpleDocTemplate(nome_arquivo, pagesize=A4, title="Vetor de Índices")
    story = []
    
    # Estilos
    styles = getSampleStyleSheet()
    titulo_style = ParagraphStyle(
        'TituloStyle',
        parent=styles['Heading1'],
        fontSize=12,
        leading=24,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=20,
        alignment=1 # Centralizado
    )
    
    texto_style = ParagraphStyle(
        'TextoStyle',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#555555'),
        alignment=1
    )
    
    # Cabeçalho
    story.append(Paragraph("<b>Visualização do Vetor de Teste</b>", titulo_style))
    story.append(Paragraph("Os índices fornecidos estão organizados sequencialmente em blocos destacados:", texto_style))
    story.append(Spacer(1, 20))
    
    # Dados do vetor
    indices = [6, 7, 8, 10, 11, 18, 19, 20, 49, 51, 61, 62, 64, 68, 74, 82, 92, 100,
               102, 125, 126, 127, 128, 131, 146, 148, 149, 170, 176, 178, 186, 187, 196]
    
    # Divide o vetor em linhas de no máximo 10 elementos para caber perfeitamente na página
    colunas_por_linha = 10
    dados_tabela = []
    
    for i in range(0, len(indices), colunas_por_linha):
        sub_lista = indices[i:i + colunas_por_linha]
        # Se a última linha tiver menos que 10 elementos, preenchemos com strings vazias para manter o alinhamento das células
        estilo_linha = [str(x) for x in sub_lista]
        dados_tabela.append(estilo_linha)
        
    # Configuração visual da tabela (quadradinhos)
    # Definimos tamanhos fixos para as células parecerem quadradinhos
    largura_celula = 40
    altura_celula = 40
    
    # Estilização genérica para os quadradinhos
    estilo_quadradinhos = [
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1E4620')), # Texto verde escuro para contraste
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]
    
    # Aplicar o fundo verde claro e as bordas apenas nas células que possuem números reais
    for linha_idx, linha in enumerate(dados_tabela):
        for col_idx, valor in enumerate(linha):
            if valor != "":
                estilo_quadradinhos.extend([
                    ('BACKGROUND', (col_idx, linha_idx), (col_idx, linha_idx), colors.HexColor('#D4EDDA')), # Verde Claro
                    ('BOX', (col_idx, linha_idx), (col_idx, linha_idx), 1.5, colors.HexColor('#C3E6CB')), # Borda verde um pouco mais firme
                ])
    
    # Construção da tabela
    tabela = Table(dados_tabela, colWidths=[largura_celula]*colunas_por_linha, rowHeights=[altura_celula]*len(dados_tabela))
    tabela.setStyle(TableStyle(estilo_quadradinhos))
    
    story.append(tabela)
    
    # Construir o PDF
    doc.build(story)

if __name__ == "__main__":
    
    criar_pdf_vetor("vetor_indices.pdf")
    print("PDF criado com sucesso como 'vetor_indices.pdf'!")
