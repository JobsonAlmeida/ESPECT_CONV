import sys
from pathlib import Path
import numpy as np
import pickle
import torch
import numpy as np

from reportlab.lib.pagesizes import A4, landscape

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ==========================================================
# CONFIGURAÇÕES DE CAMINHO & IMPORTS LOCAIS
# ==========================================================

current_file = Path(__file__).resolve()

PROJECT_ROOT = current_file.parents[2]

PIPELINE_ROOT = current_file.parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.append(
        str(PROJECT_ROOT)
    )

if str(PIPELINE_ROOT) not in sys.path:

    sys.path.append(
        str(PIPELINE_ROOT)
    )


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

N_FOLDS = 6

INPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_30_psd_array_assembly"
)

INDICES_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_40_obtain_indices"
)

BRANCHES_STAGE_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_50_execute_branch"
)

FUSIONS_STAGE_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_60_execute_fusion"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "stage_70_analysis"
)


# ==========================================================
# CONDITIONS
# ==========================================================

CONDITION_MAP = {

    "PRONOUNCED_SPEECH": 0,
    "INNER_SPEECH": 1,
    "VISUALIZED_CONDITION": 2,

}


DEFAULT_SUBJECTS = [
    f"sub-{i:02d}"
    for i in range(1, 11)
]

# ==========================================================
# SUJEITOS
# ==========================================================

subjects = [
    f"sub-{i:02d}"
    for i in range(1, 11)
]

def create_pdf(file_name, SUB_CONDIT_DIR, test_indices):

    doc = SimpleDocTemplate(
        f"{SUB_CONDIT_DIR}/{file_name}", 
        pagesize=landscape(A4), 
        title="Vetor de Índices",
        leftMargin=20,   # Margem Esquerda
        rightMargin=20,  # Margem Direita
        topMargin=30,    # Opcional: Margem Superior
        bottomMargin=30  # Opcional: Margem Inferior
    )

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
    
    # Divide o vetor em linhas de no máximo 10 elementos para caber perfeitamente na página
    colunas_por_linha = 40
    dados_tabela = []

    primeira_linha = ["indices:", ""] + [str(x) for x in test_indices[:colunas_por_linha-2]]
    dados_tabela.append(primeira_linha)
    
    for i in range(colunas_por_linha-2, len(test_indices), colunas_por_linha):
        sub_lista = test_indices[i:i + colunas_por_linha]
        # Se a última linha tiver menos que 10 elementos, preenchemos com strings vazias para manter o alinhamento das células
        estilo_linha = [str(x) for x in sub_lista]
        dados_tabela.append(estilo_linha)
        
    # Configuração visual da tabela (quadradinhos)
    # Definimos tamanhos fixos para as células parecerem quadradinhos
    largura_celula = 20
    altura_celula = 15
    
    # Estilização genérica para os quadradinhos
    estilo_quadradinhos = [

          # Mescla a célula da coluna 0 até a coluna 1 na linha 0 para o texto "Índices:"
        ('SPAN', (0, 0), (2, 0)),
        ('ALIGN', (0, 0), (1, 0), 'LEFT'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 12),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.HexColor('#555555')),



        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1E4620')), # Texto verde escuro para contraste
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]
    
    # Aplicar o fundo verde claro e as bordas apenas nas células que possuem números reais
    for linha_idx, linha in enumerate(dados_tabela):
        for col_idx, valor in enumerate(linha):
            if valor != "":

                if linha_idx == 0 and col_idx < 3:
                    continue


                estilo_quadradinhos.extend([
                    ('BACKGROUND', (col_idx, linha_idx), (col_idx, linha_idx), colors.HexColor('#D4EDDA')), # Verde Claro
                    ('BOX', (col_idx, linha_idx), (col_idx, linha_idx), 1.5, colors.HexColor('#C3E6CB')), # Borda verde um pouco mais firme
                ])
    
    # Construção da tabela
    tabela = Table(dados_tabela, colWidths=[largura_celula]*colunas_por_linha, rowHeights=[altura_celula]*len(dados_tabela))
    tabela.hAlign = 'LEFT' 
    tabela.setStyle(TableStyle(estilo_quadradinhos))
    
    story.append(tabela)
    story.append(tabela)
    
    # Construir o PDF
    doc.build(story)



def include_results(story, BRANCHES_SUB_CONDIT_DIR, branches, fusions , fold, model_state):

    """Includes lines in the comparison graph"""

    # =======================================
    # CHECKING IF ALL LABELS USED ARE THE SAME
    # =======================================

    aux = None

    for branch in branches:

        fold_checkpoint = (
            BRANCHES_SUB_CONDIT_DIR
            / f"{branch}_branch"
            / f"{branch}_fold_{fold}.pth"
        )

        if not fold_checkpoint.exists():

            raise FileNotFoundError(
                f"Fold checkpoint file not found:\n"
                f"{fold_checkpoint}"
            )

        # ==================================================
        # CARREGAR RESUMO DOS FOLDS
        # ==================================================

        checkpoint = torch.load(
            fold_checkpoint,
            map_location="cpu",
            weights_only=False
        )

        all_labels = checkpoint[model_state]["all_labels"]

        if aux is None:

            aux = all_labels.copy()

        elif not np.array_equal(aux, all_labels ):

            raise ValueError(
                    "Arrays are diferent."
                )

        



        



    # # Divide o vetor em linhas de no máximos elementos para caber perfeitamente na página
    # colunas_por_linha = 40
    # dados_tabela = []

    # primeira_linha = ["indices:", ""] + [str(x) for x in all_predictions[:colunas_por_linha-2]]
    # dados_tabela.append(primeira_linha)

    # for i in range(colunas_por_linha-2, len(all_predictions), colunas_por_linha):
    #     sub_lista = all_predictions[i:i + colunas_por_linha]
    #     # Se a última linha tiver menos que 10 elementos, preenchemos com strings vazias para manter o alinhamento das células
    #     estilo_linha = [str(x) for x in sub_lista]
    #     dados_tabela.append(estilo_linha)
        
    # # Configuração visual da tabela (quadradinhos)
    # # Definimos tamanhos fixos para as células parecerem quadradinhos
    # largura_celula = 20
    # altura_celula = 15

    # # Estilização genérica para os quadradinhos
    # estilo_quadradinhos = [

    #         # Mescla a célula da coluna 0 até a coluna 1 na linha 0 para o texto "Índices:"
    #     ('SPAN', (0, 0), (2, 0)),
    #     ('ALIGN', (0, 0), (1, 0), 'LEFT'),
    #     ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
    #     ('FONTSIZE', (0, 0), (1, 0), 12),
    #     ('TEXTCOLOR', (0, 0), (1, 0), colors.HexColor('#555555')),



    #     ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    #     ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    #     ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
    #     ('FONTSIZE', (0, 0), (-1, -1), 8),
    #     ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1E4620')), # Texto verde escuro para contraste
    #     ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    # ]

    # # Aplicar o fundo verde claro e as bordas apenas nas células que possuem números reais
    # for linha_idx, linha in enumerate(dados_tabela):
    #     for col_idx, valor in enumerate(linha):
    #         if valor != "":

    #             if linha_idx == 0 and col_idx < 3:
    #                 continue


    #             estilo_quadradinhos.extend([
    #                 ('BACKGROUND', (col_idx, linha_idx), (col_idx, linha_idx), colors.HexColor('#D4EDDA')), # Verde Claro
    #                 ('BOX', (col_idx, linha_idx), (col_idx, linha_idx), 1.5, colors.HexColor('#C3E6CB')), # Borda verde um pouco mais firme
    #             ])

    # # Construção da tabela
    # tabela = Table(dados_tabela, colWidths=[largura_celula]*colunas_por_linha, rowHeights=[altura_celula]*len(dados_tabela))
    # tabela.hAlign = 'LEFT' 
    # tabela.setStyle(TableStyle(estilo_quadradinhos))

    # story.append(tabela)



# ==========================================================
# FUNÇÃO PRINCIPAL
# ==========================================================

def model_results_comparison(
    subjects=subjects,
    condition_folder = "PRONOUNCED_SPEECH",
    N_FOLDS=N_FOLDS,
    model_states = ["minimum_loss_model", "maximum_accuracy_model"],
    branches = ["space1_space2_frequency_time"]
    
):

    # ======================================================
    # CHECAGEM DA VARIÁVEL SUBJECTS
    # ======================================================

    if isinstance(
        subjects,
        str
    ):
        subjects = [
            subjects
        ]

    elif subjects is None:

        subjects = (
            DEFAULT_SUBJECTS
        )
  
    # ======================================================
    # LOOP DOS SUJEITOS
    # ======================================================

    for subject in subjects:

        print()

        print(
            "#" * 70
        )

        print(
            f"SUJEITO: {subject}"
        )

        print(
            "#" * 70
        )

        # ==================================================
        # DIRETÓRIO DOS MODELOS
        # ==================================================

        BRANCHES_SUB_CONDIT_DIR = (
            BRANCHES_STAGE_DIR
            / subject
            / condition_folder.lower()
        )

        SUB_CONDIT_OUTPUT_DIR = (
            OUTPUT_DIR
            /subject
            /condition_folder.lower()
        )
    
        SUB_CONDIT_OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # ==================================================
        # LOOP DOS FOLDS
        # ==================================================

        for fold in range(
            1,
            N_FOLDS + 1
        ):

            print()

            print(
                "=" * 70
            )

            print(
                f"FOLD {fold}/{N_FOLDS}"
            )

            print(
                "=" * 70
            )


            doc = SimpleDocTemplate(
                f"{SUB_CONDIT_OUTPUT_DIR}/comparison_{subject}.pdf", 
                pagesize=landscape(A4), 
                title="Vetor de Índices",
                leftMargin=20,   # Margem Esquerda
                rightMargin=20,  # Margem Direita
                topMargin=30,    # Opcional: Margem Superior
                bottomMargin=30  # Opcional: Margem Inferior
            )

            story = []

            styles = getSampleStyleSheet() 
            title_style = ParagraphStyle(
                'TitleStyle',
                parent=styles['Heading1'],
                fontSize=20,
                leading=24,
                textColor=colors.HexColor('#2C3E50'),
                spaceAfter=20,
                alignment=1 # Centralized
            )

            story.append(Paragraph(f"<b> {subject.title()} - {" ".join(f"{condition_folder}".split("_")).title()}</b>", title_style))

            for model_state in model_states: 

                text_style = ParagraphStyle(
                    'TextStyle',
                    parent=styles['Normal'],
                    fontSize=12,
                    leading=16,
                    textColor=colors.HexColor('#555555'),
                    alignment=0 # Left
                )

                story.append(Paragraph(f"<b> {" ".join(f"{model_state}".split("_")).title()} State</b>"))


                include_results(
                    story, 
                    BRANCHES_SUB_CONDIT_DIR, 
                    branches,
                    "",
                    fold, 
                    model_state)


     

            doc.build(story)








# ==========================================================
# EXECUTAR
# ==========================================================

if __name__ == "__main__":

    model_results_comparison(     
        subjects=[
            "sub-01"
        ],
        condition_folder = "PRONOUNCED_SPEECH",
        branches= [
            "space1_space2_frequency_time",
            "space1_space2_time_frequency",
            "time_frequency_space2_space1",
            "time_frequency_space1_space2"
            ]
    )