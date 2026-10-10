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



COLUMNS_PER_ROW = 100
HEADING_LENGTH = 3

HEADING_WIDTH_1 = 15
CELL_WIDTH_1 = 30 # 30 é bom

HEADING_WIDTH_2 = 15
CELL_WIDTH_2 = 20


CELL_HEIGHT = 15

HEADING_FONTSIZE_1 = 8
CELL_FONTSIZE_1 = 8


HEADING_FONTSIZE_2 = 8
CELL_FONTSIZE_2 = 4

# ==========================================================
# SUJEITOS
# ==========================================================

subjects = [
    f"sub-{i:02d}"
    for i in range(1, 11)
]

def checking_indices_labels(
        level, 
        fold,
        model_state,
        BRANCHES_SUB_CONDIT_DIR = None, 
        branches = None, 
        FUSIONS_SUB_CONDIT_DIR= None,
        fusions_one_step = None,
        
):


    if not branches and not fusions_one_step:

        raise ValueError("Choose at least one branch or one fusion!")


    aux_indices = None
    aux_labels = None    

    if branches:

        for branch in branches:

            # ==================================================
            # LOADING BRANCH CHECKPOINT
            # ==================================================

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


            checkpoint = torch.load(
                fold_checkpoint,
                map_location="cpu",
                weights_only=False
            )


            match level:
            
                case "test":

                    indices = checkpoint["test_indices"]            
                    labels = checkpoint[model_state]["test_labels"]

                case "validation":

                    indices = checkpoint["val_indices"]            
                    labels = checkpoint[model_state]["validation_labels"]

                case _:

                    raise ValueError(
                        f"Invalid level: {level}"
                        )

            # =================================================
            # VERIFYING INDICES
            # =================================================
            if aux_indices is None:

                aux_indices = indices.copy()

            elif not np.array_equal(aux_indices, indices):

                raise ValueError(
                        "Indices are diferent!"
                    )

            # =================================================
            # VERIFYING LABELS
            # =================================================
            if aux_labels is None:

                aux_labels = labels.copy()

            elif not np.array_equal(aux_labels, labels ):

                raise ValueError(
                        "Labels are diferent!"
                )


    if fusions_one_step:

        for fusion in fusions_one_step:

            # ==================================================
            # LOADING FUSION CHECKPOINT
            # ==================================================

            fold_checkpoint = (
                FUSIONS_SUB_CONDIT_DIR
                / f"{fusion}_from_{model_state}_in_branches"
                / f"{fusion}_fold_{fold}.pth"
            )

            if not fold_checkpoint.exists():

                raise FileNotFoundError(
                    f"Fold checkpoint file not found:\n"
                    f"{fold_checkpoint}"
                )

            checkpoint = torch.load(
                fold_checkpoint,
                map_location="cpu",
                weights_only=False
            )


            match level:
            
                case "test":

                    indices = checkpoint["dataset"]["test_indices"]            
                    labels = checkpoint["test_set"]["labels"]

                case "validation":

                    indices = checkpoint["dataset"]["val_indices"]            
                    labels = checkpoint["validation_set"]["labels"]


                case _:

                    raise ValueError(
                        f"Invalid level: {level}"
                        )

            # =================================================
            # VERIFYING INDICES
            # =================================================
        

            if not np.array_equal(aux_indices, indices):

                raise ValueError(
                        "Indices are diferent!"
                    )

            # =================================================
            # VERIFYING LABELS
            # =================================================

            if not np.array_equal(aux_labels, labels ):

                raise ValueError(
                        "Labels are diferent!"
                )


    return  aux_indices, aux_labels


def include_indices_labels(
        story, 
        indices_labels, 
        heading_text,       
        box_background,
        box_line_color,        
        box_text_color='#2C3E50',
        heading_text_color = "#2C3E50",
        line_thickness = 0.5,        
        line_below = False,
        line_below_thickness = 0.5,
        line_below_color = '#2C3E50',
        line_above = False,
        line_above_thickness = 0.5,
        line_above_color = '#2C3E50',
        show_probabilities = False,

):



    if show_probabilities:
        heading_width = HEADING_WIDTH_1
        cell_width = CELL_WIDTH_1
        heading_fontsize = HEADING_FONTSIZE_1
        cell_fontsize = CELL_FONTSIZE_1

    else:
        heading_width = HEADING_WIDTH_2
        cell_width = CELL_WIDTH_2
        heading_fontsize = HEADING_FONTSIZE_1
        cell_fontsize = HEADING_FONTSIZE_1


    available_width = (A4)[0] - 20

    columns_per_row = min(
        COLUMNS_PER_ROW,
        HEADING_LENGTH + int(
            (available_width - HEADING_LENGTH * heading_width)
            // cell_width
        )
    )

    samples_first_row = columns_per_row - HEADING_LENGTH

    if samples_first_row <= 0:
        raise ValueError("Not enough space for the table.")
    
    table_data = []

    heading = [heading_text] + [""]*(HEADING_LENGTH-1)    

    first_line = heading + list(
        indices_labels[:samples_first_row]
    )
    table_data.append(first_line)

    elements_per_line = columns_per_row - HEADING_LENGTH
    for i in range(
        samples_first_row,
        len(indices_labels),
        elements_per_line
    ):

        line = (

            [""]*HEADING_LENGTH + indices_labels[i:i + elements_per_line].tolist()
        )

        table_data.append(line)

    # =============================================
    # DEFINING GENERAL STYLE
    # =============================================
        
    square_style = [

        # Cells
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), cell_fontsize),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor(box_text_color)),
        
        ('TOPPADDING', (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),

        # Heading
        ('SPAN', (0, 0), (HEADING_LENGTH-1, 0)),
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),
        ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, 0), heading_fontsize),
        ('TEXTCOLOR', (0, 0), (0, 0), colors.HexColor(heading_text_color)),
    ]

    # ==============================================
    # SPECIFIC STYLE FOR EACH CELL
    # ==============================================

    for line_idx, line in enumerate(table_data):
        for col_idx, value in enumerate(line):

            if value != "":          

                if line_idx == 0 and col_idx < HEADING_LENGTH:
                    continue

                square_style.extend([
                    ('BACKGROUND', (col_idx, line_idx), (col_idx, line_idx), colors.HexColor(box_background)),
                    ('BOX', (col_idx, line_idx), (col_idx, line_idx), line_thickness, colors.HexColor(box_line_color)),
                ])

                if line_below: 
                    square_style.extend([
                        ('LINEBELOW', (col_idx, line_idx), (col_idx, line_idx), line_below_thickness, colors.HexColor(line_below_color)),
                ])

                if line_above: 
                    square_style.extend([
                        ('LINEABOVE', (col_idx, line_idx), (col_idx, line_idx), line_above_thickness, colors.HexColor(line_above_color)),
                ])

    # ==============================================
    # DEFINING PARAGRAPH STYLE
    # ==============================================

    cell_p_style = ParagraphStyle(
        'CellProbabilityStyle',
        fontName='Helvetica-Bold',
        fontSize=cell_fontsize,
        leading=cell_fontsize + 3,
        textColor=colors.HexColor(box_text_color),
        alignment=1,
        spaceBefore=0,
        spaceAfter=0
    )

    # ==============================================
    # CONVERTING DATA TO STRINGS
    # ==============================================

    table_data_string = []

    for line_idx, line in enumerate(table_data):

        string_line = []

        for col_idx, value in enumerate(line):

            cell_text = str(value)

            string_line.append(
                Paragraph(cell_text, cell_p_style)
            )

        table_data_string.append(string_line)

    # ==============================================
    # CREATING TABLE
    # ==============================================

    column_width = (
        [heading_width] * HEADING_LENGTH
        + [cell_width] * (columns_per_row - HEADING_LENGTH)
    )

    table = Table(
        table_data_string,
        colWidths=column_width,
        rowHeights=[CELL_HEIGHT] * len(table_data)
    )

    table.hAlign = 'LEFT'

    table.setStyle(
        TableStyle(square_style)
    )

    story.append(table)

    return story, table_data


def retrive_probabilities_branches(       
        branch,
        level,
        fold, 
        model_state,
        BRANCHES_SUB_CONDIT_DIR,

        
):

    
    """Includes lines in the comparison graph"""


    match branch:

        # ------------------------------------------

        case "space1_space2_frequency_time":

            heading_text = "s1_s2_f_t"

        # ------------------------------------------

        case "space1_space2_time_frequency":

            heading_text = "s1_s1_t_f"


        # ------------------------------------------

        case "time_frequency_space2_space1":

            heading_text = "t_f_s2_s1"

        # ------------------------------------------

        case "time_frequency_space1_space2":

            heading_text = "t_f_s1_s2"

        case _:

            raise ValueError(
                f"Invalid branch: {branch}"
                )

    # ==================================================
    # LOADING DATA
    # ==================================================

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


    checkpoint = torch.load(
        fold_checkpoint,
        map_location="cpu",
        weights_only=False
    )

    match level:

        case "test":
            
            probabilities = checkpoint[model_state]["test_probabilities"]

        case "validation":
                    
            probabilities = checkpoint[model_state]["validation_probabilities"]

        case _:

            raise ValueError(
                f"Invalid level: {level}"
            )

    return probabilities, heading_text


def retrive_predictions_fusions_one_step(       
        FUSIONS_SUB_CONDIT_DIR,
        fusion,
        level,
        fold, 
        model_state,
        
):

    
    """Includes lines in the comparison graph"""


    match fusion:

        # ------------------------------------------

        case "majority_voting_fusion":

            heading_text = "maj_vot_fus"

        # ------------------------------------------

    
        case _:

            raise ValueError(
                f"Invalid branch: {fusion}"
                )

    # ==================================================
    # LOADING DATA
    # ==================================================

    fold_checkpoint = (
        FUSIONS_SUB_CONDIT_DIR
        / f"{fusion}_from_{model_state}_in_branches"
        / f"{fusion}_fold_{fold}.pth"
    )

    if not fold_checkpoint.exists():

        raise FileNotFoundError(
            f"Fold checkpoint file not found:\n"
            f"{fold_checkpoint}"
        )


    checkpoint = torch.load(
        fold_checkpoint,
        map_location="cpu",
        weights_only=False
    )

    match level:

        case "test":
            
            predictions = checkpoint["test_set"]["predictions"]

        case _:

            raise ValueError(
                f"Invalid level: {level}"
            )

    return predictions, heading_text



# def include_probabilities(
#         story, 
#         pr, 
#         box_standard_background,
#         box_success_background, 
#         box_line_color,
#         labels_data,
#         heading_text,
#         box_text_color='#2C3E50',
#         heading_text_color = "#2C3E50",
#         box_line_thickness = 0.5,
#         line_below = False,
#         line_below_thickness = 1.0,
#         line_below_color = '#2C3E50',
#         line_above = False,
#         line_above_thickness = 1.0,
#         line_above_color = '#2C3E50',
#         show_probabilities = False  
# ):

#     table_data = []

#     if show_probabilities:

#         pr = np.round(pr * 100, 2)

#         HEADING_WIDTH = HEADING_WIDTH_1
#         CELL_WIDTH = CELL_WIDTH_1
#         heading_fontsize = HEADING_FONTSIZE_1
#         cell_fontsize = CELL_FONTSIZE_2

#         pr = [f"[{x[0]} {x[1]}<br/>{x[2]} {x[3]}]" for x in pr]
        

#     else:

#         pr = (
#             pr.argmax(
#                 axis=1
#             )
#         )

#         HEADING_WIDTH = HEADING_WIDTH_2
#         CELL_WIDTH = CELL_WIDTH_2
#         heading_fontsize = HEADING_FONTSIZE_1
#         cell_fontsize = CELL_FONTSIZE_1
    


#     heading = [heading_text] + [""]*(HEADING_LENGTH-1)    
#     first_line = heading + [str(x) for x in pr[:COLUMNS_PER_ROW-(HEADING_LENGTH)]]
#     table_data.append(first_line)

#     for i in range(COLUMNS_PER_ROW-(HEADING_LENGTH), len(pr), COLUMNS_PER_ROW):
#         sub_list = pr[i:i + COLUMNS_PER_ROW]
#         text_line = [str(x) for x in sub_list]
#         table_data.append(text_line)
        
#     square_style = [

#         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#         ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
#         ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
#         ('FONTSIZE', (0, 0), (-1, -1), cell_fontsize),
#         ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor(box_text_color)),
#         ('BOTTOMPADDING', (0, 0), (-1, -1), 0),

#         ('SPAN', (0, 0), (HEADING_LENGTH-1, 0)),
#         ('ALIGN', (0, 0), (0, 0), 'LEFT'),
#         ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),
#         ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
#         ('FONTSIZE', (0, 0), (0, 0), heading_fontsize),
#         ('TEXTCOLOR', (0, 0), (0, 0), colors.HexColor(heading_text_color)),
#     ]

#     # Aplicar o fundo verde claro e as bordas apenas nas células que dão match
#     for line_idx, line in enumerate(table_data):
#         for col_idx, value in enumerate(line):

#             if value != "":          

#                 if line_idx == 0 and col_idx < HEADING_LENGTH:
#                     continue


#                 square_style.extend([
#                     ('BACKGROUND', (col_idx, line_idx), (col_idx, line_idx), colors.HexColor(box_standard_background)),
#                     ('BOX', (col_idx, line_idx), (col_idx, line_idx), box_line_thickness, colors.HexColor(box_line_color)),
#                 ])

#                 if table_data[line_idx][col_idx] == labels_data[line_idx][col_idx]:
#                     square_style.extend([
#                         ('BACKGROUND', (col_idx, line_idx), (col_idx, line_idx), colors.HexColor(box_success_background))
#                 ])

#                 if line_below: 
#                     square_style.extend([
#                         ('LINEBELOW', (col_idx, line_idx), (col_idx, line_idx), line_below_thickness, colors.HexColor(line_below_color)),
#                 ])

#                 if line_above: 
#                     square_style.extend([
#                         ('LINEABOVE', (col_idx, line_idx), (col_idx, line_idx), line_above_thickness, colors.HexColor(line_above_color)),
#                 ])

#     column_width = [HEADING_WIDTH]*HEADING_LENGTH + [CELL_WIDTH]*(COLUMNS_PER_ROW - HEADING_LENGTH)
#     table = Table(table_data, colWidths=column_width, rowHeights=[CELL_HEIGHT]*len(table_data))
#     table.hAlign = 'LEFT' 
#     table.setStyle(TableStyle(square_style))

#     story.append(table)

#     return story

def include_probabilities(
        story,
        probabilities,
        box_standard_background,
        box_success_background,
        box_line_color,
        labels_data,
        heading_text,
        box_text_color='#2C3E50',
        heading_text_color='#2C3E50',
        box_line_thickness=0.5,
        line_below=False,
        line_below_thickness=1.0,
        line_below_color='#2C3E50',
        line_above=False,
        line_above_thickness=1.0,
        line_above_color='#2C3E50',
        show_probabilities=False
):

    # ========================================
    # OBTAINING TABLE DATA
    # ========================================

    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError(
            "Probabilities must have shape (n_samples, 4)."
        )

    if show_probabilities:
        heading_width = HEADING_WIDTH_1
        cell_width = CELL_WIDTH_1
        heading_fontsize = HEADING_FONTSIZE_1
        cell_fontsize = CELL_FONTSIZE_2

    else:
        heading_width = HEADING_WIDTH_2
        cell_width = CELL_WIDTH_2
        heading_fontsize = HEADING_FONTSIZE_1
        cell_fontsize = CELL_FONTSIZE_1

    # ===========================================================
    # OBTAINING COLUMNS PER ROW ACCORDING OPTION OR TABLE WIDTH
    # ===========================================================
    available_width = (A4)[0] - 20

    columns_per_row = min(
        COLUMNS_PER_ROW,
        HEADING_LENGTH + int(
            (available_width - HEADING_LENGTH * heading_width)
            // cell_width
        )
    )

    samples_first_row = columns_per_row - HEADING_LENGTH

    if samples_first_row <= 0:
        raise ValueError("Not enough space for the table.")



    # ============================================
    # BUILDING THE TABLE
    # ============================================
    table_data = []

    heading = [heading_text] + [""] * (HEADING_LENGTH - 1)

    first_line = heading + list(
        probabilities[:samples_first_row]
    )

    table_data.append(first_line)

    elements_per_line = columns_per_row - HEADING_LENGTH

    for i in range(
        samples_first_row,
        len(probabilities),
        elements_per_line
    ):

        line = (
            [""]*HEADING_LENGTH + probabilities[i:i + elements_per_line].tolist()
        )

        table_data.append(line)

    # =============================================
    # DEFINING GENERAL STYLE
    # =============================================

    square_style = [

        # Cells
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), cell_fontsize),
        ('TEXTCOLOR', (0, 0), (-1, -1),
         colors.HexColor(box_text_color)),

        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),

        # Heading
        ('SPAN', (0, 0), (HEADING_LENGTH - 1, 0)),
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),
        ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, 0), heading_fontsize),
        ('TEXTCOLOR', (0, 0), (0, 0),
         colors.HexColor(heading_text_color)),
    ]


    # ====================================================
    # CHECKING IF TABLE DATA AND LABELS DATA MATCH
    # ===================================================
    assert len(table_data) == len(labels_data)

    for line_idx in range(len(table_data)):
        assert len(table_data[line_idx]) == len(labels_data[line_idx])

    # ==============================================
    # SPECIFIC STYLE FOR EACH CELL
    # ==============================================  

    for line_idx, line in enumerate(table_data):

        for col_idx, value in enumerate(line):

                if line_idx == 0 and col_idx < HEADING_LENGTH:
                    continue

                if isinstance(value, str) and value == "":
                    continue

                square_style.extend([
                    (
                        'BACKGROUND',
                        (col_idx, line_idx),
                        (col_idx, line_idx),
                        colors.HexColor(box_standard_background)
                    ),
                    (
                        'BOX',
                        (col_idx, line_idx),
                        (col_idx, line_idx),
                        box_line_thickness,
                        colors.HexColor(box_line_color)
                    ),
                ])

                predicted_label = np.argmax(table_data[line_idx][col_idx])
                true_label = int(labels_data[line_idx][col_idx])

                if predicted_label == true_label:

                    square_style.append(
                        (
                            'BACKGROUND',
                            (col_idx, line_idx),
                            (col_idx, line_idx),
                            colors.HexColor(box_success_background)
                        )
                    )

                if line_below:

                    square_style.append(
                        (
                            'LINEBELOW',
                            (col_idx, line_idx),
                            (col_idx, line_idx),
                            line_below_thickness,
                            colors.HexColor(line_below_color)
                        )
                    )

                if line_above:

                    square_style.append(
                        (
                            'LINEABOVE',
                            (col_idx, line_idx),
                            (col_idx, line_idx),
                            line_above_thickness,
                            colors.HexColor(line_above_color)
                        )
                    )

                

    # ==============================================
    # DEFINING PARAGRAPH STYLE
    # ==============================================

    cell_p_style = ParagraphStyle(
        'CellProbabilityStyle',
        fontName='Helvetica-Bold',
        fontSize=cell_fontsize,
        leading=cell_fontsize + 3,
        textColor=colors.HexColor(box_text_color),
        alignment=1,
        spaceBefore=0,
        spaceAfter=0
    )

    # ==============================================
    # CONVERTING DATA TO STRINGS
    # ==============================================

    table_data_string = []

    for line_idx, line in enumerate(table_data):

        string_line = []

        for col_idx, value in enumerate(line):

            if isinstance(value, str):
                cell_text = value 

            else : 
            
                if show_probabilities:

                    p = np.round(
                        np.asarray(value) * 100,
                        2
                    )

                    cell_text = (
                        f"{p[0]:.2f} {p[1]:.2f}<br/>"
                        f"{p[2]:.2f} {p[3]:.2f}"
                    )

                else:

                    cell_text = str(
                        np.argmax(value)
                    )

            string_line.append(
                Paragraph(cell_text, cell_p_style)
            )
        

        table_data_string.append(string_line)


    # ==============================================
    # CREATING TABLE
    # ==============================================

    column_width = (
        [heading_width] * HEADING_LENGTH
        + [cell_width] * (columns_per_row - HEADING_LENGTH)
    )

    table = Table(
        table_data_string,
        colWidths=column_width,
        rowHeights=[CELL_HEIGHT] * len(table_data)
    )

    table.hAlign = 'LEFT'

    table.setStyle(
        TableStyle(square_style)
    )

    story.append(table)

    return story


# def include_probabilities(
#         story, 
#         probabilities, 
#         box_standard_background,
#         box_success_background, 
#         box_line_color,
#         labels_data,
#         heading_text,
#         box_text_color='#2C3E50',
#         heading_text_color = "#2C3E50",
#         box_line_thickness = 0.5,
#         line_below = False,
#         line_below_thickness = 1.0,
#         line_below_color = '#2C3E50',
#         line_above = False,
#         line_above_thickness = 1.0,
#         line_above_color = '#2C3E50',
#         show_probabilities = False  
# ):


#     # ========================================
#     # OBTAINING TABLE DATA
#     # ========================================
#     table_data = []

#     heading = [heading_text] + [""]*(HEADING_LENGTH-1)
#     first_line = heading + [x for x in probabilities[:COLUMNS_PER_ROW-(HEADING_LENGTH)]]
#     table_data.append(first_line)

#     for i in range(COLUMNS_PER_ROW-(HEADING_LENGTH), len(probabilities), COLUMNS_PER_ROW):
#         line = probabilities[i:i + COLUMNS_PER_ROW]
#         table_data.append(line)

#     # =============================================
#     # DEFINING GENERAL STYLE
#     # =============================================
#     if show_probabilities:
#         heading_width = HEADING_WIDTH_1
#         cell_width = CELL_WIDTH_1
#         heading_fontsize = HEADING_FONTSIZE_1
#         cell_fontsize = CELL_FONTSIZE_1

#     else:
#         heading_width = HEADING_WIDTH_2
#         cell_width = CELL_WIDTH_2
#         heading_fontsize = HEADING_FONTSIZE_1
#         cell_fontsize = HEADING_FONTSIZE_1

#     square_style = [

#         # cells
#         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#         ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
#         ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
#         ('FONTSIZE', (0, 0), (-1, -1), cell_fontsize),
#         ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor(box_text_color)),
#         ('TOPPADDING', (0, 0), (-1, -1), 0),
#         ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
#         ('LEFTPADDING', (0, 0), (-1, -1), 0),
#         ('RIGHTPADDING', (0, 0), (-1, -1), 0),

#         # Heading
#         ('SPAN', (0, 0), (HEADING_LENGTH-1, 0)),
#         ('ALIGN', (0, 0), (0, 0), 'LEFT'),
#         ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),
#         ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
#         ('FONTSIZE', (0, 0), (0, 0), heading_fontsize),
#         ('TEXTCOLOR', (0, 0), (0, 0), colors.HexColor(heading_text_color)),
#     ]


#     # ==============================================
#     # SPECIFIC STYLE FOR EACH CELL
#     # =============================================
#     for line_idx, line in enumerate(table_data):
#         for col_idx, value in enumerate(line):
      

#             if line_idx == 0 and col_idx < HEADING_LENGTH:
#                 continue

#             square_style.extend([
#                 ('BACKGROUND', (col_idx, line_idx), (col_idx, line_idx), colors.HexColor(box_standard_background)),
#                 ('BOX', (col_idx, line_idx), (col_idx, line_idx), box_line_thickness, colors.HexColor(box_line_color)),
#             ])


#             predicted_label = np.argmax(value)
#             true_label = int(labels_data[line_idx][col_idx])

#             if predicted_label == true_label:
#                 square_style.extend([
#                     ('BACKGROUND', (col_idx, line_idx), (col_idx, line_idx), colors.HexColor(box_success_background))
#                 ])

#             if line_below: 
#                 square_style.extend([
#                     ('LINEBELOW', (col_idx, line_idx), (col_idx, line_idx), line_below_thickness, colors.HexColor(line_below_color)),
#                 ])

#             if line_above: 
#                 square_style.extend([
#                     ('LINEABOVE', (col_idx, line_idx), (col_idx, line_idx), line_above_thickness, colors.HexColor(line_above_color)),
#                 ])



#     cell_p_style = ParagraphStyle(
#         'CellProbabilityStyle',
#         fontName='Helvetica-Bold',
#         fontSize=cell_fontsize,
#         leading=cell_fontsize + 3,  # Espaçamento vertical entre as duas linhas de texto interna
#         textColor=colors.HexColor(box_text_color),
#         alignment=1,  # Centralizado horizontalmente
#         bottomPadding=0,
#         topPadding=0
#     )

#     table_data_string = []

#     for line_idx, line in enumerate(table_data):

#         string_line = []

#         for col_idx, value in enumerate(line):

#             if line_idx == 0 and col_idx < HEADING_LENGTH:
#                 string_line.append(value)
#                 continue

#             if show_probabilities:

#                 p = np.round(np.asarray(value) * 100, 2)

#                 cell_text = (
#                     f"[{p[0]:.2f} {p[1]:.2f}<br/>"
#                     f"{p[2]:.2f} {p[3]:.2f}]"
#                 )

#             else:

#                 cell_text = str(np.argmax(value))

#             string_line.append(
#                 Paragraph(cell_text, cell_p_style)
#             )

#     table_data_string.append(string_line) 
#     column_width = [heading_width]*HEADING_LENGTH + [cell_width]*(COLUMNS_PER_ROW - HEADING_LENGTH)
#     table = Table(table_data_string, colWidths=column_width, rowHeights=[CELL_HEIGHT]*len(table_data))
#     table.hAlign = 'LEFT' 
#     table.setStyle(TableStyle(square_style))

#     story.append(table)

#     return story

# ==========================================================
# FUNÇÃO PRINCIPAL
# ==========================================================

def model_results_comparison(
    subjects,
    condition_folder,
    N_FOLDS=N_FOLDS,
    model_states = ["minimum_loss_model", "maximum_accuracy_model"],
    branches = None,
    fusions_one_step = None,
    
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


    for show_probabilities in [False, True]:
            
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

            # BRANCHES
            BRANCHES_SUB_CONDIT_DIR = (
                BRANCHES_STAGE_DIR
                / subject
                / condition_folder.lower()
            )

            # FUSIONS
            FUSIONS_SUB_CONDIT_DIR = (
                FUSIONS_STAGE_DIR
                / subject
                / condition_folder.lower()
                
            )

            # SUB_OUTPUT
            SUB_CONDIT_OUTPUT_DIR = (
                OUTPUT_DIR
                /subject
                /condition_folder.lower()
            )
        
            SUB_CONDIT_OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            # =======================================
            # CREATING DOCUMENT
            # =======================================

            if show_probabilities:

                aux_string = "probabilites"

            else:

                aux_string = "only_labels"

            doc = SimpleDocTemplate(
                f"{SUB_CONDIT_OUTPUT_DIR}/comparison_{subject}_{aux_string}.pdf", 
                pagesize=(A4), 
                title="Comparison",
                leftMargin=10,   # Margem Esquerda
                rightMargin=10,  # Margem Direita
                topMargin=20,    # Opcional: Margem Superior
                bottomMargin=20  # Opcional: Margem Inferior
            )

            story = []

            styles = getSampleStyleSheet() 

            title_style = ParagraphStyle(
                'TitleStyle',
                parent=styles['Heading1'],
                fontSize=16,
                leading=20,
                textColor=colors.HexColor('#2C3E50'),
                spaceAfter=5,
                alignment=1 # Centralized
            )

            story.append(Paragraph(f"<b> {subject.title()} - {" ".join(f"{condition_folder}".split("_")).title()}</b>", title_style))


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


                fold_style = ParagraphStyle(
                    'FoldStyle',
                    parent=styles['Normal'],
                    fontSize=12,
                    leading=16,
                    textColor=colors.HexColor('#2C3E50'),
                    spaceBefore= 5,
                    spaceAfter=5,
                    alignment=0 
                )

                story.append(Paragraph(f"<b>Fold {fold}:</b>", fold_style))

                # ======================================
                # INCLUDING DATA FOR EACH MODEL STATE
                # ======================================
                for model_state in model_states: 

                    model_style = ParagraphStyle(
                        'TextStyle',
                        parent=styles['Normal'],
                        fontSize=12,
                        leading=16,
                        textColor=colors.HexColor('#2C3E50'),
                        spaceBefore= 5,
                        spaceAfter=5,
                        alignment=0 # Left
                    )

                    story.append(Paragraph(f"<b> {" ".join(f"{model_state}".split("_")).title()} State</b>", model_style))

                    # ==========================================================
                    # INCLUDING INDICES, LABELS AND PREDICTIONS FOR VALIDATION
                    # ==========================================================

                    (validation_indices, validation_labels) = checking_indices_labels(
                        "validation", 
                        fold, 
                        model_state,
                        BRANCHES_SUB_CONDIT_DIR, 
                        branches, 
                        FUSIONS_SUB_CONDIT_DIR,
                        fusions_one_step, 
                    )
                    
                    story, _ = include_indices_labels(
                        story, 
                        validation_indices, 
                        "indices:",
                        box_background= "#EBEBEB",
                        box_line_color= "#e6f2ff",
                        show_probabilities= show_probabilities,
                    )

                    story, labels_data = include_indices_labels(
                        story, 
                        validation_labels, 
                        "labels:",
                        box_background= '#e6f2ff',
                        box_line_color= "#d5e8fc",
                        line_below= True,   
                        line_below_thickness=1.0,  
                        show_probabilities=show_probabilities,                
                    )


                    for branch in branches: 

                        validation_probabilities, heading_text = retrive_probabilities_branches(
                            branch,
                            "validation",
                            fold, 
                            model_state, 
                            BRANCHES_SUB_CONDIT_DIR,
                                            
                        )

                        story = include_probabilities(
                            story,
                            validation_probabilities,
                            box_standard_background = '#FFFFFF',
                            box_success_background = '#e6f2ff',
                            box_line_color= '#d5e8fc',
                            labels_data = labels_data,
                            heading_text = heading_text,
                            show_probabilities=show_probabilities
                        )


                    # # ======================================================
                    # # INCLUDING INDICES, LABELS AND PREDICTIONS FOR TEST
                    # # ======================================================

                    # story.append(Spacer(1, 5))

                    # (test_indices, test_labels) = checking_indices_labels(
                    #     "test", 
                    #     fold, 
                    #     model_state,
                    #     BRANCHES_SUB_CONDIT_DIR, 
                    #     branches, 
                    #     FUSIONS_SUB_CONDIT_DIR,
                    #     fusions_one_step, 
                        
                    # )
                    
                    # story, _ = include_indices_labels(
                    #     story, 
                    #     test_indices, 
                    #     "indices:",
                    #     box_background= "#EBEBEB",
                    #     box_line_color= "#C3E6CB",
                    # )

                    # story, labels_data = include_indices_labels(
                    #     story, 
                    #     test_labels, 
                    #     "labels:",
                    #     box_background= '#D4EDDA',
                    #     box_line_color= "#C3E6CB",
                    #     line_below= True,   
                    #     line_below_thickness=1.0,                  
                    # )


                    # for branch in branches: 

                    #     test_predictions, heading_text = retrive_probabilities_branches(
                    #         branch,
                    #         "test",
                    #         fold, 
                    #         model_state, 
                    #         BRANCHES_SUB_CONDIT_DIR,
                                            
                    #     )

                    #     story = include_probabilities(
                    #         story,
                    #         test_predictions,
                    #         box_standard_background = '#FFFFFF',
                    #         box_success_background = '#D4EDDA',
                    #         box_line_color= "#C3E6CB",
                    #         labels_data = labels_data,
                    #         heading_text = heading_text

                    #     )

                    
                    # for fusion_one_step in fusions_one_step:

                    #     test_predictions, heading_text = retrive_predictions_fusions_one_step(
                    #         FUSIONS_SUB_CONDIT_DIR,
                    #         fusion_one_step,
                    #         "test",
                    #         fold, 
                    #         model_state,                    
                    #     )

                    #     story = include_probabilities(
                    #         story,
                    #         test_predictions,
                    #         box_standard_background = '#FFFFFF',
                    #         box_success_background = '#D4EDDA',
                    #         box_line_color= '#C3E6CB',
                    #         labels_data = labels_data,
                    #         heading_text = heading_text,
                    #         line_above=True,                      
                    #         line_below=True,                      
                    #   )

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
            ],
        fusions_one_step = [
            "majority_voting_fusion"
        ],
    )