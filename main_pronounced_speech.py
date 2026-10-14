# %% 
# [Imports]

import sys
from pathlib import Path

projeto_raiz = Path("~/Documentos/ESPECT_CONV").expanduser().resolve()

if str(projeto_raiz) not in sys.path:
    sys.path.append(str(projeto_raiz))

# ==========================================================
# CONFIGURAÇÕES DE CAMINHO & IMPORTS LOCAIS
# ==========================================================

from pipeline.stage_20_load_and_epoching.stage_20_load_and_epoching import load_and_epoching

from pipeline.stage_30_psd_array_assembly.channels_2D_distribution import CENTRALIZED_ONE_BY_ONE_2D_DISTRIBUTION
from pipeline.stage_30_psd_array_assembly.stage_30_psd_array_assembly import psd_array_assembly
from pipeline.stage_40_obtain_indices.obtain_indices import obtain_indices
from pipeline.stage_50_execute_branch.execute_branch import execute_branch
from pipeline.stage_60_execute_fusion.execute_fusion import execute_fusion

# %%
# [Condition and Subjets]
# ======================================================
# CONDITION OPTIONS
# ======================================================
#
# "PRONOUNCED_SPEECH"
#
# "INNER_SPEECH"
#
# "VISUALIZED_CONDITION"
#
# ======================================================

CONDITION = "PRONOUNCED_SPEECH"

# ==========================================
# SUBJECTS
# ==========================================
subjects = [
    f"sub-{i:02d}"
    for i in range(1, 11)
]

# %%
# [Stage 20: Load and Epoching]
# =============================================
# Stage 20: Load and Epoching
# =============================================

EPOCH_START_SECONDS = 1.0
EPOCH_END_SECONDS = 3.5

load_and_epoching(
        condition=CONDITION,
        crop_start_time=EPOCH_START_SECONDS,
        crop_end_time=EPOCH_END_SECONDS
)

# %%
# [Stage 30: PSD Array Assembly]
# ===========================================
# Stage 30: PSD Array Assembly
# ===========================================

WINDOW_DURATION = 0.5
STEP_DURATION = 0.25

psd_array_assembly(
        window_duration = WINDOW_DURATION,
        step_duration = STEP_DURATION,
        channels_distribution = CENTRALIZED_ONE_BY_ONE_2D_DISTRIBUTION,
        condition_folder = "PRONOUNCED_SPEECH",
        subjects = subjects
)

# CENTRALIZED_ONE_BY_ONE_2D_DISTRIBUTION.plot_channels_distribution()

# %%



