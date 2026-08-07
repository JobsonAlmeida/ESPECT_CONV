import numpy as np
from pathlib import Path
from epoch_3d_rcf_t import show_rcf_at_time
from epoch_3d_rcf_t import show_rcf_through_time

current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[2]

INPUT_DIR = PROJECT_ROOT / "processed_data" / "stage_30_array_assembly"
OUTPUT_DIR = PROJECT_ROOT / "processed_data" / current_file.parents[0].name

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]


#show_rcf_at_time('sub-01', 'ses-01', 0, 0)


show_rcf_through_time('sub-01', 'ses-01', 0)
