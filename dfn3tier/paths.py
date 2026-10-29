"""Repository paths. Importing this module sets the environment variables that the core code reads in place of the
absolute paths of the research machine; child processes inherit them."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / 'dfn3tier' / 'core'
STUDY = CORE / 'paper2_paper3_section_study'
EXP = STUDY / 'PAPER2_RETRAIN_AB_2026-10-28'          # final framework: evaluation drivers and analysis
ENGINE = STUDY / 'plan_d_step3_training_evaluation_2026-10-21'   # training job and shared engine
DATA = REPO / 'data'
MODELS = REPO / 'models'
FOLDS = ['S1W_P1', 'S1W_P2', 'S1W_P3', 'S1W_P4', 'S1W_W04', 'S1_EAST', 'S1_WEST', 'S2_LOW', 'S2_HIGH']
WALL_GROUP = dict(zip(FOLDS, [f'G{i}' for i in range(1, 10)]))   # fold code -> wall group G1 to G9
LAYOUT = 'S10o5'   # vertical scanlines 10 m apart, the first 5 m from the left window side

os.environ.setdefault('DFN3_CORE', CORE.as_posix())
os.environ.setdefault('DFN3_DATA', DATA.as_posix())
os.environ.setdefault('DFN3_ADFNE', (DATA / 'adfne_pools').as_posix())
