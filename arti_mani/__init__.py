__version__ = "0.1.0"
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
REPO_DIR = Path(__file__).parent.parent.resolve()
ASSET_DIR = ROOT_DIR / "assets"
AGENT_CONFIG_DIR = ASSET_DIR / "configs"
DESCRIPTION_DIR = ASSET_DIR / "descriptions"
ALG_DIR = ROOT_DIR / "algorithms"
# with DR: arti_data_seg, arti_data_seg_randombg, artidata384_seg_DR_randombg_mixfaucet
# w/o DR: arti_data_seg_noDR, arti_data_seg_noDR_randombg
SEGDATA_DIR = ROOT_DIR / "algorithms/data_process/artidata384_seg_DR_randombg"
KPTDATA_DIR = ROOT_DIR / "algorithms/data_process/artidata384_keypoints_noDR_norandombg"
REAL_DIR = ROOT_DIR / "algorithms/data_process/real_capdata_doordrawerfaucet_360_640"
REALEXP_DIR = ROOT_DIR / "test/real_exps"
VISUALMODEL_DIR = ROOT_DIR / "algorithms/visual_net/scripts/log"
RLMODEL_DIR = ROOT_DIR / "algorithms/rl_iam/sac/logs"
