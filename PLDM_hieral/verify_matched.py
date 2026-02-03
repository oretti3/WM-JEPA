import sys
from pathlib import Path
from omegaconf import OmegaConf
import torch

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pldm.models.hjepa import HJEPA
from pldm.models.hjepa_feedback import HJEPAFeedback

def count_params(module):
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

from pldm.train import TrainConfig

def get_model(config_path):
    # Load default schema
    base_cfg = OmegaConf.structured(TrainConfig)
    # Load yaml
    loaded_cfg = OmegaConf.load(config_path)
    # Merge (loaded overrides base)
    cfg = OmegaConf.merge(base_cfg, loaded_cfg)
    
    # Convert to dataclass instance
    cfg_obj = TrainConfig.parse_from_dict(OmegaConf.to_container(cfg, resolve=True))
    
    # Mock input dimensions: (2, 65, 65) based on yaml
    # channels=2, dim=null in yaml usually implies we can just set it.
    input_dim = (2, 65, 65)
    
    # Instantiate
    hierarchy_type = getattr(cfg_obj.hjepa, "hierarchy_type", "rssm")
    if hierarchy_type == "feedback":
        print(f"Instantiating HJEPAFeedback from {config_path.name}")
        model = HJEPAFeedback(
            cfg_obj.hjepa,
            input_dim=input_dim,
            normalizer=None,
            use_propio_pos=False,
            use_propio_vel=False
        )
    else:
        print(f"Instantiating HJEPA from {config_path.name}")
        model = HJEPA(
            cfg_obj.hjepa,
            input_dim=input_dim,
            normalizer=None,
            use_propio_pos=False,
            use_propio_vel=False
        )
    return model

def main():
    l1_config_path = REPO_ROOT / "PLDM_hieral/configs/tworooms_l1.yaml"
    l1_config_path = REPO_ROOT / "PLDM_hieral/configs/tworooms_l1.yaml"
    l2_config_path = REPO_ROOT / "PLDM_hieral/configs/tworooms_feedback_matched.yaml"

    print("=" * 60)
    print(f"Loading L1 config from: {l1_config_path}")
    model_l1 = get_model(l1_config_path)
    params_l1 = count_params(model_l1)
    print(f"Total Parameters: {params_l1:,}")
    
    print("  Breakdown:")
    print(f"    Level1 (JEPA): {count_params(model_l1.level1):,}")
    print(f"      Backbone: {count_params(model_l1.level1.backbone):,}")
    print(f"      Predictor: {count_params(model_l1.level1.predictor):,}")
    
    if hasattr(model_l1, 'l2_backbone') and model_l1.l2_backbone is not None:
        print(f"    L2 Backbone: {count_params(model_l1.l2_backbone):,}")
    if hasattr(model_l1, 'predictor_l2') and model_l1.predictor_l2 is not None:
        print(f"    L2 Predictor (RSSM): {count_params(model_l1.predictor_l2):,}")

    print("=" * 60)

    print(f"Loading Feedback config from: {l2_config_path}")
    model_l2 = get_model(l2_config_path)
    params_l2 = count_params(model_l2)
    print(f"Total Parameters: {params_l2:,}")

    print("  Breakdown:")
    print(f"    Level1 (JEPA): {count_params(model_l2.level1):,}")
    print(f"      Backbone: {count_params(model_l2.level1.backbone):,}")
    print(f"      Predictor: {count_params(model_l2.level1.predictor):,}")
    
    if hasattr(model_l2, 'l2_backbone') and model_l2.l2_backbone is not None:
        print(f"    L2 Backbone: {count_params(model_l2.l2_backbone):,}")
    if hasattr(model_l2, 'l2_predictor') and model_l2.l2_predictor is not None:
        print(f"    L2 Predictor: {count_params(model_l2.l2_predictor):,}")
    
    if hasattr(model_l2, 'l1_to_l2'):
         print(f"    L1->L2 Projection: {count_params(model_l2.l1_to_l2):,}")
    if hasattr(model_l2, 'l2_to_l1'):
         print(f"    L2->L1 Projection: {count_params(model_l2.l2_to_l1):,}")

    print("=" * 60)
    print("Comparison:")
    print(f"Baseline (L1 only config): {params_l1:,}")
    print(f"Feedback (L1+L2 config):   {params_l2:,}")
    diff = params_l2 - params_l1
    print(f"Difference:              {diff:+,}")

if __name__ == "__main__":
    main()
