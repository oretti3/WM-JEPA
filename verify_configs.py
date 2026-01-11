
import torch
import yaml
import sys
import os
import dataclasses
from pathlib import Path

# Add current directory to path to allow imports
sys.path.append(os.getcwd())

from pldm.models.hjepa import HJEPA, HJEPAConfig
from pldm.models.jepa import JEPA, JEPAConfig
from pldm.models.encoders.enums import BackboneConfig

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# Helper to construct config from YAML dict (Approximate, reusing logic from before or just manual mapping)
# Since we know the fields we changed, let's just manually build the config object matching the YAML content
# to be 100% sure we are testing what we think we are testing.
# BUT, better yet, let's verify that the YAML *content* maps to what we expect.

def run_check():
    print("--- Verifying Created Config Files ---")
    
    # 1. Load L1 2M YAML
    yaml_l1 = "PLDM_hieral/configs/tworooms_l1_2m.yaml"
    with open(yaml_l1, 'r') as f:
        cfg1_dict = yaml.safe_load(f)
    
    print(f"Loaded {yaml_l1}")
    l1_backbone_cfg = cfg1_dict['hjepa']['level1']['backbone']
    l1_pred_cfg = cfg1_dict['hjepa']['level1']['predictor']
    
    print(f"  Backbone FC Output Dim: {l1_backbone_cfg.get('fc_output_dim', 'N/A')}")
    print(f"  Predictor Subclass: {l1_pred_cfg.get('predictor_subclass', 'N/A')}")
    
    # Construct Object
    l1_backbone = BackboneConfig(
        arch="impala",
        backbone_subclass="i",
        backbone_width_factor=l1_backbone_cfg.get('backbone_width_factor', 1),
        channels=2,
        final_ln=True,
        fc_output_dim=l1_backbone_cfg.get('fc_output_dim', 512)
    )
    l1_config = JEPAConfig(
        backbone=l1_backbone,
        action_dim=2
    )
    l1_config.predictor.predictor_arch = "rnnV2"
    l1_config.predictor.predictor_subclass = l1_pred_cfg.get('predictor_subclass', "512-512")
    l1_config.predictor.rnn_layers = 1
    
    hjepa_l1_cfg = HJEPAConfig(level1=l1_config, disable_l2=True)
    model_l1 = HJEPA(hjepa_l1_cfg, input_dim=(2, 65, 65))
    t1, tr1 = count_params(model_l1)
    print(f"  > L1 2M Total Params: {t1:,}")
    
    # 2. Load L2 2M YAML
    yaml_l2 = "PLDM_hieral/configs/tworooms_l2_2m.yaml"
    with open(yaml_l2, 'r') as f:
        cfg2_dict = yaml.safe_load(f)
        
    print(f"\nLoaded {yaml_l2}")
    l2_hjepa_cfg = cfg2_dict['hjepa']
    
    # It should share the same L1 config
    hjepa_l2_cfg = HJEPAConfig(level1=l1_config, disable_l2=False)
    hjepa_l2_cfg.l2_z_dim = 32
    hjepa_l2_cfg.l2_posterior_arch = l2_hjepa_cfg.get('l2_posterior_arch', "512-512")
    hjepa_l2_cfg.l2_decoder_arch = l2_hjepa_cfg.get('l2_decoder_arch', "512-512")
    
    model_l2 = HJEPA(hjepa_l2_cfg, input_dim=(2, 65, 65))
    t2, tr2 = count_params(model_l2)
    print(f"  > L1+L2 2M Total Params: {t2:,}")
    
    if 2000000 <= t2 <= 2300000:
        print("\nSUCCESS: Parameter count is within ~2.0M - 2.3M range.")
    else:
        print(f"\nWARNING: Parameter count {t2:,} is outside target range.")

if __name__ == "__main__":
    run_check()
