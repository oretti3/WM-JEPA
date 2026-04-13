
import torch
import sys
import os
import dataclasses

# Add current directory to path to allow imports
sys.path.append(os.getcwd())

from pldm.models.hjepa import HJEPA, HJEPAConfig
from pldm.models.jepa import JEPA, JEPAConfig
from pldm.models.encoders.enums import BackboneConfig

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def run_check():
    # Base Configs
    l1_backbone = BackboneConfig(
        arch="impala",
        backbone_subclass="i",
        backbone_width_factor=2, # Note: This is ignored by current Impala implementation (defaults to 1)
        channels=2,
        final_ln=True
    )
    l1_config = JEPAConfig(
        backbone=l1_backbone,
        action_dim=2
    )
    l1_config.predictor.predictor_arch = "rnnV2"
    l1_config.predictor.predictor_subclass = "512-512"
    l1_config.predictor.rnn_layers = 1
    
    print("\n--- Proposal 5: Target ~2.2M Total ---")
    # Strategy: Reduce fc_output_dim significantly. Reduce Predictor MLPs.
    
    # L1 Settings
    l1_backbone_opt = dataclasses.replace(l1_backbone)
    l1_backbone_opt.fc_output_dim = 128 # drastically reduce from 512
    l1_backbone_opt.backbone_width_factor = 1 # Just to be explicit, though ignored
    
    l1_config_opt = dataclasses.replace(l1_config)
    l1_config_opt.backbone = l1_backbone_opt
    l1_config_opt.predictor = dataclasses.replace(l1_config.predictor)
    l1_config_opt.predictor.predictor_subclass = "128-128" # Small predictor
    
    # Calculate L1 Only
    hjepa_l1_opt_cfg = HJEPAConfig(level1=l1_config_opt, disable_l2=True)
    model_l1_opt = HJEPA(hjepa_l1_opt_cfg, input_dim=(2, 65, 65))
    t1_opt, tr1_opt = count_params(model_l1_opt)
    print(f"Prop 5 L1 Only (fc_dim=128, pred=128-128): {t1_opt:,}")
    
    # L2 Settings (enable L2)
    hjepa_opt_cfg = HJEPAConfig(level1=l1_config_opt, disable_l2=False)
    hjepa_opt_cfg.l2_z_dim = 16 # Reduce z_dim
    hjepa_opt_cfg.l2_posterior_arch = "128-128" # Small MLPs
    hjepa_opt_cfg.l2_decoder_arch = "128-128"
    
    model_opt = HJEPA(hjepa_opt_cfg, input_dim=(2, 65, 65))
    t_opt, tr_opt = count_params(model_opt)
    print(f"Prop 5 L1+L2 Total: {t_opt:,}")
    print(f"Prop 5 L2 Overhead: {t_opt - t1_opt:,}")

    print("\n--- Proposal 6: Slightly larger (Target 2.2M precise) ---")
    # Maybe 128 is too small? Try 200 or 256 for output_dim?
    # Or keep output_dim 128 and increase predictor?
    
    l1_backbone_opt2 = dataclasses.replace(l1_backbone_opt)
    l1_backbone_opt2.fc_output_dim = 192 
    
    l1_config_opt2 = dataclasses.replace(l1_config_opt)
    l1_config_opt2.backbone = l1_backbone_opt2
    
    hjepa_opt2_cfg = HJEPAConfig(level1=l1_config_opt2, disable_l2=False)
    hjepa_opt2_cfg.l2_z_dim = 32
    hjepa_opt2_cfg.l2_posterior_arch = "128-128"
    hjepa_opt2_cfg.l2_decoder_arch = "128-128"
    
    model_opt2 = HJEPA(hjepa_opt2_cfg, input_dim=(2, 65, 65))
    t_opt2, tr_opt2 = count_params(model_opt2)
    print(f"Prop 6 L1+L2 Total (fc_dim=192): {t_opt2:,}")

    print("\n--- Proposal 7: Target ~2.2M (fc_dim=256, pred=256-256) ---")
    l1_backbone_opt3 = dataclasses.replace(l1_backbone_opt)
    l1_backbone_opt3.fc_output_dim = 256 
    
    l1_config_opt3 = dataclasses.replace(l1_config_opt)
    l1_config_opt3.backbone = l1_backbone_opt3
    l1_config_opt3.predictor.predictor_subclass = "256-256"
    
    hjepa_opt3_cfg = HJEPAConfig(level1=l1_config_opt3, disable_l2=False)
    hjepa_opt3_cfg.l2_z_dim = 32
    hjepa_opt3_cfg.l2_posterior_arch = "256-256"
    hjepa_opt3_cfg.l2_decoder_arch = "256-256"
    
    model_opt3 = HJEPA(hjepa_opt3_cfg, input_dim=(2, 65, 65))
    t_opt3, tr_opt3 = count_params(model_opt3)
    print(f"Prop 7 L1+L2 Total (fc_dim=256, pred=256-256): {t_opt3:,}")

    print("\n--- Proposal 8: Validation (fc_dim=256, pred=128-128) ---")
    # In case Prop 7 is too big (~2.5M?)
    l1_config_opt4 = dataclasses.replace(l1_config_opt3)
    l1_config_opt4.predictor.predictor_subclass = "128-128"
    
    hjepa_opt4_cfg = HJEPAConfig(level1=l1_config_opt4, disable_l2=False)
    hjepa_opt4_cfg.l2_z_dim = 32
    hjepa_opt4_cfg.l2_posterior_arch = "128-128"
    hjepa_opt4_cfg.l2_decoder_arch = "128-128"
    
    model_opt4 = HJEPA(hjepa_opt4_cfg, input_dim=(2, 65, 65))
    t_opt4, tr_opt4 = count_params(model_opt4)
    print(f"Prop 8 L1+L2 Total (fc_dim=256, pred=128-128): {t_opt4:,}")

if __name__ == "__main__":
    run_check()
