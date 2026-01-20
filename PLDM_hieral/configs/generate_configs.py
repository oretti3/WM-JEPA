
import yaml
import os
from copy import deepcopy

# Base configs paths
BASE_L1_PATH = "PLDM_hieral/configs/tworooms_l1_6m.yaml"
BASE_L2_PATH = "PLDM_hieral/configs/tworooms_l2.yaml"
OUTPUT_DIR = "PLDM_hieral/configs"

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(config, path):
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def update_horizon(config, horizon):
    # Update max_plan_length in eval_cfg
    config['eval_cfg']['wall_planning']['easy']['max_plan_length'] = horizon
    config['eval_cfg']['wall_planning']['medium']['max_plan_length'] = horizon
    config['eval_cfg']['wall_planning']['level1']['max_plan_length'] = horizon
    
def update_dataset_and_epochs(config, dataset_size_key, epochs):
    # Update dataset path
    # Assuming standard paths based on size key
    if dataset_size_key == '634':
        path = "pldm_envs/wall/presaved_datasets/rendered/ds_size_634.npz"
    elif dataset_size_key == '100k':
        path = "pldm_envs/wall/presaved_datasets/rendered/ds_size_100k.npz" # Placeholder name
    elif dataset_size_key == '1500k':
        path = "pldm_envs/wall/presaved_datasets/rendered/ds_size_1500K.npz"
    
    config['data']['offline_wall_config']['offline_data_path'] = path
    config['epochs'] = epochs

def update_model_size(config, size_type, is_l2):
    if size_type == '2M':
        # L1 settings
        config['hjepa']['level1']['backbone']['backbone_width_factor'] = 1
        config['hjepa']['level1']['backbone']['fc_output_dim'] = 256
        config['hjepa']['level1']['predictor']['predictor_subclass'] = "128-128"
        
        # L2 settings if applicable
        if is_l2:
            config['hjepa']['l2_posterior_arch'] = "128-128"
            config['hjepa']['l2_decoder_arch'] = "128-128"
            
    elif size_type == '6M':
        # L1 settings
        config['hjepa']['level1']['backbone']['backbone_width_factor'] = 2
        config['hjepa']['level1']['backbone']['fc_output_dim'] = 1024
        config['hjepa']['level1']['predictor']['predictor_subclass'] = "512-512"
        
        # L2 settings if applicable
        if is_l2:
            config['hjepa']['l2_posterior_arch'] = "512-512"
            config['hjepa']['l2_decoder_arch'] = "512-512"

def main():
    base_l1 = load_yaml(BASE_L1_PATH)
    base_l2 = load_yaml(BASE_L2_PATH)

    # Common modifications for base templates (e.g. ensure 1500k/1ep defaults if needed, or set explicitly below)
    
    # --- Group 1: Horizon Comparison ---
    # Model: 6M, Data: 1500k, Epochs: 1
    horizons = [5, 24, 48, 96, 192]
    for h in horizons:
        for model_type, base_cfg in [('l1', base_l1), ('l2', base_l2)]:
            cfg = deepcopy(base_cfg)
            
            # Set Base Params (1500k, 1ep, 6M)
            update_dataset_and_epochs(cfg, '1500k', 1)
            update_model_size(cfg, '6M', model_type=='l2')
            
            # Set Horizon
            update_horizon(cfg, h)
            
            # Save
            filename = f"tworooms_{model_type}_6m_h{h}.yaml"
            cfg['output_dir'] = filename.replace('.yaml', '') # Update output dir to match config name
            cfg['run_name'] = filename.replace('.yaml', '')
            
            save_yaml(cfg, os.path.join(OUTPUT_DIR, filename))
            print(f"Created {filename}")

    # --- Group 2: Data Scale Comparison ---
    # Model: 6M, Horizon: 96
    # Variations: 
    # 634 -> 2366 ep
    # 100k -> 15 ep
    # 1500k -> 1 ep (Already covered by h96, but enabling explicit naming)
    
    data_scales = [
        ('634', 2366),
        ('100k', 15),
        ('1500k', 1)
    ]
    
    for d_name, eps in data_scales:
        for model_type, base_cfg in [('l1', base_l1), ('l2', base_l2)]:
            cfg = deepcopy(base_cfg)
            
            # Set Base Params (6M)
            update_model_size(cfg, '6M', model_type=='l2')
            
            # Set Horizon (Fixed 96)
            update_horizon(cfg, 96)
            
            # Set Data/Epochs
            update_dataset_and_epochs(cfg, d_name, eps)
            
            # Save
            filename = f"tworooms_{model_type}_6m_d{d_name}.yaml"
            cfg['output_dir'] = filename.replace('.yaml', '')
            cfg['run_name'] = filename.replace('.yaml', '')
            
            save_yaml(cfg, os.path.join(OUTPUT_DIR, filename))
            print(f"Created {filename}")

    # --- Group 3: Model Size Comparison ---
    # Horizon: 96, Data: 1500k, Epochs: 1
    # 6M is covered by Group 1 (h96)
    # 2M needs creation
    
    for model_type, base_cfg in [('l1', base_l1), ('l2', base_l2)]:
        cfg = deepcopy(base_cfg)
        
        # Set Base Params (1500k, 1ep)
        update_dataset_and_epochs(cfg, '1500k', 1)
        
        # Set Horizon (Fixed 96)
        update_horizon(cfg, 96)
        
        # Set Model Size (2M)
        update_model_size(cfg, '2M', model_type=='l2')
        
        # Save
        filename = f"tworooms_{model_type}_2m_h96.yaml"
        cfg['output_dir'] = filename.replace('.yaml', '')
        cfg['run_name'] = filename.replace('.yaml', '')
        
        save_yaml(cfg, os.path.join(OUTPUT_DIR, filename))
        print(f"Created {filename}")

if __name__ == "__main__":
    main()
