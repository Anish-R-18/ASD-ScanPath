import os
import glob
import json
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

def main():
    root_dir = Path("c:/Users/anish/OneDrive/Desktop/gaze-ag/external_data/Gaze4ASD")
    out_dir = Path("c:/Users/anish/OneDrive/Desktop/gaze-ag/external_data/Gaze4ASD_processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    asd_tsvs = glob.glob(str(root_dir / "ASD_ScanPath/**/*.tsv"), recursive=True)
    td_tsvs = glob.glob(str(root_dir / "TD_ScanPath/**/*.tsv"), recursive=True)
    
    img_dir = root_dir / "Image Stimuli"
    dims = {}
    for img_path in glob.glob(str(img_dir / "*.*")):
        name = Path(img_path).name
        dims[name] = Image.open(img_path).size
        
    all_tsvs = [(f, 'ASD') for f in asd_tsvs] + [(f, 'TD') for f in td_tsvs]
    
    for policy in ['A', 'B']:
        out_data = []
        stats = {
            "participants_ASD": set(),
            "participants_TD": set(),
            "trials_ASD": 0,
            "trials_TD": 0,
            "stimuli": set(),
            "fixations": 0,
            "durations": [],
            "removed_first_fixation": 0,
            "removed_duration_too_short": 0,
            "removed_out_of_bounds": 0, # Only for B
            "truncated_end": 0,
            "removed_too_few_fixations": 0,
            "total_initial_scanpaths": len(all_tsvs),
            "max_x": -float('inf'), "min_x": float('inf'),
            "max_y": -float('inf'), "min_y": float('inf')
        }
        
        for tsv, group in all_tsvs:
            name = Path(tsv).name
            img_name = name.replace('.tsv', '.png')
            if img_name not in dims: img_name = name.replace('.tsv', '.jpg')
            W, H = dims.get(img_name, (1200, 800))
            
            try:
                df = pd.read_csv(tsv, sep='\t', header=None, names=['X', 'Y', 'Duration'])
            except Exception:
                continue
                
            offset_x = (2560 - W) / 2
            offset_y = (1440 - H) / 2
            df['img_x'] = df['X'] - offset_x
            df['img_y'] = df['Y'] - offset_y
            
            if policy == 'B':
                is_out = (df['img_x'] < 0) | (df['img_x'] > W) | (df['img_y'] < 0) | (df['img_y'] > H)
                stats["removed_out_of_bounds"] += is_out.sum()
                df = df[~is_out].copy()
            
            if len(df) == 0:
                stats["removed_too_few_fixations"] += 1
                continue
                
            # 1. Remove first fixation
            df = df.iloc[1:]
            stats["removed_first_fixation"] += 1
            if len(df) == 0:
                stats["removed_too_few_fixations"] += 1
                continue
                
            # 2. Duration >= 60
            too_short = df['Duration'] < 60.0
            stats["removed_duration_too_short"] += too_short.sum()
            df = df[~too_short]
            if len(df) == 0:
                stats["removed_too_few_fixations"] += 1
                continue
                
            # 3, 4, 5. Normalize and clip
            df['xn'] = np.clip((df['img_x'] / W) * 2.0 - 1.0, -1.0, 1.0)
            df['yn'] = np.clip((df['img_y'] / H) * 2.0 - 1.0, -1.0, 1.0)
            
            # 6, 7. Time and Cutoff
            df['Duration'] = df['Duration'] / 1000.0 # seconds
            starts = [0.0] + list(df['Duration'].cumsum()[:-1])
            df['start'] = starts
            
            trunc_mask = df['start'] >= 3.0
            stats["truncated_end"] += trunc_mask.sum()
            df = df[~trunc_mask].copy()
            if len(df) == 0:
                stats["removed_too_few_fixations"] += 1
                continue
                
            last_idx = df.index[-1]
            if df.loc[last_idx, 'start'] + df.loc[last_idx, 'Duration'] > 3.0:
                new_d = 3.0 - df.loc[last_idx, 'start']
                if new_d < 0.06:
                    df = df.drop(last_idx)
                    stats["truncated_end"] += 1
                else:
                    df.loc[last_idx, 'Duration'] = new_d
                    
            # 8. Min 3 fixations
            if len(df) < 3:
                stats["removed_too_few_fixations"] += 1
                continue
                
            participant = Path(tsv).parent.name
            if group == 'ASD':
                stats["participants_ASD"].add(participant)
                stats["trials_ASD"] += 1
            else:
                stats["participants_TD"].add(participant)
                stats["trials_TD"] += 1
                
            stats["stimuli"].add(img_name)
            stats["fixations"] += len(df)
            stats["durations"].extend(df['Duration'].values)
            
            stats["min_x"] = min(stats["min_x"], df['xn'].min())
            stats["max_x"] = max(stats["max_x"], df['xn'].max())
            stats["min_y"] = min(stats["min_y"], df['yn'].min())
            stats["max_y"] = max(stats["max_y"], df['yn'].max())
            
            # Format exactly as Saliency4ASD TPP-Gaze expects
            # [(x, y, duration), ...]
            scanpath = []
            for _, r in df.iterrows():
                scanpath.append([round(r['xn'], 6), round(r['yn'], 6), round(r['Duration'], 6)])
                
            out_data.append({
                "image_id": img_name,
                "group": group,
                "participant": participant,
                "scanpath": scanpath
            })
            
        with open(out_dir / f"gaze4asd_policy_{policy.lower()}.json", 'w') as f:
            json.dump(out_data, f)
            
        stats["participants_ASD"] = len(stats["participants_ASD"])
        stats["participants_TD"] = len(stats["participants_TD"])
        stats["stimuli"] = len(stats["stimuli"])
        stats["duration_median"] = np.median(stats["durations"])
        stats["duration_mean"] = np.mean(stats["durations"])
        del stats["durations"]
        
        for k, v in stats.items():
            if hasattr(v, 'item'):
                stats[k] = v.item()
                
        with open(out_dir / f"validation_report_policy_{policy.lower()}.json", 'w') as f:
            json.dump(stats, f, indent=4)
            
        print(f"\nPolicy {policy} Processing Complete!")
        for k, v in stats.items():
            print(f"  {k}: {v}")

if __name__ == '__main__':
    main()
