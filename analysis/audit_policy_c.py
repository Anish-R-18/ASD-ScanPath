import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

def main():
    root_dir = Path("c:/Users/anish/OneDrive/Desktop/gaze-ag/external_data/Gaze4ASD")
    
    asd_tsvs = glob.glob(str(root_dir / "ASD_ScanPath/**/*.tsv"), recursive=True)
    td_tsvs = glob.glob(str(root_dir / "TD_ScanPath/**/*.tsv"), recursive=True)
    
    img_dir = root_dir / "Image Stimuli"
    dims = {}
    for img_path in glob.glob(str(img_dir / "*.*")):
        name = Path(img_path).name
        dims[name] = Image.open(img_path).size
        
    all_tsvs = [(f, 'ASD') for f in asd_tsvs] + [(f, 'TD') for f in td_tsvs]
    
    fixations = []
    for tsv, group in all_tsvs:
        name = Path(tsv).name
        img_name = name.replace('.tsv', '.png')
        if img_name not in dims: img_name = name.replace('.tsv', '.jpg')
        W, H = dims.get(img_name, (1200, 800))
        try:
            df = pd.read_csv(tsv, sep='\t', header=None, names=['X', 'Y', 'Duration'])
            df['group'] = group
            df['participant'] = Path(tsv).parent.name
            df['stimulus'] = name
            df['W'] = W
            df['H'] = H
            offset_x = (2560 - W) / 2
            offset_y = (1440 - H) / 2
            df['img_x'] = df['X'] - offset_x
            df['img_y'] = df['Y'] - offset_y
            fixations.append(df)
        except Exception:
            pass
            
    full_df = pd.concat(fixations, ignore_index=True)
    
    is_outside_image = (full_df['img_x'] < 0) | (full_df['img_x'] > full_df['W']) | \
                       (full_df['img_y'] < 0) | (full_df['img_y'] > full_df['H'])
    full_df['outside'] = is_outside_image
    
    valid_scanpaths = []
    
    for name, group_df in full_df.groupby(['participant', 'stimulus', 'group', 'W', 'H']):
        part, stim, grp, W, H = name
        
        df_curr = group_df.copy()
        
        if len(df_curr) == 0: continue
        
        # 1. Remove first fixation
        df_curr = df_curr.iloc[1:]
        if len(df_curr) == 0: continue
        
        # 2. Duration >= 60
        df_curr = df_curr[df_curr['Duration'] >= 60.0]
        if len(df_curr) == 0: continue
        
        # 3. Truncate at 3.0s (3000ms)
        starts = [0.0] + list(df_curr['Duration'].cumsum()[:-1])
        df_curr['start'] = starts
        
        df_curr = df_curr[df_curr['start'] < 3000.0]
        if len(df_curr) == 0: continue
        
        last_idx = df_curr.index[-1]
        if df_curr.loc[last_idx, 'start'] + df_curr.loc[last_idx, 'Duration'] > 3000.0:
            new_d = 3000.0 - df_curr.loc[last_idx, 'start']
            if new_d < 60.0:
                df_curr = df_curr.drop(last_idx)
            else:
                df_curr.loc[last_idx, 'Duration'] = new_d
                
        # 4. If ANY retained fixation is outside, reject entire trial
        if df_curr['outside'].any():
            continue
            
        # 5. min_len = 3
        if len(df_curr) < 3:
            continue
            
        df_curr['participant'] = part
        df_curr['stimulus'] = stim
        df_curr['group'] = grp
        valid_scanpaths.append(df_curr)
        
    res_df = pd.concat(valid_scanpaths, ignore_index=True)
    
    # Stats
    part_stats = res_df.groupby(['group', 'participant'])['stimulus'].nunique().reset_index()
    asd_part_stats = part_stats[part_stats['group'] == 'ASD']['stimulus']
    td_part_stats = part_stats[part_stats['group'] == 'TD']['stimulus']
    
    print("--- POLICY C AUDIT ---")
    print(f"ASD retained trials: {asd_part_stats.sum()} / 800 ({asd_part_stats.sum()/800*100:.1f}%)")
    print(f"TD retained trials: {td_part_stats.sum()} / 3733 ({td_part_stats.sum()/3733*100:.1f}%)")
    print(f"ASD participant count: {len(asd_part_stats)}")
    print(f"TD participant count: {len(td_part_stats)}")
    print(f"ASD valid trials per part - Min: {asd_part_stats.min()}, Median: {asd_part_stats.median()}, Mean: {asd_part_stats.mean():.1f}, Max: {asd_part_stats.max()}")
    print(f"TD valid trials per part - Min: {td_part_stats.min()}, Median: {td_part_stats.median()}, Mean: {td_part_stats.mean():.1f}, Max: {td_part_stats.max()}")
    print(f"Unique stimuli represented: {res_df['stimulus'].nunique()}")
    
    asd_durs = res_df[res_df['group'] == 'ASD']['Duration']
    td_durs = res_df[res_df['group'] == 'TD']['Duration']
    print(f"ASD Duration - Median: {asd_durs.median():.1f}, Mean: {asd_durs.mean():.1f}, Min: {asd_durs.min():.1f}, Max: {asd_durs.max():.1f}")
    print(f"TD Duration - Median: {td_durs.median():.1f}, Mean: {td_durs.mean():.1f}, Min: {td_durs.min():.1f}, Max: {td_durs.max():.1f}")
    
    fix_counts = res_df.groupby(['group', 'participant', 'stimulus']).size().reset_index(name='counts')
    asd_fc = fix_counts[fix_counts['group'] == 'ASD']['counts']
    td_fc = fix_counts[fix_counts['group'] == 'TD']['counts']
    print(f"ASD Fixations per scanpath - Median: {asd_fc.median():.1f}, Mean: {asd_fc.mean():.1f}")
    print(f"TD Fixations per scanpath - Median: {td_fc.median():.1f}, Mean: {td_fc.mean():.1f}")
    
if __name__ == '__main__':
    main()
