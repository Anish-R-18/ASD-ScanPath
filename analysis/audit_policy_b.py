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
        if img_name not in dims:
            img_name = name.replace('.tsv', '.jpg')
        
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
        except Exception as e:
            pass
            
    full_df = pd.concat(fixations, ignore_index=True)
    
    is_outside_image = (full_df['img_x'] < 0) | (full_df['img_x'] > full_df['W']) | \
                       (full_df['img_y'] < 0) | (full_df['img_y'] > full_df['H'])
                       
    
    valid_scanpaths = []
    
    for name, group_df in full_df.groupby(['participant', 'stimulus', 'group', 'W', 'H']):
        part, stim, grp, W, H = name
        
        # Policy B: Remove outside image
        df_curr = group_df[~group_df.index.isin(group_df[is_outside_image].index)].copy()
        
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
        
        # Straddle clip
        last_idx = df_curr.index[-1]
        if df_curr.loc[last_idx, 'start'] + df_curr.loc[last_idx, 'Duration'] > 3000.0:
            new_d = 3000.0 - df_curr.loc[last_idx, 'start']
            if new_d < 60.0:
                df_curr = df_curr.drop(last_idx)
            else:
                df_curr.loc[last_idx, 'Duration'] = new_d
                
        # 4. min_len = 3
        if len(df_curr) < 3:
            continue
            
        df_curr['participant'] = part
        df_curr['stimulus'] = stim
        df_curr['group'] = grp
        valid_scanpaths.append(df_curr)
        
    res_df = pd.concat(valid_scanpaths, ignore_index=True)
    
    # Generate statistics
    
    # 1 & 3. Per-participant scanpath counts / valid trials
    part_stats = res_df.groupby(['group', 'participant'])['stimulus'].nunique().reset_index()
    
    print("--- 1 & 3. PER-PARTICIPANT VALID TRIALS ---")
    asd_part_stats = part_stats[part_stats['group'] == 'ASD']['stimulus']
    td_part_stats = part_stats[part_stats['group'] == 'TD']['stimulus']
    
    print(f"ASD valid trials per participant:\n{asd_part_stats.describe()}")
    print(f"TD valid trials per participant:\n{td_part_stats.describe()}")
    
    # 2. Number of participants retained
    print("\n--- 2. PARTICIPANTS RETAINED ---")
    print(f"ASD: {len(asd_part_stats)} / 33")
    print(f"TD: {len(td_part_stats)} / 133")
    
    # 4. Number of static-image trials retained overall
    print("\n--- 4. TOTAL TRIALS RETAINED ---")
    asd_total = asd_part_stats.sum()
    td_total = td_part_stats.sum()
    print(f"ASD: {asd_total} (max possible {33*30})")
    print(f"TD: {td_total} (max possible {133*30})")
    
    # 5. Unique combinations
    combos = res_df.groupby(['participant', 'stimulus']).ngroups
    print("\n--- 5. EXACTLY ONE P+I COMBINATION? ---")
    print(f"Number of scanpaths (groups): {combos}")
    print(f"Total unique Participant + Image combinations in dataset: {len(valid_scanpaths)}")
    print("Matches exactly!" if combos == len(valid_scanpaths) else "WARNING: Mismatch!")
    
    # 6. Duration stats
    print("\n--- 6. DURATION STATISTICS ---")
    asd_durs = res_df[res_df['group'] == 'ASD']['Duration']
    td_durs = res_df[res_df['group'] == 'TD']['Duration']
    print(f"ASD Duration - Median: {asd_durs.median():.1f}, Mean: {asd_durs.mean():.1f}, Min: {asd_durs.min():.1f}, Max: {asd_durs.max():.1f}")
    print(f"TD Duration - Median: {td_durs.median():.1f}, Mean: {td_durs.mean():.1f}, Min: {td_durs.min():.1f}, Max: {td_durs.max():.1f}")
    
    # 7. Fixation-count statistics
    print("\n--- 7. FIXATION COUNT STATISTICS ---")
    fix_counts = res_df.groupby(['group', 'participant', 'stimulus']).size().reset_index(name='counts')
    asd_fc = fix_counts[fix_counts['group'] == 'ASD']['counts']
    td_fc = fix_counts[fix_counts['group'] == 'TD']['counts']
    print(f"ASD Fixations per scanpath - Median: {asd_fc.median():.1f}, Mean: {asd_fc.mean():.1f}")
    print(f"TD Fixations per scanpath - Median: {td_fc.median():.1f}, Mean: {td_fc.mean():.1f}")
    
    # 8 & 9. Checks
    print("\n--- 8 & 9. INTEGRITY CHECKS ---")
    out_of_image = ((res_df['img_x'] < 0) | (res_df['img_x'] > res_df['W']) | (res_df['img_y'] < 0) | (res_df['img_y'] > res_df['H'])).sum()
    print(f"Coordinates outside image: {out_of_image} (Should be 0)")
    
    too_short = (res_df['Duration'] < 60.0).sum()
    print(f"Durations < 60ms: {too_short} (Should be 0)")

if __name__ == '__main__':
    main()
