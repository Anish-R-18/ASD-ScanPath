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
    
    # Store dimensions
    dims = {}
    for img_path in glob.glob(str(img_dir / "*.*")):
        name = Path(img_path).name
        dims[name] = Image.open(img_path).size
        
    all_tsvs = [(f, 'ASD') for f in asd_tsvs] + [(f, 'TD') for f in td_tsvs]
    
    fixations = []
    
    for tsv, group in all_tsvs:
        name = Path(tsv).name
        # Match .tsv to .png/.jpg
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
            
            # Map to image
            offset_x = (2560 - W) / 2
            offset_y = (1440 - H) / 2
            df['img_x'] = df['X'] - offset_x
            df['img_y'] = df['Y'] - offset_y
            
            fixations.append(df)
        except Exception as e:
            pass
            
    full_df = pd.concat(fixations, ignore_index=True)
    
    total_fix = len(full_df)
    
    # Out of image
    is_outside_image = (full_df['img_x'] < 0) | (full_df['img_x'] > full_df['W']) | \
                       (full_df['img_y'] < 0) | (full_df['img_y'] > full_df['H'])
                       
    is_outside_screen = (full_df['X'] < 0) | (full_df['X'] > 2560) | \
                        (full_df['Y'] < 0) | (full_df['Y'] > 1440)
                        
    pct_inside = 100 * (~is_outside_image).sum() / total_fix
    pct_outside_img = 100 * is_outside_image.sum() / total_fix
    pct_outside_scr = 100 * is_outside_screen.sum() / total_fix
    
    print("--- 2. OUT-OF-IMAGE GAZE ---")
    print(f"Total fixations: {total_fix}")
    print(f"Inside image: {pct_inside:.2f}%")
    print(f"Outside image: {pct_outside_img:.2f}%")
    print(f"Physically outside 2560x1440 screen: {pct_outside_scr:.2f}%")
    
    full_df['is_valid'] = ~is_outside_image
    scanpaths = full_df.groupby(['participant', 'stimulus', 'group']).agg(
        total_fix=('is_valid', 'count'),
        valid_fix=('is_valid', 'sum')
    ).reset_index()
    
    scanpaths['frac_outside'] = 1.0 - (scanpaths['valid_fix'] / scanpaths['total_fix'])
    
    valid_3 = (scanpaths['valid_fix'] >= 3).sum()
    invalid_3 = (scanpaths['valid_fix'] < 3).sum()
    
    print(f"\nScanpaths with >= 3 valid in-image fixations: {valid_3}")
    print(f"Scanpaths with < 3 valid in-image fixations: {invalid_3}")
    print(f"\nFraction of scanpath outside image (distribution):")
    print(scanpaths['frac_outside'].describe())
    
    # --- 3. COMPARE POLICIES ---
    print("\n--- 3. COMPARE POLICIES ---")
    
    # A. CLIP
    # B. REMOVE
    
    # Define TPP-Gaze preprocessing params exactly like Saliency4ASD Variant B
    # First fixation removal, duration >= 60, truncate cumulative at 3.0, min_len=3
    
    def process(policy):
        valid_sp = 0
        total_fix_after = 0
        durations = []
        parts = set()
        asd_count = 0
        td_count = 0
        
        for name, group_df in full_df.groupby(['participant', 'stimulus', 'group', 'W', 'H']):
            part, stim, grp, W, H = name
            
            # Policy rules
            if policy == 'remove':
                # Remove outside image
                df_curr = group_df[~group_df.index.isin(group_df[is_outside_image].index)].copy()
            else:
                # Clip to image boundaries (norm_x in [-1, 1] means img_x in [0, W])
                df_curr = group_df.copy()
                df_curr['img_x'] = df_curr['img_x'].clip(0, W)
                df_curr['img_y'] = df_curr['img_y'].clip(0, H)
                
            if len(df_curr) == 0: continue
            
            # 1. Remove first fixation
            df_curr = df_curr.iloc[1:]
            if len(df_curr) == 0: continue
            
            # 2. Duration >= 60
            df_curr = df_curr[df_curr['Duration'] >= 60.0]
            if len(df_curr) == 0: continue
            
            # 3. Truncate at 3.0s (3000ms)
            # Need to compute cumulative from durations? 
            # In Gaze4ASD scanpaths, we just have durations.
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
                    
            if len(df_curr) < 3:
                continue
                
            # valid scanpath
            valid_sp += 1
            total_fix_after += len(df_curr)
            durations.extend(df_curr['Duration'].values)
            parts.add(part)
            if grp == 'ASD': asd_count += 1
            else: td_count += 1
            
        print(f"\nPOLICY: {policy.upper()}")
        print(f"Scanpath count: {valid_sp}")
        print(f"ASD scanpaths: {asd_count}, TD scanpaths: {td_count}")
        print(f"Fixation count: {total_fix_after}")
        print(f"Participants: {len(parts)}")
        if durations:
            print(f"Median fix duration: {np.median(durations):.1f} ms")
            print(f"Mean fix duration: {np.mean(durations):.1f} ms")

    process('clip')
    process('remove')

if __name__ == '__main__':
    main()
