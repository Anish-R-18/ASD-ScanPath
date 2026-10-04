import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def get_consecutive_blocks(series):
    # series is a boolean mask of where Category Right == 'Fixation'
    return (series != series.shift()).cumsum()[series]

def main():
    root_dir = Path("c:/Users/anish/OneDrive/Desktop/gaze-ag/external_data/Eye-Tracking Dataset")
    data_dir = root_dir / "Eye-tracking Output"
    csv_files = glob.glob(str(data_dir / "*.csv"))
    
    all_stimuli = set()
    usecols = ['Participant', 'Stimulus', 'Category Right', 'RecordingTime [ms]', 'Point of Regard Right X [px]', 'Point of Regard Right Y [px]']
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=usecols, dtype={'Participant': str, 'Stimulus': str, 'Category Right': str})
            dfs.append(df)
            all_stimuli.update(df['Stimulus'].dropna().unique())
        except Exception as e:
            # Handle missing Category Right, maybe usecols issue if different names, but we assume it's mostly there
            print(f"Failed on {f}: {e}")
            
    print(f"Total CSVs successfully loaded: {len(dfs)}")
    full_df = pd.concat(dfs, ignore_index=True)
    
    # 1. & 2. Stimulus types
    static_stim = [s for s in all_stimuli if s.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    video_stim = [s for s in all_stimuli if s.lower().endswith(('.avi', '.mp4', '.mov'))]
    ambiguous_stim = [s for s in all_stimuli if not s.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.avi', '.mp4', '.mov'))]
    
    print("\n--- STIMULUS CATEGORIZATION ---")
    print(f"Total unique stimuli: {len(all_stimuli)}")
    print(f"Static images: {len(static_stim)}")
    print(f"Video stimuli: {len(video_stim)}")
    print(f"Ambiguous: {len(ambiguous_stim)}")
    if ambiguous_stim:
        print(f"Ambiguous names: {ambiguous_stim[:10]}...")
        
    # 3. Static image details
    print("\n--- STATIC IMAGES DETAILS ---")
    print(f"Static image filenames: {static_stim[:10]}...")
    
    # Check if files exist in the dataset folder
    image_files_found = glob.glob(str(root_dir / "**" / "*.png"), recursive=True) + glob.glob(str(root_dir / "**" / "*.jpg"), recursive=True)
    print(f"Actual image files found in dataset directory: {len(image_files_found)}")
    if len(image_files_found) > 0:
        print(f"Sample paths: {image_files_found[:3]}")
        
    # 4. Gaze timestamps distribution
    print("\n--- TIMESTAMP DISTRIBUTION ---")
    # Take a sample or compute on whole dataset
    # Compute diffs grouped by Participant and Stimulus (so we don't diff across trials)
    # We will sample 10% to speed up if too large, but let's try full
    full_df['RecordingTime [ms]'] = pd.to_numeric(full_df['RecordingTime [ms]'], errors='coerce')
    full_df.dropna(subset=['RecordingTime [ms]', 'Participant', 'Stimulus'], inplace=True)
    
    full_df.sort_values(by=['Participant', 'Stimulus', 'RecordingTime [ms]'], inplace=True)
    
    # Diff within each trial
    full_df['time_diff'] = full_df.groupby(['Participant', 'Stimulus'])['RecordingTime [ms]'].diff()
    
    diffs = full_df['time_diff'].dropna()
    print(f"Diff count: {len(diffs)}")
    print(f"Mean diff: {diffs.mean():.3f} ms")
    print(f"Median diff: {diffs.median():.3f} ms")
    print(f"Min diff: {diffs.min():.3f} ms")
    print(f"Max diff: {diffs.max():.3f} ms")
    print(f"Most common diffs:\n{diffs.value_counts().head(5)}")
    
    # 5. & 6. Fixation blocks
    print("\n--- FIXATION BLOCKS ---")
    fix_mask = full_df['Category Right'] == 'Fixation'
    fix_df = full_df[fix_mask].copy()
    
    # Group by consecutive true values in mask per trial
    # Better way: in full_df, mark block changes
    full_df['is_fix'] = full_df['Category Right'] == 'Fixation'
    # Block ID: increments when 'is_fix' changes or when trial changes
    full_df['block_id'] = (full_df['is_fix'] != full_df['is_fix'].shift()).cumsum()
    # Also separate by trial
    full_df['trial_id'] = full_df.groupby(['Participant', 'Stimulus']).ngroup()
    full_df['fix_block'] = full_df.groupby('trial_id')['block_id'].transform(lambda x: x - x.min())
    
    fix_only = full_df[full_df['is_fix']]
    block_stats = fix_only.groupby(['trial_id', 'block_id']).agg(
        num_samples=('RecordingTime [ms]', 'count'),
        start_time=('RecordingTime [ms]', 'min'),
        end_time=('RecordingTime [ms]', 'max')
    )
    
    block_stats['duration'] = block_stats['end_time'] - block_stats['start_time']
    
    print(f"Total fixation blocks: {len(block_stats)}")
    print(f"Blocks with 1 sample: {sum(block_stats['num_samples'] == 1)}")
    print(f"Blocks with 2+ samples: {sum(block_stats['num_samples'] > 1)}")
    
    print(f"Mean block duration: {block_stats['duration'].mean():.2f} ms")
    print(f"Median block duration: {block_stats['duration'].median():.2f} ms")
    print(f"Max block duration: {block_stats['duration'].max():.2f} ms")
    
    # 7. Valid static participants/stimuli
    static_df = full_df[full_df['Stimulus'].isin(static_stim)]
    static_parts = static_df['Participant'].nunique()
    static_stims = static_df['Stimulus'].nunique()
    print("\n--- FILTERED STATIC SET ---")
    print(f"Participants remaining with static stimuli: {static_parts}")
    print(f"Static stimuli remaining: {static_stims}")
    
if __name__ == '__main__':
    main()
