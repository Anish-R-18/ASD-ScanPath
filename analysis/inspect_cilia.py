import os
import pandas as pd
import glob
from pathlib import Path

def main():
    root_dir = Path("c:/Users/anish/OneDrive/Desktop/gaze-ag/external_data/Eye-Tracking Dataset")
    meta_file = root_dir / "Metadata_Participants.csv"
    data_dir = root_dir / "Eye-tracking Output"
    
    # 1. Metadata
    df_meta = pd.read_csv(meta_file)
    print("--- METADATA ---")
    print(df_meta['Class'].value_counts())
    print(f"Total participants in metadata: {len(df_meta)}")
    
    # 2. File counts
    csv_files = glob.glob(str(data_dir / "*.csv"))
    print(f"\n--- DATA FILES ---")
    print(f"Found {len(csv_files)} CSV files in Eye-tracking Output.")
    
    # 3. Inspect columns
    df_sample = pd.read_csv(csv_files[0], nrows=5, low_memory=False)
    print("\n--- EXACT COLUMNS IN CSV ---")
    print(list(df_sample.columns))
    
    # 4. Check contents
    print("\n--- INSPECTING ALL CSVS FOR UNIQUE VALUES ---")
    all_participants = set()
    all_stimuli = set()
    event_types = set()
    
    # Let's inspect the first 50,000 rows of each to save time, or just distinct categories.
    # We can use pd.read_csv with usecols to make it fast
    usecols = ['Participant', 'Stimulus', 'Category Group', 'Category Right', 'Category Left']
    
    participants_per_stimulus = {}
    
    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=usecols, dtype=str)
            
            ps = df['Participant'].dropna().unique()
            all_participants.update(ps)
            
            ss = df['Stimulus'].dropna().unique()
            all_stimuli.update(ss)
            
            cats = df['Category Right'].dropna().unique()
            event_types.update(cats)
            
            # Check stimulus overlap
            for s in ss:
                parts = df[df['Stimulus'] == s]['Participant'].dropna().unique()
                if s not in participants_per_stimulus:
                    participants_per_stimulus[s] = set()
                participants_per_stimulus[s].update(parts)
                
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    print(f"Unique participants found in data: {len(all_participants)} (IDs: {sorted(list(all_participants))})")
    print(f"Unique stimuli found: {len(all_stimuli)}")
    print(f"Unique event types (Category Right): {event_types}")
    
    shared_stimuli = {k: v for k, v in participants_per_stimulus.items() if len(v) > 1}
    print(f"Stimuli shown to multiple participants: {len(shared_stimuli)} out of {len(all_stimuli)}")
    
    # 5. Check duration and sampling rate
    print("\n--- TIMESTAMP & DURATION CHECK ---")
    df_first = pd.read_csv(csv_files[0], nrows=1000, low_memory=False)
    df_fix = df_first[df_first['Category Right'] == 'Fixation']
    if len(df_fix) > 1:
        diffs = df_fix['RecordingTime [ms]'].astype(float).diff().dropna()
        print(f"Time differences between consecutive Fixation samples in first file (ms):")
        print(diffs.value_counts().head(10))
    else:
        print("Not enough fixation samples in first 1000 rows.")
    
if __name__ == '__main__':
    main()
