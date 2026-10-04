import os
import glob
import scipy.io as sio
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

def main():
    root_dir = Path("c:/Users/anish/OneDrive/Desktop/gaze-ag/external_data/Gaze4ASD")
    asd_raw_files = glob.glob(str(root_dir / "ASD_rawData/**/*.mat"), recursive=True)
    
    print("--- MAT FILE INSPECTION ---")
    if not asd_raw_files:
        print("No raw MAT files found!")
        return
        
    mat_file = asd_raw_files[0]
    print(f"Loading a .mat file...")
    mat = sio.loadmat(mat_file, simplify_cells=True)
    
    # Check geometry / settings / expt
    geom = mat.get('geometry', {})
    settings = mat.get('settings', {})
    expt = mat.get('expt', {})
    
    print("\nGeometry keys:", list(geom.keys()) if isinstance(geom, dict) else geom)
    print("Settings keys:", list(settings.keys()) if isinstance(settings, dict) else settings)
    print("Expt keys:", list(expt.keys()) if isinstance(expt, dict) else expt)
    
    if isinstance(geom, dict):
        for k, v in geom.items():
            print(f"geometry.{k} = {v}")
    if isinstance(settings, dict):
        for k, v in settings.items():
            print(f"settings.{k} = {v}")
            
    print("\n--- SAMPLE SCANPATHS vs MAT ---")
    # Let's take a few TSV files and audit them.
    tsv_files = glob.glob(str(root_dir / "ASD_ScanPath/**/*.tsv"), recursive=True)
    
    out_of_bounds_count = 0
    total_fixations = 0
    negative_count = 0
    
    img_dir = root_dir / "Image Stimuli"
    
    for tsv_file in tsv_files[:5]:
        img_name = Path(tsv_file).name.replace('.tsv', '.png')
        img_path = img_dir / img_name
        if not img_path.exists():
            img_name = Path(tsv_file).name.replace('.tsv', '.jpg')
            img_path = img_dir / img_name
            
        W, H = (1200, 800) # Fallback
        if img_path.exists():
            W, H = Image.open(img_path).size
            
        print(f"\nEvaluating {Path(tsv_file).parent.name}/{Path(tsv_file).name} (Image: {W}x{H})")
        df = pd.read_csv(tsv_file, sep='\t', header=None, names=['X', 'Y', 'Duration'])
        
        # Calculate centered mapping assuming 2560x1440
        # If centered, top-left of image on screen is at X = (2560-W)/2, Y = (1440-H)/2
        offset_x = (2560 - W) / 2
        offset_y = (1440 - H) / 2
        
        df['img_x'] = df['X'] - offset_x
        df['img_y'] = df['Y'] - offset_y
        
        print(df.head(5).to_string())
        
        for _, row in df.iterrows():
            total_fixations += 1
            if row['X'] < 0 or row['X'] > 2560 or row['Y'] < 0 or row['Y'] > 1440:
                out_of_bounds_count += 1
            if row['img_x'] < 0 or row['img_x'] > W or row['img_y'] < 0 or row['img_y'] > H:
                negative_count += 1
                
    # Audit across ALL files
    out_of_bounds_count = 0
    total_fixations = 0
    negative_count = 0
    for tsv_file in tsv_files:
        img_name = Path(tsv_file).name.replace('.tsv', '.png')
        img_path = img_dir / img_name
        W, H = 1200, 800
        if img_path.exists():
            W, H = Image.open(img_path).size
            
        try:
            df = pd.read_csv(tsv_file, sep='\t', header=None, names=['X', 'Y', 'Duration'])
            offset_x = (2560 - W) / 2
            offset_y = (1440 - H) / 2
            img_x = df['X'] - offset_x
            img_y = df['Y'] - offset_y
            
            total_fixations += len(df)
            out_of_bounds_count += sum((df['X'] < 0) | (df['X'] > 2560) | (df['Y'] < 0) | (df['Y'] > 1440))
            negative_count += sum((img_x < 0) | (img_x > W) | (img_y < 0) | (img_y > H))
        except Exception as e:
            pass
            
    print(f"\n--- AUDIT OF ALL {len(tsv_files)} ASD TSVs ---")
    print(f"Total fixations: {total_fixations}")
    print(f"Screen Out of bounds (<0 or >2560/1440): {out_of_bounds_count}")
    print(f"Image Out of bounds (after center mapping): {negative_count}")

if __name__ == '__main__':
    main()
