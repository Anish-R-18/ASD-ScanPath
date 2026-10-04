import os
import glob
import pandas as pd
from PIL import Image
from pathlib import Path

def main():
    root_dir = Path("c:/Users/anish/OneDrive/Desktop/gaze-ag/external_data/Gaze4ASD")
    if not root_dir.exists():
        print(f"Directory {root_dir} does not exist yet. Waiting for clone.")
        return
        
    img_dir = root_dir / "Image Stimuli"
    asd_raw_dir = root_dir / "ASD_rawData"
    td_raw_dir = root_dir / "TD_rawData"
    asd_scan_dir = root_dir / "ASD_ScanPath"
    td_scan_dir = root_dir / "TD_ScanPath"
    
    # 1. Images
    img_files = glob.glob(str(img_dir / "*.*"))
    print("\n--- IMAGES ---")
    print(f"Found {len(img_files)} images.")
    if img_files:
        sample_img = Image.open(img_files[0])
        print(f"Sample image: {os.path.basename(img_files[0])}, dimensions: {sample_img.size}")
        
    # 2 & 3. File counts and 4. formats
    asd_raw = glob.glob(str(asd_raw_dir / "**/*.*"), recursive=True)
    td_raw = glob.glob(str(td_raw_dir / "**/*.*"), recursive=True)
    asd_scan = glob.glob(str(asd_scan_dir / "**/*.*"), recursive=True)
    td_scan = glob.glob(str(td_scan_dir / "**/*.*"), recursive=True)
    
    print("\n--- FILE COUNTS ---")
    print(f"ASD raw: {len(asd_raw)} files, format: {Path(asd_raw[0]).suffix if asd_raw else 'N/A'}")
    print(f"TD raw: {len(td_raw)} files, format: {Path(td_raw[0]).suffix if td_raw else 'N/A'}")
    print(f"ASD scanpath: {len(asd_scan)} files, format: {Path(asd_scan[0]).suffix if asd_scan else 'N/A'}")
    print(f"TD scanpath: {len(td_scan)} files, format: {Path(td_scan[0]).suffix if td_scan else 'N/A'}")
    
    # 5. Exact columns in raw
    print("\n--- RAW DATA COLUMNS ---")
    if asd_raw:
        try:
            df_raw = pd.read_csv(asd_raw[0], sep=None, engine='python', nrows=5)
            print(f"Sample raw file: {os.path.basename(asd_raw[0])}")
            print(list(df_raw.columns))
            print("Row 0:")
            print(df_raw.iloc[0].to_dict())
        except Exception as e:
            print("Failed to read raw file:", e)
            
    # 6 & 7. Exact columns in scanpath
    print("\n--- SCANPATH DATA COLUMNS ---")
    if asd_scan:
        try:
            df_scan = pd.read_csv(asd_scan[0], sep=None, engine='python', nrows=5)
            print(f"Sample scanpath file: {os.path.basename(asd_scan[0])}")
            print(list(df_scan.columns))
            print("Row 0:")
            print(df_scan.iloc[0].to_dict())
        except Exception as e:
            print("Failed to read scanpath file:", e)
            
    # 8. Participant IDs
    print("\n--- PARTICIPANT IDs AND MULTIPLE VIEWERS ---")
    # Usually filenames like imgName_ParticipantID.csv or similar
    if asd_scan:
        print(f"Sample ASD filenames: {[os.path.basename(x) for x in asd_scan[:5]]}")
        print(f"Sample TD filenames: {[os.path.basename(x) for x in td_scan[:5]]}")

if __name__ == '__main__':
    main()
