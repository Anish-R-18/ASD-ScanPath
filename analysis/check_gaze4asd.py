import glob
from pathlib import Path
from PIL import Image

def main():
    root_dir = Path("c:/Users/anish/OneDrive/Desktop/gaze-ag/external_data/Gaze4ASD")
    
    img_files = glob.glob(str(root_dir / "Image Stimuli/*.*"))
    sizes = set()
    for f in img_files:
        sizes.add(Image.open(f).size)
    print(f"Image sizes: {sizes}")
    
    asd_scan = glob.glob(str(root_dir / "ASD_ScanPath/**/*.tsv"), recursive=True)
    td_scan = glob.glob(str(root_dir / "TD_ScanPath/**/*.tsv"), recursive=True)
    
    asd_parts = set([Path(p).parent.name for p in asd_scan])
    td_parts = set([Path(p).parent.name for p in td_scan])
    print(f"Unique ASD participants: {len(asd_parts)}")
    print(f"Unique TD participants: {len(td_parts)}")
    
    asd_imgs = set([Path(p).name for p in asd_scan])
    td_imgs = set([Path(p).name for p in td_scan])
    print(f"Unique images in ASD scanpaths: {len(asd_imgs)}")
    print(f"Unique images in TD scanpaths: {len(td_imgs)}")
    
    print("Sample missing checks:")
    # Expected: 33 ASD * 30 imgs = 990. We have 800 ASD scanpaths (190 missing)
    # Expected: 132 TD * 30 imgs = 3960. We have 3733 TD scanpaths (227 missing)
    print(f"ASD expected 990, actual {len(asd_scan)}")
    print(f"TD expected 3960, actual {len(td_scan)}")

if __name__ == '__main__':
    main()
