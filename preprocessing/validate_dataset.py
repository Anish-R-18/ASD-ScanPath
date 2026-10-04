import json
import os
import numpy as np

def validate():
    processed_dir = 'c:/Users/anish/OneDrive/Desktop/gaze-ag/processed_data'
    
    with open(os.path.join(processed_dir, 'asd_scanpaths.json'), 'r') as f:
        asd_data = json.load(f)
    with open(os.path.join(processed_dir, 'td_scanpaths.json'), 'r') as f:
        td_data = json.load(f)

    # 1. Check all 300 images have ASD and TD data
    asd_images = set(sp['image_id'] for sp in asd_data)
    td_images = set(sp['image_id'] for sp in td_data)
    
    assert len(asd_images) == 300, f"ASD images count is {len(asd_images)}, not 300"
    assert len(td_images) == 300, f"TD images count is {len(td_images)}, not 300"
    assert asd_images == set(range(1, 301)), "ASD missing some image IDs 1-300"
    assert td_images == set(range(1, 301)), "TD missing some image IDs 1-300"

    print("PASS: all 300 images have ASD and TD data")

    # Group all for further checks
    all_scanpaths = asd_data + td_data

    for sp in all_scanpaths:
        scanpath = sp['scanpath']
        
        # 2. Check no scanpath has <3 fixations
        assert len(scanpath) >= 3, f"Scanpath with {len(scanpath)} fixations found."
        
        total_duration = 0
        for fix in scanpath:
            x, y, duration = fix
            
            # 3. Check no duration <60 ms (0.06 s)
            # floating point safety
            assert duration >= 0.06 - 1e-6, f"Duration {duration} < 0.06 s"
            
            # 5. Check coordinates within [-1, 1]
            assert -1.0 <= x <= 1.0, f"x coord {x} out of bounds"
            assert -1.0 <= y <= 1.0, f"y coord {y} out of bounds"
            
            total_duration += duration
            
        # 4. Check no cumulative duration >3.0 s
        assert total_duration <= 3.0 + 1e-6, f"Cumulative duration {total_duration} > 3.0 s"

    print("PASS: no scanpath has <3 fixations")
    print("PASS: no duration <60 ms")
    print("PASS: no cumulative duration >3.0 s")
    print("PASS: coordinates are within [-1,1]")
    print("All validations completed successfully!")

if __name__ == '__main__':
    validate()
