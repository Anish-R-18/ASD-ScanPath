# Inference / Demo Testing

This directory is intended for testing the `inference.py` script. 

To test the model on your own images:
1. Place any valid image (e.g., `my_image.png`) in `test/input/`.
2. Run the inference script from the root repository directory:
   ```bash
   python inference.py --image test/input/my_image.png --condition both --out-dir test/outputs
   ```
3. The model will predict visual scanpaths and save visual renders (with arrows and durations) to `test/outputs/my_image/`.

Sampling uses temperature 1.0 by default. Add `--temperature 2.0` to use the official TPP-Gaze config value. Temperature changes sampling only, not the model or its likelihood.

> **Note:** The `test/input/` images and `test/outputs/` directories are ignored by Git. We strictly avoid uploading any raw dataset images or experimental testing outputs to the GitHub repository to respect licensing and data privacy.
