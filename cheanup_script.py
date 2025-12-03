import os
import shutil
import glob


def cleanup():
    # 1. Remove cache directories
    print("Removing cache directories...")
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d == "__pycache__" or d.endswith(".egg-info") or d == "build":
                path = os.path.join(root, d)
                print(f"Deleting {path}")
                shutil.rmtree(path, ignore_errors=True)

    # 2. Remove output/experiment folders if you are starting fresh
    # These contain logs and generated images from previous runs
    folders_to_delete = ["experiments", "results", "tb_logger", "wandb"]

    for folder in folders_to_delete:
        if os.path.exists(folder):
            print(f"Deleting {folder}")
            shutil.rmtree(folder, ignore_errors=True)

    # 3. Clean up input folders (Optional: keep if you want to use provided samples)
    # Removing content but keeping directories structure
    input_folders = [
        "inputs/cropped_faces",
        "inputs/gray_faces",
        "inputs/masked_faces",
        "inputs/whole_imgs",
    ]

    print("Cleaning input directories (files only)...")
    for folder in input_folders:
        if os.path.exists(folder):
            # Remove all files in the folder
            files = glob.glob(os.path.join(folder, "*"))
            for f in files:
                if os.path.isfile(f) and not f.endswith(".gitkeep"):
                    print(f"Removing {f}")
                    os.remove(f)

    # 4. Remove unnecessary asset images (documentation images)
    # Keeping 'assets' folder but removing images used only for README
    assets_to_remove = [
        "assets/color_enhancement_result1.png",
        "assets/color_enhancement_result2.png",
        "assets/imgsli_1.jpg",
        "assets/imgsli_2.jpg",
        "assets/imgsli_3.jpg",
        "assets/inpainting_result1.png",
        "assets/inpainting_result2.png",
        "assets/network.jpg",
        "assets/restoration_result1.png",
        "assets/restoration_result2.png",
        "assets/restoration_result3.png",
        "assets/restoration_result4.png",
    ]

    print("Removing documentation assets...")
    for asset in assets_to_remove:
        if os.path.exists(asset):
            print(f"Removing {asset}")
            os.remove(asset)

    print("Cleanup complete.")


if __name__ == "__main__":
    cleanup()
