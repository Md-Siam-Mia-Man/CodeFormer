import argparse
import os
from os import path as osp
import sys

# Add root directory to sys.path to find patches.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from patches import apply_torchvision_patches

    apply_torchvision_patches()
except ImportError:
    pass


import gdown


def download_pretrained_models(method, file_ids):
    save_path_root = f"./weights/{method}"
    os.makedirs(save_path_root, exist_ok=True)

    for file_name, file_id in file_ids.items():
        file_url = "https://drive.google.com/uc?id=" + file_id
        save_path = osp.abspath(osp.join(save_path_root, file_name))
        if osp.exists(save_path):
            user_response = input(
                f"{file_name} already exist. Do you want to cover it? Y/N\n"
            )
            if user_response.lower() == "y":
                print(f"Covering {file_name} to {save_path}")
                gdown.download(file_url, save_path, quiet=False)
            elif user_response.lower() == "n":
                print(f"Skipping {file_name}")
            else:
                raise ValueError("Wrong input. Only accepts Y/N.")
        else:
            print(f"Downloading {file_name} to {save_path}")
            gdown.download(file_url, save_path, quiet=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "method",
        type=str,
        help=(
            "Options: 'CodeFormer' 'facelib'. Set to 'all' to download all the models."
        ),
    )
    args = parser.parse_args()

    file_ids = {
        "CodeFormer": {"codeformer.pth": "1v_E_vZvP-dQPF55Kc5SRCjaKTQXDz-JB"},
        "facelib": {
            "yolov5l-face.pth": "131578zMA6B2x8VQHyHfa6GEPtulMCNzV",
            "parsing_parsenet.pth": "16pkohyZZ8ViHGBk3QtVqxLZKzdo466bK",
        },
    }

    if args.method == "all":
        for method in file_ids.keys():
            download_pretrained_models(method, file_ids[method])
    else:
        download_pretrained_models(args.method, file_ids[args.method])
