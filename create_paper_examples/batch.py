# call demo multiple times

import os
import subprocess

# traverse source&driving folder to find all combinations
source_folder = "src"
image_files = []
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith(".png") or file.endswith(".jpg"):
            image_files.append(file)
driving_folder = "driving"

video_files = []
for root, dirs, files in os.walk(driving_folder):
    for file in files:
        if file.endswith(".mp4"):
            video_files.append(file)

# call demo.py for each combination
for ckpt in ["cos10", "cos20", "cos30", "cos40", "original"]:
    for source in image_files:
        for driving in video_files:
            print(f"Processing {source} and {driving}")
        
            subprocess.run([
                "python", 
                "demo.py",
                "--config", r"C:\Users\mobil\Desktop\verilight_attacks\CVPR2022-DaGAN\config\vox-adv-256.yaml",
                "--checkpoint", rf"C:\Users\mobil\Desktop\verilight_attacks\CVPR2022-DaGAN\create_paper_examples\ckpts\{ckpt}.tar",
                "--source_image", rf"./src/{source}",
                "--driving_video", rf"./driving/{driving}",
                "--relative",
                "--adapt_scale",
                "--kp_num", "15",
                "--result_video", rf"./res/{source}_{driving}_{ckpt}.mp4",
                "--generator", "DepthAwareGenerator"
            ])

# extract frames from videos for all generated deepfakes
for ckpt in ["cos10", "cos20", "cos30", "cos40", "original"]:
    for source in image_files:
        for driving in video_files:
            print(f"Processing {source} and {driving}")
        
            os.makedirs(f"res/{source}_{driving}_{ckpt}", exist_ok=True)
            subprocess.run([
                "ffmpeg",
                "-i", rf"res/{source}_{driving}_{ckpt}.mp4",
                "-r", "10",
                rf"res/{source}_{driving}_{ckpt}/%04d.png"
            ])

