import os
import random
import csv
import cv2
import pdb
import numpy as np

def create_csv(path):
    videos = os.listdir(path)
    source = videos.copy()
    driving = videos.copy()
    random.shuffle(source)
    random.shuffle(driving)
    source = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    zeros = np.zeros((len(source),1))
    content = np.concatenate((source,driving,zeros),1)
    f = open('vox256.csv','w',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","frame"])
    csv_writer.writerows(content)
    f.close()

def select_and_save_frames(video_dir, output_dir, csv_path):
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4'))]
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["video_path", "source_frame", "target_frame"])
        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Unable to open video file {video_path}")
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 2:
                print(f"Error: Video {video_path} does not have enough frames to select two.")
                cap.release()
                continue
            for _ in range(5):
                if total_frames < 2:
                    break
                frame_indices = random.sample(range(total_frames), 2)
                selected_frames = []
                for i, frame_idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Error: Unable to read frame at index {frame_idx} in {video_file}")
                        continue
                    output_path = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}_frame_{frame_idx}.png")
                    cv2.imwrite(output_path, frame)
                    selected_frames.append(frame_idx)
                if len(selected_frames) == 2:
                    csv_writer.writerow([video_path, selected_frames[0], selected_frames[1]])
            cap.release()

if __name__ == '__main__':
    # create_csv('/data/fhongac/origDataset/vox1/test')
    video_dir = r"C:\Users\mobil\Desktop\verilight\CVPR2022-DaGAN\video-preprocessing\vox\test"
    output_dir = r"C:\Users\mobil\Desktop\verilight\CVPR2022-DaGAN\video-preprocessing\vox\test\frames"
    csv_path = r"C:\Users\mobil\Desktop\verilight\CVPR2022-DaGAN\data\vox_evaluation_v3.csv"
    select_and_save_frames(video_dir, output_dir, csv_path)