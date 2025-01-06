import os

f_in = open("vox_evaluation_v3.csv", "r")
f_new = open("vox_evaluation_v4.csv", "w")
for line in f_in:
    video_path, source_frame, driving_frame = line.split(",")
    video_name = video_path.split("\\")[-1].split(".mp4")[0]
    source_frame_path = f"C:\\Users\\mobil\\Desktop\\verilight\\CVPR2022-DaGAN\\video-preprocessing\\vox\\test\\frames\\{video_name}_frame_{source_frame}.png"
    driving_frame_path = f"test\\{video_name}_frame_{driving_frame}.png"
    if not os.path.exists(source_frame_path):
        print("alert")
        print(source_frame_path)
    else:
        print("good")
    