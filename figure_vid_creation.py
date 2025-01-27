import os

for i in range(20):
    os.system(f" python demo.py  --config config\\vox-adv-256.yaml --driving_video C:\\Users\\mobil\\Desktop\\cropped_economy_videos\\frame_{i}.mp4 --source_image C:\\Users\\mobil\\Desktop\\cropped_deficit\\frame_{i}.png --checkpoint .\DaGAN_vox_adv_256.pth.tar  --relative --adapt_scale --kp_num 15 --generator DepthAwareGenerator --cpu --result_video result_{i}.mp4")