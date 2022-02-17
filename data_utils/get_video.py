import os

ori_dir = '../../version1'
for animal in os.listdir(ori_dir):
    command = 'ffmpeg - y - r 24 - i {}{}.png - pix_fmt yuv420p {}{}.mp4'