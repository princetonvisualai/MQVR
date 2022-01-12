import os
import argparse
import subprocess
import glob
from multiprocessing import Pool
from tqdm import tqdm



def extract_images(video_path):

    video_name = video_path.split('/')[-1]
    save_dir = '/'.join(video_path.split('/')[:-2]) + '/extracted_frames'
    os.makedirs(os.path.join(save_dir, video_name), exist_ok=True)
    command = ['/usr/bin/ffmpeg',
               '-i', video_path,
               '-qscale:v', '2',
               '-f', 'image2',
               '-start_number', '0',
               save_dir + '/' + video_name + '/' + '%d.jpg']
    command = ' '.join(command)
    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return status, str(err.output) + command
    

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw_video_path', type=str)
    
    args = p.parse_args()
    video_list = glob.glob(args.raw_video_path + '/*')

    pool = Pool(processes=5)
    for _ in tqdm(pool.imap_unordered(extract_images, video_list), total=len(video_list)):
        pass

