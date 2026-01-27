import cv2
import glob
import os

def save_to_video(output_dir, video_name, fps):
    processed_images = sorted(glob.glob(os.path.join(output_dir, '*.png')))
    if processed_images:
        frame = cv2.imread(processed_images[0])
        h, w, _ = frame.shape
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        for img_path in processed_images:
            video.write(cv2.imread(img_path))
        video.release()
        return True
    return False