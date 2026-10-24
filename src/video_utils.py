import glob
import os

import cv2


def save_to_video(frames_dir, video_name, fps):
    processed_images = sorted(glob.glob(os.path.join(str(frames_dir), '*.png')))
    if processed_images:
        frame = cv2.imread(processed_images[0])
        h, w, _ = frame.shape
        video = cv2.VideoWriter(str(video_name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        for img_path in processed_images:
            video.write(cv2.imread(img_path))
        video.release()
        return True
    return False


def save_gif(frames_dir, gif_path, fps, width=560, step=3):
    """Small looping GIF for the README (every ``step``-th frame, resized)."""
    from PIL import Image

    paths = sorted(glob.glob(os.path.join(str(frames_dir), '*.png')))[::step]
    if not paths:
        return False
    imgs = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
        imgs.append(im.quantize(colors=128, method=Image.Quantize.FASTOCTREE))
    imgs[0].save(str(gif_path), save_all=True, append_images=imgs[1:],
                 duration=int(1000 * step / fps), loop=0, optimize=True)
    return True
