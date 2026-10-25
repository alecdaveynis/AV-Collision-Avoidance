"""Download KITTI raw drives (public S3 mirror) and keep only what the pipeline reads.

    python scripts/download_kitti.py                       # default city drives of 2011_09_26
    python scripts/download_kitti.py 0001 0002 --date 2011_09_26
    python scripts/download_kitti.py --calib-only

Keeps image_02, velodyne_points and oxts (a few hundred MB per drive instead of
1-2 GB), deletes the zip afterwards, and drops the day's calibration files into
data/ where config.CALIB_DIR picks them up. Check the KITTI terms of use first:
https://www.cvlibs.net/datasets/kitti/raw_data.php
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

BASE = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data'
DATA = Path(__file__).resolve().parent.parent / 'data'
KEEP = ('image_02/', 'velodyne_points/', 'oxts/')
DEFAULT_DRIVES = ['0001', '0002', '0011', '0014', '0017', '0018']


def _progress(block, size, total):
    if total > 0 and block % 400 == 0:
        sys.stdout.write(f"\r    {min(100, block * size * 100 // total):3d}%")
        sys.stdout.flush()


def fetch(url, dest):
    print(f"  GET {url}")
    urlretrieve(url, dest, _progress)
    print()


def download_calib(date):
    zip_path = DATA / f'{date}_calib.zip'
    fetch(f'{BASE}/{date}_calib.zip', zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith('.txt'):
                (DATA / Path(name).name).write_bytes(z.read(name))
    zip_path.unlink()
    print(f"  calibration -> {DATA}")


def download_drive(date, drive):
    name = f'{date}_drive_{drive}_sync'
    if (DATA / name / 'image_02').exists():
        print(f"{name}: already present")
        return
    zip_path = DATA / f'{name}.zip'
    print(f"{name}")
    fetch(f'{BASE}/{date}_drive_{drive}/{name}.zip', zip_path)
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if any(f'/{k}' in m for k in KEEP)]
        z.extractall(DATA, members)
    zip_path.unlink()
    # zips unpack as <date>/<drive>/...; flatten to data/<drive>/...
    src = DATA / date / name
    shutil.move(str(src), str(DATA / name))
    try:
        (DATA / date).rmdir()
    except OSError:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('drives', nargs='*', default=DEFAULT_DRIVES, help='4-digit drive numbers')
    p.add_argument('--date', default='2011_09_26')
    p.add_argument('--calib-only', action='store_true')
    a = p.parse_args()
    DATA.mkdir(exist_ok=True)
    if not (DATA / 'calib_cam_to_cam.txt').exists():
        download_calib(a.date)
    if a.calib_only:
        return
    for d in a.drives:
        download_drive(a.date, d)


if __name__ == '__main__':
    main()
