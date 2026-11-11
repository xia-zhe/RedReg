"""Extract the original InfoReg 3-fps JPEG recipe with portable paths."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile

import cv2

ROOT = Path(__file__).resolve().parents[1]


def extract_one(video, destination, expected):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + '.partial-', dir=str(destination.parent)))
    capture = cv2.VideoCapture(str(video))
    try:
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        count = 0
        frame_id = 0
        frame_interval = fps // 3
        if not capture.isOpened() or frame_interval < 1:
            raise ValueError('Cannot decode video: ' + str(video))
        total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        while count < total:
            success, image = capture.read()
            if not success:
                break
            if count % fps == 0:
                frame_id = 0
            if frame_id < frame_interval * 3 and frame_id % frame_interval == 0:
                success, encoded = cv2.imencode('.jpg', image)
                if not success:
                    raise RuntimeError('JPEG encoding failed: ' + str(video))
                encoded.tofile(str(temporary / ('{:05d}.jpg'.format(count))))
            frame_id += 1
            count += 1
        if set(os.listdir(str(temporary))) != set(expected):
            raise ValueError('Extracted frame names differ from RedReg: ' + str(video))
        os.replace(str(temporary), str(destination))
    finally:
        capture.release()
        if temporary.exists():
            shutil.rmtree(str(temporary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default='datasets/CREMA-D')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()
    root = Path(args.dataset_root).resolve()
    metadata = ROOT / 'data/CREMAD'
    training = json.loads((metadata / 'training_frame_order.json').read_text())
    names = []
    for split in ['train', 'test']:
        with (metadata / (split + '.csv')).open(encoding='utf-8-sig') as stream:
            names += [row[0] for row in csv.reader(stream)]
    assert len(names) == len(set(names)) == 7442

    def prepare(name):
        destination = root / 'Image-03-FPS' / name
        if args.skip_existing and destination.is_dir():
            files = os.listdir(str(destination))
            if not files or any(not x.endswith('.jpg') for x in files):
                raise ValueError('Invalid existing frame directory: ' + str(destination))
            if name in training and set(files) != set(training[name]):
                raise ValueError('Incomplete existing training frames: ' + name)
            return name
        # Test frame counts are checked against the source video while extracting;
        # Training filename sets additionally match the preserved RedReg manifest.
        expected = training.get(name)
        if expected is None:
            capture = cv2.VideoCapture(str(root / 'VideoFlash' / (name + '.flv')))
            try:
                fps = int(capture.get(cv2.CAP_PROP_FPS))
                total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                interval = fps // 3
                if fps < 3:
                    raise ValueError('Invalid video FPS: ' + name)
                expected = ['{:05d}.jpg'.format(i) for i in range(total)
                            if i % fps < interval * 3 and (i % fps) % interval == 0]
            finally:
                capture.release()
        extract_one(root / 'VideoFlash' / (name + '.flv'), destination, expected)
        return name

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, _ in enumerate(pool.map(prepare, names), 1):
            if index % 250 == 0 or index == len(names):
                print('FRAMES {}/{}'.format(index, len(names)), flush=True)


if __name__ == '__main__':
    main()
