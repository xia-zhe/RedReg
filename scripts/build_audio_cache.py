"""Precompute the unmodified deterministic CREMA-D audio recipe once."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import librosa
import numpy as np


def feature(path):
    samples, _ = librosa.load(str(path), sr=22050)
    samples = np.tile(samples, 3)[:22050 * 3]
    samples[samples > 1.] = 1.
    samples[samples < -1.] = -1.
    return np.log(np.abs(librosa.stft(samples, n_fft=512, hop_length=353)) + 1e-7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', required=True)
    p.add_argument('--audio-path', default='datasets/CREMA-D/AudioWAV')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    folder = Path(args.folder); folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'manifest.json').exists():
        raise FileExistsError('Completed cache already exists')
    root = Path(__file__).resolve().parents[1]
    names = []
    for split in ['train', 'test']:
        with (root / 'data/CREMAD' / (split + '.csv')).open(encoding='utf-8-sig') as f:
            names += [row[0] for row in csv.reader(f)]
    assert len(names) == len(set(names)) == 7442
    started = time.monotonic()
    audio_root = Path(args.audio_path)
    first = feature(audio_root / (names[0] + '.wav'))  # Warm up resampling/JIT once.
    shape = (len(names),) + first.shape
    temp = folder / 'features.partial.npy'
    data = np.lib.format.open_memmap(str(temp), mode='w+', dtype=first.dtype, shape=shape)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, value in enumerate(pool.map(feature, (audio_root / (name + '.wav') for name in names))):
            assert value.shape == first.shape and value.dtype == first.dtype
            data[index] = value
            if (index + 1) % 500 == 0:
                print('CACHE {}/{} elapsed={:.1f}s'.format(index + 1, len(names), time.monotonic() - started), flush=True)
    data.flush(); del data
    os.replace(str(temp), str(folder / 'features.npy'))
    digest = hashlib.sha256()
    with (folder / 'features.npy').open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    manifest = {'recipe': 'librosa0.9.2-sr22050-tile3-3sec-clip-stft512-hop353-log1e-7-float32',
                'names': names, 'shape': shape, 'dtype': str(first.dtype),
                'features_sha256': digest.hexdigest(), 'librosa': librosa.__version__,
                'elapsed_seconds': time.monotonic() - started,
                'source': str(audio_root.resolve()),
                'split_sha256': {s: hashlib.sha256((root / 'data/CREMAD' / (s + '.csv')).read_bytes()).hexdigest()
                                 for s in ['train', 'test']}}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'names'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
