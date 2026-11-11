import csv
import hashlib
import json
import os
from functools import partial
from pathlib import Path
import random
import numpy as np
import librosa
import torch
from PIL import Image
from torchvision import transforms
from dataset.CramedDataset import CramedDataset


def seed_worker(worker_id, mode, seed):
    if mode == 'legacy':
        # Explicitly reproduce the inherited NumPy state of the existing baseline.
        np.random.seed(seed)
    else:
        np.random.seed(torch.initial_seed() % (2 ** 32))
    # Python and torch have already received distinct worker seeds from DataLoader.


def worker_initializer(mode, seed):
    return partial(seed_worker, mode=mode, seed=seed)


def make_fixed_test_frames(args, run_dir):
    """Load RedReg filenames instead of regenerating filesystem-dependent choices."""
    source = Path(__file__).resolve().parents[1] / 'data/CREMAD'
    blob = (source / 'evaluation_frames.json').read_bytes()
    payload = json.loads(blob)
    if args.fps != 3 or args.eval_seed != payload['seed']:
        raise ValueError('The RedReg frame manifest requires fps=3 and eval_seed=0')
    if hashlib.sha256((source / 'test.csv').read_bytes()).hexdigest() != payload['split_sha256']:
        raise ValueError('Test split differs from the RedReg frame manifest')
    path = Path(run_dir) / 'evaluation_frames.json'
    if path.exists() and path.read_bytes() != blob:
        raise ValueError('Existing run uses a different evaluation frame manifest')
    path.write_bytes(blob)
    return payload['frames'], hashlib.sha256(blob).hexdigest()


class FixedFrameDataset(CramedDataset):
    def __init__(self, args, frame_manifest, audio_cache=None, split='test'):
        super().__init__(args, mode=split)
        self.audio_cache = audio_cache
        self.frame_manifest = frame_manifest
        self.transform = transforms.Compose([
            transforms.Resize(size=(224, 224)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def __getitem__(self, idx):
        if self.audio_cache is None:
            samples, _ = librosa.load(self.audio[idx], sr=22050)
            samples = np.tile(samples, 3)[:22050 * 3]
            samples = np.clip(samples, -1., 1.)
            spec = np.log(np.abs(librosa.stft(samples, n_fft=512, hop_length=353)) + 1e-7)
        else:
            spec = self.audio_cache[Path(self.audio[idx]).stem]
        name = Path(self.image[idx]).name
        images = torch.zeros(self.args.fps, 3, 224, 224)
        for j, filename in enumerate(self.frame_manifest[name]):
            with Image.open(os.path.join(self.image[idx], filename)) as im:
                images[j] = self.transform(im.convert('RGB'))
        return spec, images.permute(1, 0, 2, 3), self.label[idx]
