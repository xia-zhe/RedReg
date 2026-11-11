"""Exact-feature caching and persistent workers with the original epoch RNG protocol.

The iterator adapter is validated specifically for the deployed PyTorch 1.8.
"""
import hashlib
import itertools
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torchvision import transforms

from dataset.CramedDataset import CramedDataset
from redreg.data import seed_worker


class EpochSeedDataset(Dataset):
    def __init__(self, dataset, mode, seed):
        self.dataset = dataset
        self.mode, self.seed = mode, seed
        self.seen_version = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, tagged_index):
        version, base_seed, index = tagged_index
        if version != self.seen_version:
            worker = get_worker_info()
            random.seed(base_seed + worker.id)
            torch.manual_seed(base_seed + worker.id)
            seed_worker(worker.id, self.mode, self.seed)
            self.seen_version = version
        return self.dataset[index]


class EpochTaggedBatches:
    def __init__(self, batches, epoch_seed):
        self.batches, self.epoch_seed = batches, epoch_seed

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for indices in self.batches:
            version, seed = self.epoch_seed
            yield [(version, seed, index) for index in indices]


class ProtocolPersistentLoader(DataLoader):
    def __init__(self, dataset, worker_mode='legacy', training_seed=0, **kwargs):
        if not torch.__version__.startswith('1.8.'):
            raise RuntimeError('Persistent RNG adapter requires revalidation outside PyTorch 1.8')
        if kwargs.get('num_workers', 0) <= 0:
            raise ValueError('Persistent loader requires workers')
        self.epoch_seed = [0, 0]
        wrapped = EpochSeedDataset(dataset, worker_mode, training_seed)
        super().__init__(wrapped, persistent_workers=True, **kwargs)
        # Tag indices at dispatch, so prefetched work from a truncated old epoch
        # never consumes the new epoch's RNG before the new epoch actually starts.
        self.__dict__['batch_sampler'] = EpochTaggedBatches(self.batch_sampler, self.epoch_seed)

    def __iter__(self):
        if self._iterator is not None:
            # A fresh reference iterator consumes this CPU seed BEFORE the sampler
            # is advanced. Persistent _reset does not, so reproduce that draw.
            base_seed = torch.empty((), dtype=torch.int64).random_(generator=self.generator).item()
            self.epoch_seed[0] += 1
            self.epoch_seed[1] = base_seed
            # PyTorch 1.8 retains round-robin position across persistent resets;
            # a newly created reference iterator starts again at worker zero.
            self._iterator._worker_queue_idx_cycle = itertools.cycle(range(self.num_workers))
        return super().__iter__()


class AudioCache:
    def __init__(self, folder, verify=True):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / 'manifest.json').read_text())
        if self.manifest['recipe'] != 'librosa0.9.2-sr22050-tile3-3sec-clip-stft512-hop353-log1e-7-float32':
            raise ValueError('Unrecognized audio feature cache recipe')
        root = Path(__file__).resolve().parents[1]
        for split, expected in self.manifest['split_sha256'].items():
            if hashlib.sha256((root / 'data/CREMAD' / (split + '.csv')).read_bytes()).hexdigest() != expected:
                raise ValueError('Audio cache split SHA256 mismatch')
        self.index = {name: i for i, name in enumerate(self.manifest['names'])}
        self.array = None
        if verify:
            digest = hashlib.sha256()
            with (self.folder / 'features.npy').open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != self.manifest['features_sha256']:
                raise ValueError('Audio cache SHA256 mismatch')

    def __getstate__(self):
        return {**self.__dict__, 'array': None}

    def __getitem__(self, name):
        if self.array is None:
            self.array = np.load(str(self.folder / 'features.npy'), mmap_mode='r')
        return self.array[self.index[name]].copy()

    def validate_source(self, audio_path):
        if Path(audio_path).resolve() != Path(self.manifest['source']).resolve():
            raise ValueError('Audio cache belongs to a different WAV source directory')


class CachedTrainingDataset(CramedDataset):
    def __init__(self, args, cache):
        super().__init__(args, mode='train')
        self.cache = cache
        minimum = getattr(args, 'crop_min_scale', .08)
        if not 0. < minimum <= 1.:
            raise ValueError('Crop minimum scale must be in (0, 1]')
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(minimum, 1.)), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def __getitem__(self, idx):
        # Same listdir order, NumPy calls, torch augmentation calls and zero padding
        # as the reference CramedDataset. Audio preprocessing consumes no RNG.
        folder = Path(self.image[idx])
        spec = self.cache[folder.name]
        candidates = self.frame_order[folder.name]
        selected = np.random.choice(len(candidates), size=min(self.args.fps, len(candidates)), replace=False)
        selected.sort()
        images = torch.zeros(self.args.fps, 3, 224, 224)
        for j, index in enumerate(selected):
            with Image.open(str(folder / candidates[index])) as im:
                images[j] = self.transform(im.convert('RGB'))
        return spec, images.permute(1, 0, 2, 3), self.label[idx]
