"""Apply the RedReg transport policy before importing the training entry point."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader
from dp_host_transport import install

original_init = DataLoader.__init__


def without_pinning(self, *args, **kwargs):
    if len(args) > 7:
        raise ValueError('Use keyword arguments for DataLoader pin_memory')
    kwargs['pin_memory'] = False
    return original_init(self, *args, **kwargs)


if __name__ == '__main__':
    DataLoader.__init__ = without_pinning
    install()
    runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
