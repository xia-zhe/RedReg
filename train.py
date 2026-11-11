import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset.CramedDataset import CramedDataset
from redreg.core import build_from_baseline_initialization, RedundancyMonitor, prediction_consistency, gated_compression_loss, align_global_readouts
from redreg.data import FixedFrameDataset, make_fixed_test_frames, worker_initializer
from utils.utils import setup_seed
from redreg.randomness import seed_gaussian_streams
from redreg.fast_data import AudioCache, CachedTrainingDataset, ProtocolPersistentLoader

ROOT = Path(__file__).resolve().parent
_SOURCE_HASHES = None


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config')
    p.add_argument('--mode', choices=['joint', 'star'], default='star')
    p.add_argument('--run-name', default='RedReg_star_seed0')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gaussian-streams', choices=['independent'], default='independent')
    p.add_argument('--eval-seed', type=int, default=0)
    p.add_argument('--learning-rate', type=float, default=.002)
    p.add_argument('--probe-learning-rate', type=float, default=.002)
    p.add_argument('--bottleneck-lr-multiplier', type=float, default=1.)
    p.add_argument('--probe-lr-decay-step', type=int)
    p.add_argument('--probe-lr-decay-ratio', type=float)
    p.add_argument('--optimizer', choices=['sgd'], default='sgd')
    p.add_argument('--probe-optimizer', choices=['sgd'], default='sgd')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--momentum', type=float, default=.9)
    p.add_argument('--lr-decay-step', type=int, default=30)
    p.add_argument('--lr-decay-ratio', type=float, default=.1)
    p.add_argument('--window', type=int, default=10)
    p.add_argument('--beta', type=float, default=.3)
    p.add_argument('--gamma', type=float, default=.5)
    p.add_argument('--redundancy-threshold', type=float, default=.90)
    p.add_argument('--agreement-threshold', type=float, default=.5)
    p.add_argument('--epsilon', type=float, default=1e-8)
    p.add_argument('--latent-dim', type=int, default=512)
    p.add_argument('--initial-logvar', type=float, default=-6.)
    p.add_argument('--initial-mean-scale', type=float, default=1.)
    p.add_argument('--bottleneck-type', choices=['gaussian'], default='gaussian')
    p.add_argument('--compression-reduction', choices=['mean_dimension'], default='mean_dimension')
    p.add_argument('--auxiliary-weight', type=float, default=1.)
    p.add_argument('--auxiliary-heads', choices=['shared'], default='shared')
    p.add_argument('--readout-operator', choices=['fused_linear'], default='fused_linear')
    p.add_argument('--backward-policy', choices=['aggregate'], default='aggregate')
    p.set_defaults(execution_mode='data_parallel', development_split=None)
    p.add_argument('--worker-seeding', choices=['legacy'], default='legacy')
    p.add_argument('--num-workers', type=int, default=32)
    p.add_argument('--eval-workers', type=int, default=32)
    p.add_argument('--data-pipeline', choices=['optimized'], default='optimized')
    p.add_argument('--audio-cache', default='datasets/CREMA-D/RedRegAudioCache')
    p.add_argument('--crop-min-scale', type=float, default=.5)
    p.add_argument('--fps', type=int, default=3)
    p.add_argument('--audio-path', default='datasets/CREMA-D/AudioWAV')
    p.add_argument('--visual-path', default='datasets/CREMA-D')
    p.add_argument('--regulation', choices=['redreg'], default='redreg')
    p.add_argument('--max-train-batches', type=int, default=0)
    p.add_argument('--max-eval-batches', type=int, default=0)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--resume')
    p.add_argument('--evaluate')
    known, _ = p.parse_known_args()
    if known.config:
        config = json.loads(Path(known.config).read_text())
        unknown = set(config) - {a.dest for a in p._actions}
        if unknown:
            raise ValueError('Unknown config keys: ' + str(unknown))
        p.set_defaults(**config)
    args = p.parse_args()
    args.dataset, args.fusion_method = 'CREMAD', 'concat'
    return args


def csv_append(path, row):
    fresh = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if fresh:
            writer.writeheader()
        writer.writerow(row)


def source_hashes():
    global _SOURCE_HASHES
    if _SOURCE_HASHES is not None:
        return dict(_SOURCE_HASHES)
    files = ['train.py', 'redreg/core.py', 'redreg/data.py', 'redreg/randomness.py', 'redreg/fast_data.py',
             'models/backbone.py', 'models/basic_model.py', 'models/fusion_modules.py',
             'dataset/CramedDataset.py', 'utils/utils.py', 'scripts/dp_host_transport.py', 'scripts/run.py']
    _SOURCE_HASHES = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}
    return dict(_SOURCE_HASHES)


def make_optimizer(kind, parameters, lr, args):
    if kind == 'adam':
        return torch.optim.Adam(parameters, lr=lr, betas=(.9, .999), eps=1e-8,
                                weight_decay=args.weight_decay)
    return torch.optim.SGD(parameters, lr=lr, momentum=args.momentum,
                           weight_decay=args.weight_decay)


def make_main_optimizer(args, model):
    scale = args.bottleneck_lr_multiplier
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Bottleneck LR multiplier must be finite and positive')
    if scale == 1.:
        return make_optimizer(args.optimizer, model.parameters(), args.learning_rate, args)
    phi = list(model.module.bottlenecks.parameters())
    ids = {id(p) for p in phi}
    common = [p for p in model.parameters() if id(p) not in ids]
    groups = [{'params': common, 'lr': args.learning_rate},
              {'params': phi, 'lr': args.learning_rate * scale}]
    return make_optimizer(args.optimizer, groups, args.learning_rate, args)


def reconcile_journal(path, key, highest_inclusive):
    if not path.exists():
        return
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    kept = [row for row in rows if int(row[key]) <= highest_inclusive]
    if len(kept) != len(rows):
        backup = path.with_name(path.stem + '.interrupted-' + str(time.time_ns()) + '.csv')
        backup.write_bytes(path.read_bytes())
        temp = path.with_suffix('.resume-tmp')
        with temp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(kept)
        os.replace(str(temp), str(path))


def atomic_save(state, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, str(temp))
    os.replace(str(temp), str(path))


def pack_state(args, model, probes, optimizer, probe_optimizer, scheduler,
               probe_scheduler, monitor, epoch, best, metrics, frame_hash):
    return {'epoch': epoch, 'global_step': monitor.steps, 'model': model.module.state_dict(),
            'probes': probes.state_dict(), 'optimizer': optimizer.state_dict(),
            'probe_optimizer': probe_optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'probe_scheduler': probe_scheduler.state_dict(), 'monitor': monitor.state_dict(),
            'best': best.copy(), 'metrics': metrics, 'config': vars(args),
            'evaluation_frames_sha256': frame_hash, 'source_hashes': source_hashes(),
            'execution_rng': None,
            'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                    'numpy': np.random.get_state(), 'python': random.getstate()}}


def evaluate(args, model, probes, loader):
    model.eval(); probes.eval()
    independent = getattr(args, 'auxiliary_heads', 'shared') == 'independent'
    correct = np.zeros(7 if independent else 5, dtype=np.int64)
    loss_sum, n = 0., 0
    with torch.no_grad():
        for step, (spec, images, labels) in enumerate(loader):
            if args.max_eval_batches and step >= args.max_eval_batches:
                break
            labels = labels.cuda(non_blocking=True)
            out = model(spec.cuda(non_blocking=True).unsqueeze(1).float(), images.cuda(non_blocking=True).float())
            out = align_global_readouts(out, model.module)
            probe_logits = probes(out['mu'])
            logits = [out['logits'], out['audio_logits'], out['video_logits'],
                      probe_logits[:, 0], probe_logits[:, 1]]
            if independent:
                logits.extend([out['audio_task_logits'], out['video_task_logits']])
            correct += np.array([int((x.argmax(-1) == labels).sum()) for x in logits])
            loss_sum += float(F.cross_entropy(out['logits'], labels)) * len(labels)
            n += len(labels)
    if n == 0:
        raise RuntimeError('Empty evaluation loader')
    names = ['joint_accuracy', 'audio_readout_accuracy', 'video_readout_accuracy',
             'audio_probe_accuracy', 'video_probe_accuracy']
    if independent:
        names.extend(['audio_task_head_accuracy', 'video_task_head_accuracy'])
    return dict(zip(names, (correct / n).tolist()), evaluation_loss=loss_sum / n,
                evaluation_samples=n)


def train_epoch(args, epoch, model, probes, loader, optimizer, probe_optimizer, monitor, run):
    model.train(); probes.train()
    total_samples, total_steps = 0, 0
    sums = {'joint_ce': 0., 'supervised_task_loss': 0., 'total_loss': 0.,
            'audio_rate': 0., 'video_rate': 0., 'weighted_kl': 0.}
    gate_counts = [0, 0]
    for batch, (spec, images, labels) in enumerate(loader):
        if args.max_train_batches and batch >= args.max_train_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        probe_optimizer.zero_grad(set_to_none=True)
        labels = labels.cuda(non_blocking=True)
        out = model(spec.cuda(non_blocking=True).unsqueeze(1).float(), images.cuda(non_blocking=True).float())
        out = align_global_readouts(out, model.module)
        joint_ce = F.cross_entropy(out['logits'], labels)
        audio_ce = F.cross_entropy(out.get('audio_task_logits', out['audio_logits']), labels)
        video_ce = F.cross_entropy(out.get('video_task_logits', out['video_logits']), labels)
        auxiliary = audio_ce + video_ce
        task_loss = joint_ce + (args.auxiliary_weight * auxiliary if args.mode == 'star' else 0.)
        probe_logits = probes(out['z'])  # ProbeBank always detaches encoder representations.
        probe_losses = torch.stack([F.cross_entropy(probe_logits[:, m], labels) for m in range(2)])
        rates = out['rates'].mean(0)
        with torch.no_grad():
            similarity = prediction_consistency(probes(out['mu']))
        state = monitor.observe(probe_losses.detach().cpu().tolist(), rates.detach().cpu().tolist(), similarity)
        gate = state['gate']
        if args.regulation == 'stochastic_bottleneck':
            gate = [0., 0.]
        elif args.regulation == 'uniform_vib':
            gate = [float(state['ready'])] * 2
        weighted_kl = gated_compression_loss(rates, gate, args.beta, args.latent_dim, args.compression_reduction)
        loss = task_loss + weighted_kl
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite loss at step ' + str(monitor.steps - 1))
        if args.backward_policy == 'aggregate':
            loss.backward()
        else:
            # Match the official baseline's v -> a -> joint accumulation order.
            # Keep v -> a -> joint task order. Include KL with the last full
            # graph pass: an extra partial KL backward can deadlock old DP/NCCL.
            if args.mode == 'star':
                (video_ce if args.auxiliary_weight == 1. else args.auxiliary_weight * video_ce).backward(retain_graph=True)
                (audio_ce if args.auxiliary_weight == 1. else args.auxiliary_weight * audio_ce).backward(retain_graph=True)
            active_compression = bool(any(gate)) and weighted_kl.requires_grad and args.beta != 0
            (joint_ce + weighted_kl if active_compression else joint_ce).backward()
        optimizer.step()
        # Probe update happens after the model update using the same detached z,
        # exactly one step, and cannot add a gradient to the modality encoders.
        probe_losses.sum().backward()
        probe_optimizer.step()
        b = len(labels); total_samples += b; total_steps += 1
        values = {'joint_ce': float(joint_ce.detach()), 'supervised_task_loss': float(task_loss.detach()),
                  'total_loss': float(loss.detach()), 'audio_rate': float(rates[0].detach()),
                  'video_rate': float(rates[1].detach()), 'weighted_kl': float(weighted_kl.detach())}
        for key, value in values.items():
            sums[key] += value * b
        gate_counts = [gate_counts[m] + int(gate[m]) for m in range(2)]
        row = {'epoch_index': epoch, 'batch_index': batch, 'global_step': monitor.steps - 1,
               **values, 'audio_probe_loss': float(probe_losses[0].detach()),
               'video_probe_loss': float(probe_losses[1].detach()), 'similarity': similarity,
               'dominant': state['dominant'], 'gate_audio': gate[0], 'gate_video': gate[1]}
        for m, name in enumerate(['audio', 'video']):
            for key in ['smooth_loss', 'task_gain', 'normalized_gain', 'normalized_rate', 'redundancy']:
                row[name + '_' + key] = state[key][m]
        csv_append(run / 'steps.csv', row)
        if batch % args.log_every == 0:
            print('TRAIN epoch={} step={}/{} global={} ce={:.4f} total={:.4f} rate=({:.2f},{:.2f}) gate={} sim={:.3f}'.format(
                epoch + 1, batch, len(loader), monitor.steps - 1, values['joint_ce'], values['total_loss'],
                values['audio_rate'], values['video_rate'], gate, similarity), flush=True)
    if total_steps == 0:
        raise RuntimeError('Empty training loader')
    return {**{k: v / total_samples for k, v in sums.items()}, 'training_samples': total_samples,
            'training_steps': total_steps, 'audio_gate_fraction': gate_counts[0] / total_steps,
            'video_gate_fraction': gate_counts[1] / total_steps}


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA device is required for this experiment')
    if args.fps != 3:
        raise ValueError('This reproduction is pinned to the prepared 3 fps dataset')
    if args.crop_min_scale != .08 and args.data_pipeline != 'optimized':
        raise ValueError('Nondefault crop policy requires the cached training dataset')
    if args.auxiliary_heads == 'independent' and args.mode != 'star':
        raise ValueError('Independent auxiliary task heads require Star supervision')
    if not args.evaluate and not args.max_train_batches and (torch.cuda.device_count() != 2 or args.batch_size != 64):
        raise ValueError('RedReg full training requires two visible GPUs and global batch size 64')
    run = ROOT / 'runs' / args.run_name
    if (run / 'config.json').exists() and not args.resume and not args.evaluate:
        raise FileExistsError('Existing experiment; use --resume or a new --run-name')
    run.mkdir(parents=True, exist_ok=True)
    (run / 'checkpoints').mkdir(exist_ok=True)
    setup_seed(args.seed)
    seed_gaussian_streams(args.seed, args.gaussian_streams)
    torch.backends.cudnn.benchmark = False
    frames, frame_hash = make_fixed_test_frames(args, run)
    evaluation_split = 'test'
    if args.data_pipeline == 'optimized':
        if not args.audio_cache:
            raise ValueError('Optimized pipeline requires --audio-cache')
        cache = AudioCache(args.audio_cache)
        cache.validate_source(args.audio_path)
        dataset = CachedTrainingDataset(args, cache)
        test_dataset = FixedFrameDataset(args, frames, audio_cache=cache, split=evaluation_split)
    else:
        dataset = CramedDataset(args, mode='train')
        test_dataset = FixedFrameDataset(args, frames, split=evaluation_split)
    expected = (6698, 744)
    if (len(dataset), len(test_dataset)) != expected:
        raise ValueError('Unexpected CREMA-D split size')
    train_kwargs = {'num_workers': args.num_workers, 'pin_memory': True,
                    'worker_init_fn': worker_initializer(args.worker_seeding, args.seed)}
    test_kwargs = {'num_workers': args.eval_workers, 'pin_memory': True}
    if args.num_workers:
        train_kwargs['multiprocessing_context'] = 'spawn'
    if args.eval_workers:
        test_kwargs['multiprocessing_context'] = 'spawn'
    if args.data_pipeline == 'optimized':
        loader = ProtocolPersistentLoader(dataset, batch_size=args.batch_size, shuffle=True,
                    worker_mode=args.worker_seeding, training_seed=args.seed, **train_kwargs)
        test_loader = ProtocolPersistentLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                    worker_mode=args.worker_seeding, training_seed=args.seed, **test_kwargs)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **train_kwargs)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **test_kwargs)
    model, probes = build_from_baseline_initialization(args)
    model = torch.nn.DataParallel(model.cuda())
    probes = probes.cuda()
    optimizer = make_main_optimizer(args, model)
    probe_optimizer = make_optimizer(args.probe_optimizer, probes.parameters(), args.probe_learning_rate, args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)
    probe_scheduler = torch.optim.lr_scheduler.StepLR(probe_optimizer,
        args.probe_lr_decay_step if args.probe_lr_decay_step is not None else args.lr_decay_step,
        args.probe_lr_decay_ratio if args.probe_lr_decay_ratio is not None else args.lr_decay_ratio)
    monitor = RedundancyMonitor(args.window, args.gamma, args.redundancy_threshold,
                                args.agreement_threshold, args.epsilon)
    best = {k: -1. for k in ['joint_accuracy', 'audio_readout_accuracy', 'video_readout_accuracy',
                              'audio_probe_accuracy', 'video_probe_accuracy']}
    start_epoch = 0
    checkpoint_path = args.resume or args.evaluate
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if args.resume:
            if checkpoint.get('inference_only'):
                raise ValueError('This checkpoint is an inference-only evaluation variant; resume its original training checkpoint instead')
            invariant_keys = ['mode', 'seed', 'eval_seed', 'batch_size', 'latent_dim',
                              'initial_logvar', 'window', 'beta', 'gamma', 'redundancy_threshold',
                              'agreement_threshold', 'epsilon', 'auxiliary_weight', 'worker_seeding',
                              'num_workers', 'learning_rate', 'probe_learning_rate', 'momentum',
                              'weight_decay', 'lr_decay_step', 'lr_decay_ratio', 'regulation',
                              'max_train_batches', 'max_eval_batches', 'optimizer', 'probe_optimizer',
                              'gaussian_streams', 'compression_reduction', 'initial_mean_scale', 'bottleneck_type',
                              'execution_mode', 'auxiliary_heads', 'development_split',
                              'probe_lr_decay_step', 'probe_lr_decay_ratio', 'readout_operator', 'backward_policy',
                              'bottleneck_lr_multiplier', 'crop_min_scale']
            changed = [key for key in invariant_keys if getattr(args, key) != checkpoint['config'].get(
                key, 'sgd' if key in ['optimizer', 'probe_optimizer'] else
                'legacy' if key == 'gaussian_streams' else
                'sum' if key == 'compression_reduction' else
                1. if key == 'initial_mean_scale' else
                'gaussian' if key == 'bottleneck_type' else
                'data_parallel' if key == 'execution_mode' else
                'shared' if key == 'auxiliary_heads' else
                'fused_linear' if key == 'readout_operator' else
                'aggregate' if key == 'backward_policy' else
                1. if key == 'bottleneck_lr_multiplier' else
                .08 if key == 'crop_min_scale' else None)]
            if changed:
                raise ValueError('Resume changes experiment semantics: ' + ', '.join(changed))
        if checkpoint['evaluation_frames_sha256'] != frame_hash:
            raise ValueError('Evaluation frame manifest does not match checkpoint')
        model.module.load_state_dict(checkpoint['model'])
        probes.load_state_dict(checkpoint['probes'])
        if args.evaluate:
            metrics = evaluate(args, model, probes, test_loader)
            print(json.dumps(metrics, indent=2), flush=True)
            (run / 'reevaluation.json').write_text(json.dumps(metrics, indent=2))
            return
        optimizer.load_state_dict(checkpoint['optimizer'])
        probe_optimizer.load_state_dict(checkpoint['probe_optimizer'])
        for opt in [optimizer, probe_optimizer]:
            for state in opt.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.cuda()
        scheduler.load_state_dict(checkpoint['scheduler'])
        probe_scheduler.load_state_dict(checkpoint['probe_scheduler'])
        monitor.load_state_dict(checkpoint['monitor'])
        best, start_epoch = checkpoint['best'], checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng']['torch'])
        torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
        np.random.set_state(checkpoint['rng']['numpy']); random.setstate(checkpoint['rng']['python'])
        reconcile_journal(run / 'steps.csv', 'global_step', checkpoint['global_step'] - 1)
        reconcile_journal(run / 'epochs.csv', 'epoch_index', checkpoint['epoch'])
    config = {**vars(args), 'gpu_count': torch.cuda.device_count(), 'torch': torch.__version__,
              'logical_batchnorm_shards': torch.cuda.device_count(),
              'evaluation_role': 'internal_development_validation' if args.development_split else 'benchmark_test',
              'evaluation_frames_sha256': frame_hash, 'source_hashes': source_hashes(),
              'kl_reduction': 'sum_latent_dimensions_then_mean_batch',
              'compression_loss_reduction': args.compression_reduction,
              'effective_raw_kl_beta': args.beta / (args.latent_dim if args.compression_reduction == 'mean_dimension' else 1),
              'monitor_bootstrap': 'available-window means during first L iterations; gate off until t>=L',
              'selection': 'benchmark best test accuracy, explicitly labeled; all branch bests saved separately'}
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)
    print('SPLIT train={} {}={}; '.format(len(dataset), config['evaluation_role'], len(test_dataset)) + ('training state restored from checkpoint' if args.resume
          else 'model initialized from scratch; no trained InfoReg weights loaded'), flush=True)
    started = time.monotonic()
    elapsed_offset = float(checkpoint['metrics'].get('elapsed_seconds', 0.)) if args.resume else 0.
    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]['lr']
        phase_start = time.monotonic()
        train = train_epoch(args, epoch, model, probes, loader, optimizer, probe_optimizer, monitor, run)
        train_seconds = time.monotonic() - phase_start
        phase_start = time.monotonic()
        metrics = evaluate(args, model, probes, test_loader)
        evaluation_seconds = time.monotonic() - phase_start
        scheduler.step(); probe_scheduler.step()
        row = {'epoch_index': epoch, 'epoch_number': epoch + 1, 'learning_rate': lr,
               **train, **metrics, 'elapsed_seconds': elapsed_offset + time.monotonic() - started}
        csv_append(run / 'epochs.csv', row)
        improved = [k for k in best if metrics[k] > best[k]]
        for key in improved:
            best[key] = metrics[key]
        phase_start = time.monotonic()
        state = pack_state(args, model, probes, optimizer, probe_optimizer, scheduler,
                           probe_scheduler, monitor, epoch, best, row, frame_hash)
        atomic_save(state, run / 'checkpoints/last.pt')
        for key in improved:
            atomic_save(state, run / ('checkpoints/best_' + key + '.pt'))
        if epoch + 1 == 50:
            atomic_save(state, run / 'checkpoints/epoch_050.pt')
        checkpoint_seconds = time.monotonic() - phase_start
        csv_append(run / 'performance.csv', {'epoch_number': epoch + 1,
                   'pipeline': args.data_pipeline, 'train_seconds': train_seconds,
                   'evaluation_seconds': evaluation_seconds, 'checkpoint_seconds': checkpoint_seconds})
        summary = {'completed_epochs': epoch + 1, 'target_epochs': args.epochs, 'best': best,
                   'last': row, 'status': 'completed' if epoch + 1 == args.epochs else 'running',
                   'evaluation_role': config['evaluation_role'],
                   'mode': args.mode, 'regulation': args.regulation, 'evaluation_frames_sha256': frame_hash}
        (run / 'summary.json').write_text(json.dumps(summary, indent=2))
        print('EPOCH {} joint={:.4%} audio={:.4%} video={:.4%} probe=({:.4%},{:.4%}) gate=({:.3f},{:.3f}) best_joint={:.4%}'.format(
            epoch + 1, metrics['joint_accuracy'], metrics['audio_readout_accuracy'], metrics['video_readout_accuracy'],
            metrics['audio_probe_accuracy'], metrics['video_probe_accuracy'], train['audio_gate_fraction'],
            train['video_gate_fraction'], best['joint_accuracy']), flush=True)
    print('TRAINING_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
