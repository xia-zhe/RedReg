import argparse
import copy as cp
import csv
import os
import pdb
import pickle
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from dataset.CramedDataset import CramedDataset
# from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import AVDataset
from dataset.KSDataset import KS_dataset
from models.basic_model import AVClassifier
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils import setup_seed, weight_init


def get_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default='KineticSound', type=str)
    parser.add_argument('--dataset', default='CREMAD', type=str)
    parser.add_argument('--modulation', default='InfoReg', type=str, choices=['Normal', 'OGM', 'OGM_GE'])

    parser.add_argument('--fusion_method', default='concat', type=str)
    parser.add_argument('--fps', default=3, type=int) 
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--audio_path', default='/C/BML/InfoReg_CVPR2025-main/crema-d-mirror/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='/C/BML/InfoReg_CVPR2025-main/crema-d-mirror', type=str)
    # parser.add_argument('--audio_path', default='/C/BML/InfoReg_CVPR2025-main/data/ks/data/users/yake_wei/KS_2023/train_spec', type=str)
    # parser.add_argument('--visual_path', default='/C/BML/InfoReg_CVPR2025-main/data/ks/data/users/yake_wei/KS_2023/train-frames-1fps/train', type=str)


    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=80, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.002, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=40, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')
    
    parser.add_argument('--ckpt_path', default= r'/C/BML/InfoReg_CVPR2025-main/ckpt', type=str, help='path to save trained models')
    parser.add_argument('--resume_path', default=None, type=str, help='path to load trained models')
    parser.add_argument('--train', type=int, default=1, help='turn on train mode (1 for train, 0 for eval)')

    parser.add_argument('--use_tensorboard', default=True, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', default='/C/BML/InfoReg_CVPR2025-main/log',type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0,1', type=str, help='GPU ids')
    # parser.add_argument('--audio_fim_path',default='/C/BML/crema-d-mirror/AudioMP3',type=str,help='path to store the audio fim')
    # parser.add_argument('--visual_fim_path',default='/C/BML/crema-d-mirror/Image-01-FPS',type=str,help='path to store the visual fim')
    # parser.add_argument('--mm_fim_path',default='/home/hcx/mm_fim_folder',type=str,help='path to store the mm fim')
    # parser.add_argument('--accuracy_path',default='/home/hcx/accuracy_folder',type=str,help='path to store accuracy csv file')

    return parser.parse_args()

def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None, visual_trace_list=None, audio_trace_list=None):
    import copy as cp

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    backbone = model.module if hasattr(model, "module") else model

    class _temp_eval:
        def __init__(self, m): self.m, self.prev = m, m.training
        def __enter__(self): self.m.eval()
        def __exit__(self, exc_type, exc, tb): self.m.train(self.prev)

    def aug_audio(x, sigma):
        noise = torch.randn_like(x) * sigma
        return x + noise

    def aug_image(x, sigma):
        # x: [B, C, H, W]
        noise = torch.randn_like(x) * sigma
        return torch.clamp(x + noise, 0.0, 1.0)  


    total_audio_grad_sum = 0.0
    total_visual_grad_sum = 0.0
    total_audio_count = 0
    total_visual_count = 0

    score_a_hist, score_v_hist = [], []   
    ema_red_a, ema_red_v = 0.0, 0.0       
    ema_momentum = 0.9
    W = 10
    gamma = 0.6
    R = 0.15
    beta_scale = 0.9
    eps = 1e-12

    if hasattr(args, "epochs") and args.epochs:
        AGREE_THR = 0.2 + (0.5 - 0.2) * float(epoch) / max(1, int(args.epochs) - 1)
    else:
        AGREE_THR = 0.3

    p_start, p_end, warm_phase = 0.6, 0.1, 0.3
    if hasattr(args, "epochs") and args.epochs:
        phase_ep = max(1, int(args.epochs * warm_phase))
        ratio = min(1.0, epoch / phase_ep)
        modal_drop_p = p_start + (p_end - p_start) * ratio
    else:
        modal_drop_p = 0.3  

    def _growth_rate(hist, W=10):
        if len(hist) <= W:
            return 1.0
        s1 = sum(hist[-W:]) / W
        s2 = sum(hist[-W-1:-1]) / W
        return (s1 - s2) / (abs(s2) + eps)

    criterion = nn.CrossEntropyLoss()
    global_model = cp.deepcopy(model)
    model.train()
    print("Start training ... ")

    record_names_audio, record_names_visual = [], []
    for name, param in model.named_parameters():
        if 'head' in name:
            continue
        if 'audio' in name:
            record_names_audio.append((name, param))
        elif 'visual' in name:
            record_names_visual.append((name, param))

    _loss = _loss_a = _loss_v = 0.0


    fim, fim_audio, fim_visual = {}, {}, {}
    fim_audio_head, fim_visual_head, fim_mm_head = {}, {}, {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight.requires_grad:
            if 'audio_net' in name:
                fim_audio[name] = torch.zeros_like(module.weight)
            elif 'visual_net' in name:
                fim_visual[name] = torch.zeros_like(module.weight)
            elif 'head_audio' in name:
                fim_audio_head[name] = torch.zeros_like(module.weight)
            elif 'head_video' in name:
                fim_visual_head[name] = torch.zeros_like(module.weight)
            elif 'head' in name:
                fim_mm_head[name] = torch.zeros_like(module.weight)
            fim[name] = torch.zeros_like(module.weight)

    drop_a_cnt = 0
    drop_v_cnt = 0

    for step, (spec, image, label) in enumerate(dataloader):
        spec = spec.to(device)          
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        lr = optimizer.param_groups[0]['lr']


        with torch.no_grad():
            u = torch.rand((), device=device)
            drop_a = bool(u < (modal_drop_p * 0.5))
            drop_v = bool((u >= (modal_drop_p * 0.5)) and (u < modal_drop_p))
            if drop_a and drop_v:
                if torch.rand((), device=device) < 0.5:
                    drop_v = False
                else:
                    drop_a = False

        if drop_a: drop_a_cnt += 1
        if drop_v: drop_v_cnt += 1

        spec_in  = torch.zeros_like(spec)  if drop_a else spec
        image_in = torch.zeros_like(image) if drop_v else image

        audio, visual, a, v, out_a, out_v, out = model(spec_in.unsqueeze(1).float(), image_in.float())


        with torch.no_grad():
            prob_a = F.softmax(out_a, dim=1)[torch.arange(out_a.size(0), device=out_a.device), label].mean().item()
            prob_v = F.softmax(out_v, dim=1)[torch.arange(out_v.size(0), device=out_v.device), label].mean().item()
        score_a_hist.append(prob_a)
        score_v_hist.append(prob_v)

        s_a = _growth_rate(score_a_hist, W=W)
        s_v = _growth_rate(score_v_hist, W=W)

        sigma_audio = 0.03  
        sigma_image = 0.03

        with torch.no_grad(), _temp_eval(backbone.audio_net), _temp_eval(backbone.visual_net):
            spec_base = spec
            image_base = image

            spec_t1 = aug_audio(spec_base, sigma_audio)
            spec_t2 = aug_audio(spec_base, sigma_audio)
            img_t1  = aug_image(image_base, sigma_image)
            img_t2  = aug_image(image_base, sigma_image)

            z_a_t1 = backbone.audio_net(spec_t1.unsqueeze(1).float())
            z_a_t2 = backbone.audio_net(spec_t2.unsqueeze(1).float())
            z_v_t1 = backbone.visual_net(img_t1.float())
            z_v_t2 = backbone.visual_net(img_t2.float())

            num_a = (z_a_t1 - z_a_t2).pow(2).mean()
            den_a = (spec_t1 - spec_t2).pow(2).mean() + eps
            red_a = (num_a / den_a).item()

            num_v = (z_v_t1 - z_v_t2).pow(2).mean()
            den_v = (img_t1  - img_t2 ).pow(2).mean() + eps
            red_v = (num_v / den_v).item()

        ema_red_a = ema_momentum * ema_red_a + (1 - ema_momentum) * red_a
        ema_red_v = ema_momentum * ema_red_v + (1 - ema_momentum) * red_v
        red_a_norm = red_a / (ema_red_a + eps)
        red_v_norm = red_v / (ema_red_v + eps)
        r_a = red_a_norm - gamma * max(s_a, 0.0)
        r_v = red_v_norm - gamma * max(s_v, 0.0)

        if (prob_a < prob_v) and (r_v > R):
            delta  = torch.tensor(prob_v - prob_a, device=out_v.device).clamp_min(0.0)
            beta_v = torch.exp(torch.tensor(beta_scale, device=out_v.device) * torch.tanh(delta))
            beta_a = torch.tensor(0.0, device=out_v.device)
        elif (prob_v < prob_a) and (r_a > R):
            delta  = torch.tensor(prob_a - prob_v, device=out_a.device).clamp_min(0.0)
            beta_a = torch.exp(torch.tensor(beta_scale, device=out_a.device) * torch.tanh(delta))
            beta_v = torch.tensor(0.0, device=out_a.device)
        else:
            any_dev = out_a.device
            beta_a = torch.tensor(0.0, device=any_dev)
            beta_v = torch.tensor(0.0, device=any_dev)

        with torch.no_grad():
            a_n = F.normalize(a.flatten(1), dim=1)
            v_n = F.normalize(v.flatten(1), dim=1)
            sim = a_n @ v_n.t()
            pos_sim = sim.diag().mean().item()

        agree_gate = 1.0 if pos_sim > AGREE_THR else 0.0

        beta_a = beta_a * agree_gate
        beta_v = beta_v * agree_gate


        loss = criterion(out, label)
        loss_out_v = criterion(out_v, label)
        loss_out_a = criterion(out_a, label)

        losses = [loss, loss_out_a, loss_out_v]
        all_loss = ['both', 'audio', 'visual']
        grads_audio, grads_visual = {}, {}

        for idx, loss_type in enumerate(all_loss):
            loss_tem = losses[idx]
            loss_tem.backward(retain_graph=True)
            if loss_type == 'visual':
                for tensor_name, param in record_names_visual:
                    if loss_type not in grads_visual:
                        grads_visual[loss_type] = {}
                    grads_visual[loss_type][tensor_name] = param.grad.data.clone() if param.grad is not None else torch.zeros_like(param.data)
                grads_visual[loss_type]["concat"] = torch.cat([grads_visual[loss_type][n].flatten() for n, _ in record_names_visual])
                average_grad_visual = torch.mean(grads_visual[loss_type]["concat"]).item()
                if writer is not None:
                    writer.add_scalar('average_grad_visual', average_grad_visual, epoch)
            elif loss_type == 'audio':
                for tensor_name, param in record_names_audio:
                    if loss_type not in grads_audio:
                        grads_audio[loss_type] = {}
                    grads_audio[loss_type][tensor_name] = param.grad.data.clone() if param.grad is not None else torch.zeros_like(param.data)
                grads_audio[loss_type]["concat"] = torch.cat([grads_audio[loss_type][n].flatten() for n, _ in record_names_audio])
                average_grad_audio = torch.mean(grads_audio[loss_type]["concat"]).item()
                if writer is not None:
                    writer.add_scalar('average_grad_audio', average_grad_audio, epoch)
            else:  
                for tensor_name, param in record_names_audio:
                    if loss_type not in grads_audio:
                        grads_audio[loss_type] = {}
                    grads_audio[loss_type][tensor_name] = param.grad.data.clone() if param.grad is not None else torch.zeros_like(param.data)
                grads_audio[loss_type]["concat"] = torch.cat([grads_audio[loss_type][n].flatten() for n, _ in record_names_audio])
                average_grad_audio_mm = torch.mean(grads_audio[loss_type]["concat"]).item()
                if writer is not None:
                    writer.add_scalar('average_grad_audio_mm', average_grad_audio_mm, epoch)

                for tensor_name, param in record_names_visual:
                    if loss_type not in grads_visual:
                        grads_visual[loss_type] = {}
                    grads_visual[loss_type][tensor_name] = param.grad.data.clone() if param.grad is not None else torch.zeros_like(param.data)
                grads_visual[loss_type]["concat"] = torch.cat([grads_visual[loss_type][n].flatten() for n, _ in record_names_visual])
                average_grad_visual_mm = torch.mean(grads_visual[loss_type]["concat"]).item()
                if writer is not None:
                    writer.add_scalar('average_grad_visual_mm', average_grad_visual_mm, epoch)

            optimizer.zero_grad()

        loss_out_v.backward(retain_graph=True)
        loss_out_a.backward(retain_graph=True)
        loss.backward()

        EPS = 1e-12
        for model_param, global_model_param in zip(model.parameters(), global_model.parameters()):
            if not model_param.requires_grad:
                continue

            d = model_param - global_model_param
            g = model_param.grad

            if (g is None) or (g.abs().sum() == 0):
                d_safe = d
            else:
                dot = (d * g).sum()
                g2  = (g * g).sum() + EPS
                d_proj = (dot / g2) * g
                d_safe = d - d_proj

            if any(model_param is p for p in backbone.audio_net.parameters()):
                model_param.grad = model_param.grad + beta_a * d_safe
            if any(model_param is p for p in backbone.visual_net.parameters()):
                model_param.grad = model_param.grad + beta_v * d_safe

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight.requires_grad and module.weight.grad is not None:
                fim[name] += (module.weight.grad * module.weight.grad); fim[name].detach_()

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight.requires_grad and module.weight.grad is not None:
                if 'audio_net' in name:
                    fim_audio[name] += (module.weight.grad * module.weight.grad); fim_audio[name].detach_()
                elif 'visual_net' in name:
                    fim_visual[name] += (module.weight.grad * module.weight.grad); fim_visual[name].detach_()
                elif 'head_audio' in name:
                    fim_audio_head[name] += (module.weight.grad * module.weight.grad); fim_audio_head[name].detach_()
                elif 'head_video' in name:
                    fim_visual_head[name] += (module.weight.grad * module.weight.grad); fim_visual_head[name].detach_()
                elif 'head' in name:
                    fim_mm_head[name] += (module.weight.grad * module.weight.grad); fim_mm_head[name].detach_()

        for name, parms in model.named_parameters():
            if parms.grad is not None:
                layer = str(name).split('.')[1]
                if 'audio' in layer and len(parms.grad.size()) == 4:
                    total_audio_grad_sum += lr * torch.sum(parms.grad ** 2).item(); total_audio_count += 1
                if 'visual' in layer and len(parms.grad.size()) == 4:
                    total_visual_grad_sum += lr * torch.sum(parms.grad ** 2).item(); total_visual_count += 1

        optimizer.step()

        _loss  += loss.item()
        _loss_a += loss_out_a.item()
        _loss_v += loss_out_v.item()

    epoch_audio_L2_norm_mean  = (total_audio_grad_sum  / total_audio_count ) if total_audio_count  > 0 else 0.0
    epoch_visual_L2_norm_mean = (total_visual_grad_sum / total_visual_count) if total_visual_count > 0 else 0.0

    fim_trace = 0.0
    for name in fim:
        fim[name] = fim[name].mean().item(); fim_trace += fim[name]

    fim_trace_audio = 0.0
    for name in fim_audio:
        fim_audio[name] = fim_audio[name].mean().item(); fim_trace_audio += fim_audio[name]

    fim_trace_visual = 0.0
    for name in fim_visual:
        fim_visual[name] = fim_visual[name].mean().item(); fim_trace_visual += fim_visual[name]

    fim_trace_visual_head = 0.0
    for name in fim_visual_head:
        fim_visual_head[name] = fim_visual_head[name].mean().item(); fim_trace_visual_head += fim_visual_head[name]

    fim_trace_audio_head = 0.0
    for name in fim_audio_head:
        fim_audio_head[name] = fim_audio_head[name].mean().item(); fim_trace_audio_head += fim_audio_head[name]

    fim_trace_mm_head = 0.0
    for name in fim_mm_head:
        fim_mm_head[name] = fim_mm_head[name].mean().item(); fim_trace_mm_head += fim_mm_head[name]

    if visual_trace_list is not None: visual_trace_list.append(fim_trace_visual)
    if audio_trace_list  is not None: audio_trace_list.append(fim_trace_audio)

    if writer is not None:
        writer.add_scalar('fim_trace_visual_head', fim_trace_visual_head, epoch)
        writer.add_scalar('fim_trace_audio_head',  fim_trace_audio_head,  epoch)
        writer.add_scalar('fim_trace_mm_head',     fim_trace_mm_head,     epoch)
        writer.add_scalar('modal_drop_p', modal_drop_p, epoch)
        if (drop_a_cnt + drop_v_cnt) > 0:
            writer.add_scalar('modal_drop_audio_ratio', drop_a_cnt / max(1, drop_a_cnt + drop_v_cnt), epoch)
            writer.add_scalar('modal_drop_visual_ratio', drop_v_cnt / max(1, drop_a_cnt + drop_v_cnt), epoch)

    scheduler.step()

    average_grad_visual     = locals().get('average_grad_visual', 0.0)
    average_grad_audio      = locals().get('average_grad_audio', 0.0)
    average_grad_visual_mm  = locals().get('average_grad_visual_mm', 0.0)
    average_grad_audio_mm   = locals().get('average_grad_audio_mm', 0.0)

    return (_loss / len(dataloader),
            _loss_a / len(dataloader),
            _loss_v / len(dataloader),
            epoch_audio_L2_norm_mean,
            epoch_visual_L2_norm_mean,
            fim_trace, fim_trace_audio, fim_trace_visual,
            average_grad_visual, average_grad_audio, average_grad_visual_mm, average_grad_audio_mm)


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():

        model.eval()

        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            audio, visual, a, v,a_output,v_output, out = model(spec.unsqueeze(1).float(), image.float())

            if args.fusion_method == 'sum':
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                         model.module.fusion_module.fc_y.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                         model.module.fusion_module.fc_x.bias / 2)
            else:
                out_v = torch.mm(v, torch.transpose(model.module.head.weight[:, 512:], 0, 1)) +model.module.head.bias / 2

                out_a = torch.mm(a, torch.transpose(model.module.head.weight[:, :512], 0, 1))  + model.module.head.bias / 2

            prediction = softmax(out)
            pred_v = softmax(v_output)
            pred_a = softmax(a_output)

            # prediction = softmax(out)
            # pred_v = softmax(out_v)
            # pred_a = softmax(out_a)


            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():
    args = get_arguments()
    print(args)

    audio_NormList = [0,0,0,0,0,0,0,0,0,0]
    visual_NormList = [0,0,0,0,0,0,0,0,0,0]
    audio_FGNList = []
    visual_FGNList = []
    audio_GNorm_list = []
    visual_GNorm_list = []
    fim_list = []
    audio_fim_list = []
    visual_fim_list = []
    accuracy_list = []
    accuracy_list_audio = []
    accuracy_list_visual = []
    avarage_gradient_visual_mm_list = []
    avarage_gradient_audio_mm_list = []
    avarage_gradient_visual_list = []
    avarage_gradient_audio_list = []
    
    visual_trace_list = [0,0,0,0,0,0,0,0,0,0]
    audio_trace_list = [0,0,0,0,0,0,0,0,0,0]



    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    model = AVClassifier(args)

    model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.cuda()


    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)


    if args.dataset == 'KineticSound':
        train_dataset = KS_dataset(args, mode='train')
        test_dataset = KS_dataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')

    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    print(f"Number of samples in training dataset: {len(train_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=32, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True)


    if args.train:
        if args.resume_path is not None:
            ckpt = torch.load(args.resume_path, map_location=device)
            modulation = ckpt['modulation']
            fusion = ckpt['fusion']
            state_dict = ckpt['model']
            target = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model

            model_keys_have_module = any(k.startswith('module.') for k in target.state_dict().keys())
            ckpt_keys_have_module  = any(k.startswith('module.') for k in state_dict.keys())

            if model_keys_have_module and not ckpt_keys_have_module:
                state_dict = OrderedDict((f'module.{k}', v) for k, v in state_dict.items())
            elif ckpt_keys_have_module and not model_keys_have_module:
                state_dict = OrderedDict((k.replace('module.', '', 1), v) for k, v in state_dict.items())

            incompat = target.load_state_dict(state_dict, strict=False)

        best_acc = 0.0
        for epoch in range(args.epochs):
            total_audio_grad_sum = 0
            total_visual_grad_sum = 0
            total_audio_count = 0
            total_visual_count = 0

            print('Epoch: {}: '.format(epoch))

            if args.use_tensorboard:
                
                writer_path = os.path.join(args.tensorboard_path, args.dataset)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_{}'.format(args.fusion_method, args.modulation)
                writer = SummaryWriter(os.path.join(writer_path, log_name))

                
                batch_loss, batch_loss_a, batch_loss_v, epoch_audio_L2_norm_mean, epoch_visual_L2_norm_mean,fim_trace ,fim_trace_audio,fim_trace_visual,average_grad_visual,average_grad_audio,average_grad_visual_mm,average_grad_audio_mm = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler,writer,visual_trace_list,audio_trace_list)
                fim_list.append(fim_trace)
                avarage_gradient_visual_list.append(average_grad_visual)
                avarage_gradient_audio_list.append(average_grad_audio)
                avarage_gradient_visual_mm_list.append(average_grad_visual_mm)
                avarage_gradient_audio_mm_list.append(average_grad_audio_mm)
                audio_fim_list.append(fim_trace_audio)
                visual_fim_list.append(fim_trace_visual)

                audio_GNorm_list.append(epoch_audio_L2_norm_mean)
                visual_GNorm_list.append(epoch_visual_L2_norm_mean)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader) 

                accuracy_list.append(acc)
                accuracy_list_visual.append(acc_v)
                accuracy_list_audio.append(acc_a)
                audio_NormList.append(epoch_audio_L2_norm_mean)
                audio_OldNorm = max([np.mean(audio_NormList[-11:-1]), 0.0000001])
                audio_NewNorm = np.mean(audio_NormList[-11:])
                audio_FGNList.append((audio_NewNorm - audio_OldNorm) / audio_NewNorm)

                visual_NormList.append(epoch_visual_L2_norm_mean)
                visual_OldNorm = max([np.mean(visual_NormList[-11:-1]), 0.0000001])
                visual_NewNorm = np.mean(visual_NormList[-11:])
                visual_FGNList.append((visual_NewNorm - visual_OldNorm) / visual_NewNorm)


                writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                            'Audio Loss': batch_loss_a,
                                            'Visual Loss': batch_loss_v}, epoch)

                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Audio Accuracy': acc_a,
                                                  'Visual Accuracy': acc_v}, epoch)

                writer.add_scalar('FIM_Trace', fim_trace,epoch)
                writer.add_scalar('FIM_Trace_audio', fim_trace_audio,epoch)
                writer.add_scalar('Fim_Trace_visual',fim_trace_visual,epoch)
                                            

            else:
                batch_loss, batch_loss_a, batch_loss_v, epoch_audio_L2_norm_mean, epoch_visual_L2_norm_mean,fim_trace,fim_trace_audio,fim_trace_visual = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
                audio_GNorm_list.append(epoch_audio_L2_norm_mean)
                visual_GNorm_list.append(epoch_visual_L2_norm_mean)
                audio_NormList.append(epoch_audio_L2_norm_mean)
                audio_OldNorm = max([np.mean(audio_NormList[-2:-1]), 0.0000001])
                audio_NewNorm = np.mean(audio_NormList[-2:])
                audio_FGNList.append((audio_NewNorm - audio_OldNorm) / audio_NewNorm)
                print("audio_FGN:", (audio_NewNorm - audio_OldNorm) / audio_OldNorm)
                visual_NormList.append(epoch_visual_L2_norm_mean)
                visual_OldNorm = max([np.mean(visual_NormList[-2:-1]), 0.0000001])
                visual_NewNorm = np.mean(visual_NormList[-2:])
                visual_FGNList.append((visual_NewNorm - visual_OldNorm) / visual_NewNorm)
                print("visual_FGN:", (visual_NewNorm - visual_OldNorm) / visual_OldNorm)

            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = 'best_model_of_dataset_{}_{}_' \
                             'optimizer_{}_' \
                             'epoch_{}_acc_{}.pth'.format(args.dataset,
                                                          args.modulation,
                                                          args.optimizer,
                                                          epoch, acc)

                saved_dict = {'saved_epoch': epoch,
                              'modulation': args.modulation,
                              'fusion': args.fusion_method,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)

                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
            else:
                print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"experiments_{timestamp}"

        results_path = os.path.join(os.getcwd(), "results", experiment_folder)
        os.makedirs(results_path, exist_ok=True)


        data_to_save = {
            "fim_list": fim_list,
            "audio_fim_list": audio_fim_list,
            "visual_fim_list": visual_fim_list,
            "accuracy_list": accuracy_list,
            "accuracy_list_audio": accuracy_list_audio,
            "accuracy_list_visual": accuracy_list_visual,
            "avarage_gradient_visual_mm_list": avarage_gradient_visual_mm_list,
            "avarage_gradient_audio_mm_list": avarage_gradient_audio_mm_list,
            "avarage_gradient_visual_list": avarage_gradient_visual_list,
            "avarage_gradient_audio_list": avarage_gradient_audio_list
            
        }


        for name, data in data_to_save.items():

            pkl_path = os.path.join(results_path, f"{name}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)


            csv_path = os.path.join(results_path, f"{name}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)

                if isinstance(data[0], (list, tuple)):
                    writer.writerows(data)
                else:
                    writer.writerow(data)


    else:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        modulation = ckpt['modulation']
        fusion = ckpt['fusion']
        state_dict = ckpt['model']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'


        target = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model

        model_keys_have_module = any(k.startswith('module.') for k in target.state_dict().keys())
        ckpt_keys_have_module  = any(k.startswith('module.') for k in state_dict.keys())

        if model_keys_have_module and not ckpt_keys_have_module:
            state_dict = OrderedDict((f'module.{k}', v) for k, v in state_dict.items())
        elif ckpt_keys_have_module and not model_keys_have_module:
            state_dict = OrderedDict((k.replace('module.', '', 1), v) for k, v in state_dict.items())

        incompat = target.load_state_dict(state_dict, strict=False)
        if incompat.missing_keys or incompat.unexpected_keys:
            print('Missing keys:', incompat.missing_keys)
            print('Unexpected keys:', incompat.unexpected_keys)

        model.to(device)
        model.eval()
        print('Trained model loaded!')

        with torch.no_grad():
            acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))




if __name__ == "__main__":
    main()
