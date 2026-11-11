import json
import csv
import os
import librosa
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CramedDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        self.frame_order = None

        self.data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}
        if mode == 'train':
            with open(os.path.join(self.data_root, 'CREMAD/training_frame_order.json')) as f:
                self.frame_order = json.load(f)

        self.visual_feature_path = args.visual_path
        self.audio_feature_path = args.audio_path

        self.train_csv = os.path.join(self.data_root, args.dataset + '/train.csv')
        self.test_csv = os.path.join(self.data_root, args.dataset + '/test.csv')

        if mode == 'train':
            csv_file = self.train_csv
        else:
            csv_file = self.test_csv

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.args.fps), item[0])
                # print(visual_path)

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])
                else:
                    raise FileNotFoundError(f'Missing CREMA-D pair: {audio_path}, {visual_path}')


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # audio
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        resamples = np.tile(samples, 3)[:22050*3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        #mean = np.mean(spectrogram)
        #std = np.std(spectrogram)
        #spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        # if len(image_samples) < self.args.fps:
        #     print("--------------------------------------------------------------------------------------")
        #     print(len(image_samples))
        #     print("--------------------------------------------------------------------------------------")
        if len(image_samples) >= self.args.fps:
            select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=False)
        else:
            select_index = np.random.choice(len(image_samples), size=len(image_samples), replace=False)
        select_index.sort()
        # print(select_index)
        images = torch.zeros((self.args.fps, 3, 224, 224))
        for i, value in enumerate(select_index):
            img = Image.open(os.path.join(self.image[idx], image_samples[value])).convert('RGB')
            # print(os.path.join(self.image[idx], image_samples[value]))
            img = transform(img)
            images[i] = img

        images = images.permute(1,0,2,3)

        # label
        label = self.label[idx]

        return spectrogram, images, label