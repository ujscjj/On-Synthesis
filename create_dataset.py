
import librosa
import numpy as np
import os


def create_dataset(root, mode, split, txt_file):
    ori_sr = 16000
    dst_sr = 8000
    paths = open(txt_file, 'r').readlines()
    i = 0
    for path in paths:
        path = path.strip('\n').split(' ')
        save_path = root + mode + '/' + split + '/' + path[0].split('/')[-1].split('.')[0] + '_' + path[2].split('/')[-1].split('.')[0]
        spk1 = root + path[0]
        spk1_snr = np.float(path[1])
        spk2 = root + path[2]
        spk2_snr = np.float(path[3])

        spk1, _ = librosa.load(spk1, sr=ori_sr)
        spk2, _ = librosa.load(spk2, sr=ori_sr)
        spk1 = librosa.resample(spk1, ori_sr, dst_sr)
        spk2 = librosa.resample(spk2, ori_sr, dst_sr)

        length = max(len(spk1), len(spk2))
        spk1 = np.pad(spk1, (0, length - len(spk1)), 'constant', constant_values=0)
        spk2 = np.pad(spk2, (0, length - len(spk2)), 'constant', constant_values=0)
        spk1 = spk1 * np.power(10, spk1_snr/20.)
        spk2 = spk2 * np.power(10, spk2_snr/20.)
        mix = spk1 + spk2
        max_amp = np.max(np.stack([np.abs(spk1), np.abs(spk2), np.abs(mix)], axis=0))
        mix_scale = 1 / max_amp * 0.9
        spk1 = spk1 * mix_scale
        spk2 = spk2 * mix_scale
        mix = mix * mix_scale

        os.makedirs(save_path, exist_ok=True)
        librosa.output.write_wav(save_path + '/spk1.wav', spk1, dst_sr)
        librosa.output.write_wav(save_path + '/spk2.wav', spk2, dst_sr)
        librosa.output.write_wav(save_path + '/mix.wav', mix, dst_sr)
        i += 1
        print('Have mixed ' + str(i) + ' mixtures ' + 'for ' + split)


def create_dataset_noisy(root, mode, split, txt_file):
    ori_sr = 16000
    dst_sr = 8000
    paths = open(txt_file, 'r').readlines()
    i = 0
    for path in paths:
        path = path.strip('\n').split(' ')
        save_path = root + mode + '/' + split + '/' + path[0].split('/')[-1].split('.')[0] + '_' + path[2].split('/')[-1].split('.')[0] + '_' + path[4].split('/')[-1].split('.')[0]
        spk1 = root + path[0]
        spk1_snr = np.float(path[1])
        spk2 = root + path[2]
        spk2_snr = np.float(path[3])
        noise = root + path[4]
        noise_snr = np.float(path[5])

        spk1, _ = librosa.load(spk1, sr=ori_sr)
        spk2, _ = librosa.load(spk2, sr=ori_sr)
        noise, _ = librosa.load(noise, sr=ori_sr)
        spk1 = librosa.resample(spk1, ori_sr, dst_sr)
        spk2 = librosa.resample(spk2, ori_sr, dst_sr)
        noise = librosa.resample(noise, ori_sr, dst_sr)

        length = max(len(spk1), len(spk2), len(noise))
        spk1 = np.pad(spk1, (0, length - len(spk1)), 'constant', constant_values=0)
        spk2 = np.pad(spk2, (0, length - len(spk2)), 'constant', constant_values=0)
        noise = np.concatenate([np.concatenate([noise]*(length//len(noise))), noise[:length%len(noise)]])
        spk1 = spk1 * np.power(10, spk1_snr/20.)
        spk2 = spk2 * np.power(10, spk2_snr/20.)
        noise = noise / np.sqrt(np.sum(noise ** 2) + 1e-8) * np.sqrt(np.sum((spk1 + spk2) ** 2) + 1e-8)
        noise = noise * np.power(10, noise_snr/20.)
        mix = spk1 + spk2 + noise
        max_amp = np.max(np.stack([np.abs(spk1), np.abs(spk2), np.abs(noise), np.abs(mix)], axis=0))
        mix_scale = 1 / max_amp * 0.9
        spk1 = spk1 * mix_scale
        spk2 = spk2 * mix_scale
        noise = noise * mix_scale
        mix = mix * mix_scale

        os.makedirs(save_path, exist_ok=True)
        librosa.output.write_wav(save_path + '/spk1.wav', spk1, dst_sr)
        librosa.output.write_wav(save_path + '/spk2.wav', spk2, dst_sr)
        librosa.output.write_wav(save_path + '/noise.wav', noise, dst_sr)
        librosa.output.write_wav(save_path + '/mix.wav', mix, dst_sr)
        i += 1
        print('Have mixed ' + str(i) + ' mixtures ' + 'for ' + split)


libri_path = '' # the absolute path for LibriSpeech data

create_dataset(root=libri_path, mode='2speakers', split='test', txt_file='2spks_test.txt')
create_dataset(root=libri_path, mode='2speakers', split='dev', txt_file='2spks_dev.txt')
create_dataset(root=libri_path, mode='2speakers', split='train', txt_file='2spks_train.txt')

create_dataset_noisy(root=libri_path, mode='2speakers_noisy', split='test', txt_file='2spks_test_noisy.txt')
create_dataset_noisy(root=libri_path, mode='2speakers_noisy', split='dev', txt_file='2spks_dev_noisy.txt')
create_dataset_noisy(root=libri_path, mode='2speakers_noisy', split='train', txt_file='2spks_train_noisy.txt')
