
import librosa
import numpy as np
import os


def create_dataset_increment(root, mode, splits, txt_file):
    ori_sr = 16000
    dst_sr = 8000
    paths = open(txt_file, 'r').readlines()
    i = 0
    for path in paths:
        i += 1
        path = path.strip('\n').split(' ')
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
        spk1_raw = np.pad(spk1, (0, length - len(spk1)), 'constant', constant_values=0)
        spk2_raw = np.pad(spk2, (0, length - len(spk2)), 'constant', constant_values=0)
        noise_raw = np.concatenate([np.concatenate([noise]*(length//len(noise))), noise[:length%len(noise)]])

        for split in splits:
            spk1 = spk1_raw * np.power(10, spk1_snr/20.)
            spk2 = spk2_raw * np.power(10, spk2_snr/20.)
            noise = noise_raw / np.sqrt(np.sum(noise_raw ** 2) + 1e-8) * np.sqrt(np.sum((spk1 + spk2) ** 2) + 1e-8)
            noise = noise * np.power(10, split/20.)
            mix = spk1 + spk2 + noise
            max_amp = np.max(np.stack([np.abs(spk1), np.abs(spk2), np.abs(noise), np.abs(mix)], axis=0))
            mix_scale = 1 / max_amp * 0.9
            spk1 = spk1 * mix_scale
            spk2 = spk2 * mix_scale
            noise = noise * mix_scale
            mix = mix * mix_scale

            save_path = root + mode + '/' + str(split) + '/' + path[0].split('/')[-1].split('.')[0] + '_' + path[2].split('/')[-1].split('.')[0] + '_' + path[4].split('/')[-1].split('.')[0]
            os.makedirs(save_path, exist_ok=True)
            librosa.output.write_wav(save_path + '/spk1.wav', spk1, dst_sr)
            librosa.output.write_wav(save_path + '/spk2.wav', spk2, dst_sr)
            librosa.output.write_wav(save_path + '/noise.wav', noise, dst_sr)
            librosa.output.write_wav(save_path + '/mix.wav', mix, dst_sr)
            print('Have mixed ' + str(i) + ' mixtures ' + 'for ' + str(split))


libri_path = '' # the absolute path for LibriSpeech data
create_dataset_increment(root=libri_path, mode='2speakers_noisy/test_increment', splits=[-10, -5, 0, 5, 10], txt_file='2spks_test_noisy.txt')

