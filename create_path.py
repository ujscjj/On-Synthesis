
import os
import numpy as np


def save_path(path, txt_file):
    try:
        open(txt_file, 'w')
    except FileNotFoundError:
        os.mknod(txt_file)
    with open(txt_file, 'w') as f:
        for i in range(len(path)):
            f.write(path[i] + '\n')
        f.close()
    print('%d paths have been saved in %s' % (len(path), txt_file))


def create_path(root, txt_file):
    split = root.split('/')[-2]
    spks = os.listdir(root)
    spks1 = spks[:len(spks)//2]
    spks2 = spks[len(spks)//2:]
    speechs1 = []
    speechs2 = []
    for spk1 in spks1:
        for chapter in os.listdir(root + spk1 + '/'):
            for speech in os.listdir(root + spk1 + '/' + chapter + '/'):
                if speech.endswith('.flac'):
                    speechs1.append(split + '/' + spk1 + '/' + chapter + '/' + speech)
    for spk2 in spks2:
        for chapter in os.listdir(root + spk2 + '/'):
            for speech in os.listdir(root + spk2 + '/' + chapter + '/'):
                if speech.endswith('.flac'):
                    speechs2.append(split + '/' + spk2 + '/' + chapter + '/' + speech)

    nums = min(len(speechs1), len(speechs2))
    np.random.seed(0)
    randoms = np.random.uniform(-2.5, 2.5, size=(nums,))

    path = []
    for i in range(nums):
        path.append(speechs1[i] + ' ' + str(randoms[i])[:6] + ' ' + speechs2[i] + ' ' + str(-randoms[i])[:6])

    save_path(path, txt_file)


def create_path_noisy(root, noise_root, txt_file):
    split = root.split('/')[-2]
    spks = os.listdir(root)
    spks1 = spks[:len(spks)//2]
    spks2 = spks[len(spks)//2:]
    speechs1 = []
    speechs2 = []
    noises = []
    for spk1 in spks1:
        for chapter in os.listdir(root + spk1 + '/'):
            for speech in os.listdir(root + spk1 + '/' + chapter + '/'):
                if speech.endswith('.flac'):
                    speechs1.append(split + '/' + spk1 + '/' + chapter + '/' + speech)
    for spk2 in spks2:
        for chapter in os.listdir(root + spk2 + '/'):
            for speech in os.listdir(root + spk2 + '/' + chapter + '/'):
                if speech.endswith('.flac'):
                    speechs2.append(split + '/' + spk2 + '/' + chapter + '/' + speech)
    for noise in os.listdir(noise_root):
        noises.append(noise_root.split('/')[-2] + '/' + noise)

    nums = min(len(speechs1), len(speechs2))
    np.random.seed(0)
    randoms = np.random.uniform(-2.5, 2.5, size=(nums,))
    randoms_noise = np.random.uniform(-5, 0, size=(nums,))

    path = []
    for i in range(nums):
        path.append(speechs1[i] + ' ' + str(randoms[i])[:6] + ' ' + speechs2[i] + ' ' + str(-randoms[i])[:6] + ' ' + noises[i%len(noises)] + ' ' + str(randoms_noise[i])[:6])

    save_path(path, txt_file)


libri_path = '' # the absolute path for LibriSpeech data, you can download the corpus from 'http://www.openslr.org/12'
noise_path = '' # the absolute path for noise data, you can download the corpus from 'http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/HuCorpus.html'

create_path(root=libri_path + 'test-clean/', txt_file='2spks_test.txt')
create_path(root=libri_path + 'dev-clean/', txt_file='2spks_dev.txt')
create_path(root=libri_path + 'train-clean-100/', txt_file='2spks_train.txt')

create_path_noisy(root=libri_path + 'test-clean/', noise_root=noise_path, txt_file='2spks_test_noisy.txt')
create_path_noisy(root=libri_path + 'dev-clean/', noise_root=noise_path, txt_file='2spks_dev_noisy.txt')
create_path_noisy(root=libri_path + 'train-clean-100/', noise_root=noise_path, txt_file='2spks_train_noisy.txt')



