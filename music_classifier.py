# -*- coding: UTF-8 -*-

from scipy.io import wavfile
from scipy import fft
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import pprint


def create_fft(g, n):
    rad = '../genres/' + g + '/converted/' + g + '.' + str(n).zfill(5) + '.au.wav'
    print rad
    (sample_rate, x) = wavfile.read(rad)
    fft_features = abs(fft(x)[:2000])
    sad = '../fft_text/' + g + '.' + str(n).zfill(5) + '.fft'
    np.save(sad, fft_features)


genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
# for g in genre_list:
# for n in range(100):
# create_fft(g, n)

x = []
y = []
for g in genre_list:
    for n in range(2):
        rad = '../trainset/' + g + '.' + str(n).zfill(5) + '.fft.npy'
        fft_features = np.load(rad)
        x.append(fft_features)
        y.append(genre_list.index(g))
x = np.array(x)
y = np.array(y)

model = LogisticRegression()
model.fit(x, y)
# 存储训练的模型
# output = open('data.pkl', 'wb')
# pickle.dump(model, output)
# output.close()
# 使用存储的模型
pll_file = open('data.pkl', 'rb')
model_load = pickle.load(pll_file)
# pprint.pprint(model_load)
pll_file.close()

# 音乐分类
print 'Starting read wavfile...'
samle_rate, test = wavfile.read('Goldmund.wav')
test_fft_features = abs(fft(test)[:1000])
type_index = model_load.predict([test_fft_features])[0]
print genre_list[type_index]
