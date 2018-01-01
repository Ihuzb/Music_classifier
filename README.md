# Music_classifier
使用python基于逻辑回归开发的音乐分类器
# 代码清单
music_classifier.py 其中主要功能由音频转文本，训练音乐分类模型和测试模型
data.pkl 保存的训练模型文件
trainset 音频文本文件夹
# 技术简介
音频通过傅里叶转换成频率矩阵，取前1000Hz数据存储<br>
采用Python内建的持久性模型 pickle 来保存通过逻辑回归训练的模型