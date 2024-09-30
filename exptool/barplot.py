import matplotlib.pyplot as plt
import numpy as np

# colors = ['red', 'blue','aqua', 'pink', 'green']
# w = 0.2
# idx = np.arange(5)
# # 데이터셋
# # top-k
# onlytopk = [0.0721, 0.0725, 0.0729, 0.0724, 0.0722]
# ours = [0.0721, 0.0730, 0.0733, 0.0731, 0.0726]
# param = ['0', '1', '2', '4', '8']

# plt.figure(figsize=(10,5))
# plt.title('Yelp2018', fontsize=23, fontweight='bold')
# plt.xlabel('Top-k', fontsize=20)
# plt.ylabel('Recall@k', fontsize=20)
# plt.bar(idx[1:] - w, onlytopk[1:], color='darkcyan', width=0.4, label='Top-k')
# plt.bar(idx[1:] + w, ours[1:], color='salmon', width=0.4, label='Top-k + Threshold')
# plt.bar(idx[0], onlytopk[0], color='navajowhite', width=0.6, hatch='///', label='SimGCL')
# # plt.bar(idx[0] + w, ours[0], color='salmon', width=0.4, hatch='///', label='SimGCL')
# plt.xticks(idx, param)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=15)
# plt.ylim(0.0715, 0.0734)
# plt.legend()
# plt.show()

# plt.savefig('yelp'+'_topk_both.png', dpi=300, bbox_inches='tight', pad_inches=0)

# threshold
w = 0.2
idx = np.arange(4)
# 데이터셋
# top-k
onlyth = [0.0703, 0.0726, 0.0729, 0.0721]
ours = [0.0726, 0.0728, 0.0733, 0.0721]
param = ['0.5', '0.7', '0.9', '1']

plt.figure(figsize=(10,5))
plt.title('Yelp2018', fontsize=23, fontweight='bold')
plt.xlabel('Threshold', fontsize=20)
plt.ylabel('Recall@k', fontsize=20)
plt.bar(idx[:-1] - w, onlyth[:-1], color='darkcyan', width=0.4, label='Threshold')
plt.bar(idx[:-1] + w, ours[:-1], color='salmon', width=0.4, label='Top-k + Threshold')
plt.bar(idx[-1], onlyth[-1], color='navajowhite', width=0.6, hatch='///', label='SimGCL')
plt.xticks(idx, param)
plt.xticks(fontsize=20)
plt.yticks(fontsize=15)
plt.ylim(0.069, 0.0734)
plt.legend()
plt.show()

plt.savefig('yelp'+'_threshold_both.png', dpi=300, bbox_inches='tight', pad_inches=0)
