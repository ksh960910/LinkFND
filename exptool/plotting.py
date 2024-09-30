import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

'''
Top-k, Threshold 각각의 impact를 보여주기 위한 plotting
'''

# Threshold
# x = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# y = [0.0464, 0.0469, 0.0471, 0.04723, 0.0474, 0.0403] # amazon
# # y = [0.0726, 0.0727, 0.0728, 0.0731, 0.0733, 0.0721] # yelp
# # y = [0.0776, 0.07805, 0.0782, 0.07827, 0.0786, 0.0717] #ali
# index = np.arange(len(x))
# # line plot을 그립니다.
# plt.plot(index, y, color='forestgreen', marker='o', linewidth=1, markersize=3)
# plt.plot(4, y[4], color='red', marker='s', linewidth=1, markersize=5, label='SimGCL+LinkFND')
# plt.plot(5, y[5], color='blue', marker='s', linewidth=1, markersize=5, label='SimGCL')

# plt.xlabel('threhold')
# plt.ylabel('Recall@k')
# plt.xticks(index, x)
# plt.yticks(np.arange(0.0403,0.0475,0.001)) # amazon
# # plt.yticks(np.arange(0.0720,0.0734,0.0004)) # yelp
# # plt.yticks(np.arange(0.0715,0.0787,0.0008)) # ali
# plt.tight_layout()
# plt.legend()
# plt.grid(True)

# # line plot을 png 파일로 저장합니다.
# plt.savefig('amazon'+'_threshold.png', dpi=300, bbox_inches='tight', pad_inches=0)

# Top-k
# x = [0, 1, 2, 4, 8]
# # y = [0.0403, 0.0471, 0.0473, 0.0470, 0.0464] # amazon
# # y = [0.0717, 0.0780, 0.0783, 0.0781, 0.0772] # ali
# y = [0.0721, 0.0730, 0.0733, 0.0731, 0.0726] # yelp
# index = np.arange(len(x))
# # line plot을 그립니다.
# plt.plot(index, y, color='indigo', marker='o', linewidth=1, markersize=3)
# plt.plot(2, y[2], color='red', marker='s', linewidth=1, markersize=5, label='SimGCL+LinkFND')
# plt.plot(0, y[0], color='blue', marker='s', linewidth=1, markersize=5, label='SimGCL')
# plt.xlabel('Top-k')
# plt.ylabel('Recall@k')
# plt.xticks(index, x)
# # plt.yticks(np.arange(0.0403, 0.0475, 0.001)) #amazon
# # plt.yticks(np.arange(0.0715,0.0787,0.0008)) #ali
# plt.yticks(np.arange(0.0720, 0.0734, 0.0004)) #yelp
# plt.tight_layout()
# plt.legend()
# plt.grid(True)

# # line plot을 png 파일로 저장합니다.
# plt.savefig('yelp'+'_topk.png', dpi=300, bbox_inches='tight', pad_inches=0)



'''
twiny 사용
'''
# Threshold
x_1 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# y_1 = [0.0464, 0.0469, 0.0471, 0.04723, 0.0485, 0.0403] # amazon
# y_1 = [0.0726, 0.0727, 0.0728, 0.0731, 0.0733, 0.0721] # yelp
y_1 = [0.0776, 0.07805, 0.0782, 0.07827, 0.0784, 0.0717] #ali
# Topk
x_2 = [0, 1, 2, 4, 8]
# y_2 = [0.0403, 0.0471, 0.0485, 0.0473, 0.0464] # amazon
# y_2 = [0.0721, 0.0730, 0.0733, 0.0731, 0.0726] # yelp
y_2 = [0.0717, 0.0780, 0.0784, 0.0781, 0.0772] # ali
index_1 = np.arange(len(x_1))
index_2 = np.arange(len(x_2))
# line plot을 그립니다.
plt.plot(index_1, y_1, color='saddlebrown', marker='o', linewidth=3, markersize=6)
plt.plot(4, y_1[4], color='palevioletred', marker='s', linewidth=1, markersize=6, label='SimGCL+LinkFND')
plt.plot(5, y_1[5], color='navy', marker='s', linewidth=1, markersize=6, label='SimGCL')
plt.xlabel('Threshold', color='saddlebrown', fontsize=12)
plt.xticks(index_1, x_1, color='saddlebrown', fontsize=12)


y_up = plt.twiny()
y_up.plot(index_2, y_2, color='teal', marker='o', linewidth=3, markersize=6)
y_up.plot(2, y_2[2], color='palevioletred', marker='s', linewidth=1, markersize=6, label='SimGCL+LinkFND')
y_up.plot(0, y_2[0], color='navy', marker='s', linewidth=1, markersize=6, label='SimGCL')
plt.xlabel('Top-k', color='teal', fontsize=12)
plt.xticks(index_2, x_2, color='teal', fontsize=12)

plt.ylabel('Recall@k')
# plt.yticks(np.arange(0.0403,0.0494,0.0013)) # amazon
# plt.yticks(np.arange(0.0720,0.0734,0.0004)) # yelp
plt.yticks(np.arange(0.0715,0.0787,0.0008)) # ali

# Topk
# plt.yticks(np.arange(0.0403, 0.0475, 0.001)) #amazon
# plt.yticks(np.arange(0.0715,0.0787,0.0008)) #ali
# plt.yticks(np.arange(0.0720, 0.0734, 0.0004)) #yelp
plt.tight_layout()
plt.legend(fontsize=12, loc='lower center')
# plt.grid(True)

# line plot을 png 파일로 저장합니다.
plt.savefig('ali'+'twiny.png', dpi=300, bbox_inches='tight')

# Top-k
# x = [0, 1, 2, 4, 8]
# # y = [0.0403, 0.0471, 0.0473, 0.0470, 0.0464] # amazon
# # y = [0.0717, 0.0780, 0.0783, 0.0781, 0.0772] # ali
# y = [0.0721, 0.0730, 0.0733, 0.0731, 0.0726] # yelp
# index = np.arange(len(x))
# # line plot을 그립니다.
# plt.plot(index, y, color='indigo', marker='o', linewidth=1, markersize=3)
# plt.plot(2, y[2], color='red', marker='s', linewidth=1, markersize=5, label='SimGCL+LinkFND')
# plt.plot(0, y[0], color='blue', marker='s', linewidth=1, markersize=5, label='SimGCL')
# plt.xlabel('Top-k')
# plt.ylabel('Recall@k')
# plt.xticks(index, x)
# # plt.yticks(np.arange(0.0403, 0.0475, 0.001)) #amazon
# # plt.yticks(np.arange(0.0715,0.0787,0.0008)) #ali
# plt.yticks(np.arange(0.0720, 0.0734, 0.0004)) #yelp
# plt.tight_layout()
# plt.legend()
# plt.grid(True)

# # line plot을 png 파일로 저장합니다.
# plt.savefig('yelp'+'_topk.png', dpi=300, bbox_inches='tight', pad_inches=0)


'''
lambda, epsilon의 impact를 보여주기 위한 plotting
'''
# x = [0.05, 0.1, 0.2, 0.5, 1, 2, 5]

# # SimGCL - lambda
# # y = [0.0714, 0.0713, 0.0676, 0.0666, 0.0581, 0.0493, 0.0379] # ali
# # y = [0.0391, 0.0401, 0.0378, 0.0325, 0.0274, 0.0245, 0.0199] # amazon
# # y = [0.0695, 0.0714, 0.0717, 0.0721, 0.0696, 0.0669, 0.0581] # yelp

# # SimGCL - eps
# # y = [0.0689, 0.0695, 0.0714, 0.0710, 0.0717, 0.0701, 0.0669] # ali
# y = [0.0395, 0.0403, 0.0402, 0.0393, 0.0390, 0.0384, 0.0350] # amazon
# # y = [0.0716, 0.0719, 0.0721, 0.0712, 0.0707, 0.0686, 0.0653] # yelp

# # Ours - lambda
# # z = [0.0784, 0.0783, 0.0755, 0.0673, 0.0597, 0.0503, 0.0390] # ali
# # z = [0.0485, 0.0470, 0.0444, 0.0385, 0.0334, 0.0305, 0.0257] # amazon
# # z = [0.0710, 0.0721, 0.0731, 0.0706, 0.0644, 0.0573, 0.0572] # yelp

# # Ours - eps
# # z = [0.0764, 0.0783, 0.0774, 0.0765, 0.0749, 0.0721, 0.0664] # ali
# z = [0.0464, 0.0473, 0.0469, 0.0462, 0.0458, 0.0453, 0.0428] # amazon
# # z = [0.0728, 0.0734, 0.0730, 0.0721, 0.0707, 0.0689, 0.0655] # yelp


# index = np.arange(len(x))
# plt.figure(figsize=(6,8))
# # line plot을 그립니다.
# # plt.plot(index, y, color='slateblue', marker='o', linewidth=1, markersize=5, label='SimGCL')
# # plt.plot(index, z, color='indianred', marker='s', linewidth=1, markersize=5, label='SimGCL+LinkFND')
# plt.plot(index, y, color='seagreen', marker='o', linewidth=5, markersize=8, label='SimGCL')
# plt.plot(index, z, color='chocolate', marker='s', linewidth=5, markersize=8, label='SimGCL+LinkFND')
# # plt.xlabel('Top-k')
# plt.title('Amazon', fontsize=35, fontweight='bold')
# plt.ylabel('Recall@k', fontsize=25)
# plt.xticks(fontsize=23, rotation=45)
# plt.xticks(index, x)
# plt.yticks(fontsize=23)
# # lambda
# # plt.yticks(np.arange(0.0370,0.0785,0.01)) #ali
# # plt.yticks(np.arange(0.0190, 0.0490, 0.005)) #amazon
# # plt.yticks(np.arange(0.0570, 0.0735, 0.005)) #yelp

# # eps
# # plt.yticks(np.arange(0.0660,0.0785,0.002)) #ali
# plt.yticks(np.arange(0.0350, 0.0475, 0.003)) #amazon
# # plt.yticks(np.arange(0.0653, 0.0735, 0.0015)) #yelp
# plt.tight_layout()
# plt.legend(fontsize=21)
# plt.grid(True, linestyle='--', alpha=0.5)

# # line plot을 png 파일로 저장합니다.
# # plt.savefig('yelp'+'_compare_lambda.png', dpi=300, bbox_inches='tight', pad_inches=0)
# plt.savefig('amazon'+'_compare_epsilon.png', dpi=300, bbox_inches='tight', pad_inches=0)