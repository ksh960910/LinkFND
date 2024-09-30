import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import random
from sklearn.preprocessing import normalize
import seaborn as sns
from math import pi

"""Show T-SNE"""
user_embedding = np.load('../embedding/'+'amazon-book_eps0.2_lamb1'+'_user.npy')
item_embedding = np.load('../embedding/'+'amazon-book_eps0.2_lamb1'+'_item.npy')
full_embedding = np.concatenate((user_embedding, item_embedding), axis=0)
random_embeddings_indices = np.random.choice(len(full_embedding), 2000)
random_embeddings = full_embedding[random_embeddings_indices]
# Create a TSNE object
# tsne = TSNE(n_components=2, perplexity=30, early_exaggeration=50, learning_rate=100, n_iter=2000)
tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=501)

# Fit the TSNE object to the random embeddings
tsne_embedding = tsne.fit_transform(random_embeddings)
tsne_embedding = normalize(tsne_embedding, axis=1, norm='l2')

# sns.set_style("white")

f, axs = plt.subplots(2, 1, figsize=(16,4.5),gridspec_kw={'height_ratios': [3, 1]})
kwargs = {'levels': np.arange(0, 5.5, 0.5)}
sns.kdeplot(data=tsne_embedding, bw=0.05, shade=True, cmap="GnBu", legend=True, ax=axs[0], **kwargs)
axs[0].set_title('SimGCL FN', fontsize=9, fontweight="bold")
x = [p[0] for p in tsne_embedding]
y = [p[1] for p in tsne_embedding]
angles = np.arctan2(y, x)
sns.kdeplot(data=angles, bw=0.15, shade=True, legend=True, ax=axs[1], color='green')

axs[0].tick_params(axis='x', labelsize=8)
axs[0].tick_params(axis='y', labelsize=8)
axs[0].patch.set_facecolor('white')
axs[0].collections[0].set_alpha(0)
axs[0].set_xlim(-1.2, 1.2)
axs[0].set_ylim(-1.2, 1.2)
axs[0].set_xlabel('Features', fontsize=9)
axs[0].set_ylabel('Features', fontsize=9)

axs[1].tick_params(axis='x', labelsize=8)
axs[1].tick_params(axis='y', labelsize=8)
axs[1].set_xlabel('Angles', fontsize=9)
axs[1].set_ylim(0, 0.5)
axs[1].set_xlim(-pi, pi)
axs[1].set_ylabel('Density', fontsize=9)
plt.savefig('KDEBOOK.png', dpi=500)

# plt.figure(figsize=(16,14))
# # Plot the TSNE embedding
# plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], s=50)
# # plt.savefig('amazon-book'+'_tsne_embedding.png', dpi=300)
# plt.savefig('NormGOWALLA.png', dpi=300)

# # Show the plot
# plt.show()