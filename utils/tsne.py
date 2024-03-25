import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 假设features是一个形状为(1000, 50)的数组，表示1000个特征，每个特征50维
# 这里生成一个示例数据，实际应用中应替换为真实数据
features = np.random.randn(10000, 500)

# 使用t-SNE进行降维处理，将50维数据降到2维以便于可视化
tsne = TSNE(n_components=2, random_state=0)
features_2d = tsne.fit_transform(features)

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(features_2d[:, 0], features_2d[:, 1], c='blue', s=20)
plt.title('t-SNE Visualization of Features')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig('tsne_visualization.png', dpi=300)
