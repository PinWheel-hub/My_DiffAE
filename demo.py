from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import numpy as np

# 步骤 1: 生成模拟数据
# 生成正常样本
X_normal, _ = make_blobs(n_samples=300, centers=[[0, 0]], cluster_std=1.0)
# 生成异常样本
X_abnormal, _ = make_blobs(n_samples=100, centers=[[5, 5]], cluster_std=1.5)

print(X_normal.shape, X_abnormal.shape)
# 步骤 2: 训练GMM
# 对正常样本训练GMM
gmm_normal = GaussianMixture(n_components=1, random_state=42)
gmm_normal.fit(X_normal)

# 对异常样本训练GMM
gmm_abnormal = GaussianMixture(n_components=1, random_state=42)
gmm_abnormal.fit(X_abnormal)

# 步骤 3: 评估新样本
# 假设有一个新样本
new_sample = np.array([[2, 2]])

# 计算新样本属于正常GMM的对数似然
log_likelihood_normal = gmm_normal.score_samples(new_sample)
# 计算新样本属于异常GMM的对数似然
log_likelihood_abnormal = gmm_abnormal.score_samples(new_sample)

# 对数似然差作为异常分数（正常分数 - 异常分数）
log_likelihood_diff = log_likelihood_normal - log_likelihood_abnormal

print(log_likelihood_normal, log_likelihood_abnormal, log_likelihood_diff)
