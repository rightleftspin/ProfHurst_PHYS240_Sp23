import numpy as np

scores = 100 * np.random.rand(int(1e7), 14)

m, n = scores.shape
scores_low = np.where(np.arange(n-1) < scores.argmin(axis=1)[:, None], scores[:, :-1], scores[:, 1:]).sum(axis = 1) / 13
scores_div = scores.sum(axis = 1) / 13

print(scores_div.mean())
print(scores_low.mean())

