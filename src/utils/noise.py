import numpy as np

def noise_from_level_sparsity(labels, num_classes, noise_lvl, noise_spar, noise_seed, as_one_hot=True):

	if (noise_lvl==0.0) and (noise_spar==0.0): return labels, 0.0
	if as_one_hot: labels = np.squeeze(np.argmax(labels,-1))

	# Construct "noise" matrix
	def _compute_noise_matrix(num_classes, noise_lvl, noise_spar, seed):
		np.random.seed(seed)
		num_noise_classes = np.maximum( int(np.ceil((num_classes-1)*(1.0-noise_spar))), 1)
		noise_classes = {c: [c] + np.random.choice(a=[i for i in range(num_classes) if i!=c], replace=False, size=num_noise_classes).tolist() for c in range(num_classes)}
		noise_prob_per_class = np.array([1-noise_lvl,] + [noise_lvl/num_noise_classes]*num_noise_classes)
		return noise_classes, noise_prob_per_class

	noise_classes, noise_prob_per_class = _compute_noise_matrix(num_classes=num_classes, noise_lvl=noise_lvl, noise_spar=noise_spar, seed=noise_seed)
	noisy_labels = np.full_like(labels,fill_value=-1)

	for c in range(num_classes):
		noisy_labels[np.where(labels==c)[0]] = np.random.choice(a=noise_classes[c], p=noise_prob_per_class, size=np.where(labels==c)[0].shape[0])
	achieved_noise_lvl = (noisy_labels!=labels).mean(axis=-1)

	# Reverse to categorical
	if as_one_hot: noisy_labels = np.eye(num_classes)[noisy_labels]

	return noisy_labels, achieved_noise_lvl

def inject_synthetic_noise_to_labels(cid, labels, num_classes, level, sparsity, seed, theshold=1e-2, as_one_hot=True):
	noisy_labels, achieved_noise_lvl = noise_from_level_sparsity(labels=labels, num_classes=num_classes, as_one_hot=as_one_hot, noise_lvl=level, noise_spar=sparsity, noise_seed=seed)
	print(f"[Client {cid}] - Achived noise level is {achieved_noise_lvl:.04f} and desired noise level is {level:.04f}.")
	assert np.abs(achieved_noise_lvl-level) < theshold, f'Mean noise level is not reached. Got {achieved_noise_lvl:.04f}, while desired levels set to {level:.04f}. Set a higher threshold!'
	return noisy_labels