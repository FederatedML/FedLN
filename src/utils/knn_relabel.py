import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def relabel_with_pretrained_knn(labels, features, num_classes, weights='uniform', num_neighbors=10, noise_theshold=0.15, verbose=False):
	# Initialize
	_labels = np.squeeze(np.argmax(labels, axis=-1)).astype(np.int64)
	knn = KNeighborsClassifier(n_neighbors=num_neighbors, weights=weights, n_jobs=1)
	knn.fit(features, _labels)
	knn.classes_ = np.arange(num_classes)
	predictions = np.squeeze(knn.predict(features).astype(np.int64))
	# Estimate label noise
	est_noise_lvl = (predictions!=_labels).astype(np.int64).mean()
	if verbose: print(f"Estimated noise level {100*est_noise_lvl:.02f}%")
	return np.take(np.eye(num_classes), predictions, axis=0).astype(np.float32) if est_noise_lvl>=noise_theshold else labels

def estimate_noise_with_pretrained_knn(labels, features, num_classes, weights='uniform', num_neighbors=10, noise_theshold=0.15, verbose=False):
	# Initialize
	labels = np.squeeze(np.argmax(labels, axis=-1)).astype(np.int64)
	knn = KNeighborsClassifier(n_neighbors=num_neighbors, weights=weights, n_jobs=1)
	knn.fit(features, labels)
	knn.classes_ = np.arange(num_classes)
	predictions = np.squeeze(knn.predict(features).astype(np.int64))
	est_noise_lvl = (predictions!=labels).astype(np.int64).mean()
	est_noise_lvl = est_noise_lvl if est_noise_lvl>=noise_theshold else 0.0
	if verbose: print(f"Estimated noise level {100*est_noise_lvl:.02f}%")
	return est_noise_lvl
