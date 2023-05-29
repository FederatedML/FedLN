import numpy as np

def shard_data(data, id=0, num_shards=10, seed=42):
	np.random.seed(seed)
	data_distribution = np.random.choice(a=np.arange(0,num_shards), size=len(data[0])).astype(int)
	data_distribution_mask = np.squeeze([data_distribution==id])
	data_distribution_idxs = np.argwhere((data_distribution_mask==True))
	return tuple(data[i][data_distribution_mask] for i in range(len(data))), data_distribution_idxs

def shard_records(data, id=0, num_shards=10, seed=42):
	np.random.seed(seed)
	np.random.shuffle(data[0])
	data_distribution = np.full(shape=(num_shards,), fill_value=(len(data[0])//num_shards)) # Fill minimum values
	for i in np.random.choice(a=np.arange(0, num_shards), size=(len(data[0])%num_shards)): # Add integer division remaining
		data_distribution[i]+=1
	data_distribution = np.cumsum(data_distribution)
	data_distribution_mask = [np.arange(0 if idx==0 else data_distribution[idx-1],i).tolist() for idx,i in enumerate(data_distribution)]
	distribution = [[data[0][j] for j in ci] for ci in data_distribution_mask]
	return tuple(distribution[id],)