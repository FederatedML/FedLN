# FedLN: Federated Learning with Label Noise

Federated Learning (FL) is a distributed machine learning paradigm that enables learning models from decentralized private datasets while keeping the data on users' devices. However, most existing FL approaches assume high-quality labels are readily available, which is often not the case in real-world scenarios. In this paper, we propose FedLN, a framework for addressing label noise in different FL training stages, including initialization, on-device model training, and server model aggregation. FedLN computes per-client noise-level estimation in a single federated round and improves the models' performance by correcting or limiting the effect of noisy samples. Extensive experiments on various publicly available vision and audio datasets demonstrate a 24\% improvement on average compared to other existing methods for a label noise level of 70\%. We further validate the efficiency of FedLN in human-annotated real-world noisy datasets and report a 9\% increase on average in models' recognition performance, emphasizing that FedLN can be useful for improving FL services provided to everyday users.

A complete description of our work can be found in [our paper](https://arxiv.org/abs/2208.09378).

```
git clone https://github.com/FederatedML/FedLN
cd FedLN/src
```

## Dependencies
Create a new Python enviroment (virtualenvs, anacoda, etc.) and install all required packages via:
```console
foo@bar:~$ pip install -r requirements.txt
```

## Executing experiments
From the `src` (root) directory of this repo, run:

```console
# FedLN-NNC
foo@bar:~$ ./run.sh nnc --dataset_name cifar10 --model_name resnet20 --num_rounds 100 --num_clients 10 --noisy_frac 0.8 --noise_level 0.4 --noise_sparsity 0.7
# FedLN-NA_FedAvg
foo@bar:~$ ./run.sh na_fedavg --dataset_name cifar10 --model_name resnet20 --num_rounds 100 --num_clients 10 --noisy_frac 0.8 --noise_level 0.4 --noise_sparsity 0.7
# FedLN-AKD
foo@bar:~$ ./run.sh akd --dataset_name cifar10 --model_name resnet20 --num_rounds 100 --num_clients 10 --noisy_frac 0.8 --noise_level 0.4 --noise_sparsity 0.7
```

To execute any of the baselines, run:
```console
# FedAvg
foo@bar:~$ ./run.sh fedavg --dataset_name cifar10 --model_name resnet20 --num_rounds 100 --num_clients 10 --noisy_frac 0.8 --noise_level 0.4 --noise_sparsity 0.7
# FedAvg + Bi-Tempered Logistic Loss
foo@bar:~$ ./run.sh bi_tempered_loss --dataset_name cifar10 --model_name resnet20 --num_rounds 100 --num_clients 10 --noisy_frac 0.8 --noise_level 0.4 --noise_sparsity 0.7
# FedAvg + Label Smoothing
foo@bar:~$ ./run.sh label_smoothing --dataset_name cifar10 --model_name resnet20 --num_rounds 100 --num_clients 10 --noisy_frac 0.8 --noise_level 0.4 --noise_sparsity 0.7
# FedAvg + Confidence Learning
foo@bar:~$ ./run.sh confidence_learning --dataset_name cifar10 --model_name resnet20 --num_rounds 100 --num_clients 10 --noisy_frac 0.8 --noise_level 0.4 --noise_sparsity 0.7
```

You can configure all federated parameters (i.e. number of federated rounds, number of clients, percentage of noisy data, dataset, model, etc.,) by passing them as command line arguments.

## Reference
If you use this repository, please consider citing:

<pre>@misc{tsouvalas2022federated,
	title={Federated Learning with Noisy Labels}, 
	author={Vasileios Tsouvalas and Aaqib Saeed and Tanir Ozcelebi and Nirvana Meratnia},
	year={2022},
	eprint={2208.09378},
	archivePrefix={arXiv},
	primaryClass={cs.LG}
}
</pre>

<pre>@inproceedings{tsouvalas2022federatedworkshop,
	title={Federated Learning with Noisy Labels: Achieving Generalization in the Face of Label Noise},
	author={Vasileios Tsouvalas and Aaqib Saeed and Tanir {\"O}z{\c{c}}elebi and Nirvana Meratnia},
	booktitle={First Workshop on Interpolation Regularizers and Beyond at NeurIPS 2022},
	year={2022},
	url={https://openreview.net/forum?id=gNHMC4I0Pva}
}
</pre>
