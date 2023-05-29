
arg1=$1
shift
if [ "$arg1" == "centralized" ]; then
	echo "Running centralized training"
	python ./baselines/centrlized.py "$@"

elif [ "$arg1" == "fedavg" ]; then
	echo "Running FL with FedAvg"
	python ./baselines/fedavg__standard.py "$@"

elif [ "$arg1" == "confidence_learning" ]; then
	echo "Running FL with confidence learning"
	python ./baselines/fedavg__cl.py "$@"

elif [ "$arg1" == "bi_tempered_loss" ]; then
	echo "Running FL with Bi-Tempered loss"
	python ./baselines/fedavg__loss.py "$@"

elif [ "$arg1" == "label_smoothing" ]; then
	echo "Running FL with label smoothing"
	python ./baselines/fedavg__smooth.py "$@"

elif [ "$arg1" == "nnc" ]; then
	echo "Running FL with nearest neighbor correction"
	python ./fedln/fedln__nnc.py "$@"

elif [ "$arg1" == "na_fedavg" ]; then
	echo "Running FL with nearest Noise-Aware FedAvg"
	python ./fedln/fedln__nafedavg.py "$@"

elif [ "$arg1" == "akd" ]; then
	echo "Running FL with Adaptive Knowledge Distillation"
	python ./fedln/fedln__akd.py "$@"
else
	echo "Invalid method provided."
fi
