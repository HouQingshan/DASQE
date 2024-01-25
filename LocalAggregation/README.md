# Kmeans clustering and Visualization for Unsupervised Learning of Visual Embeddings

This is a Pytorch implementation of the Kmeans clustering algorithm for Unsupervised Learning of Visual Embeddings


# Usage

### Prerequisites

* Ubuntu 16.04
* Pytorch 1.2.0
* [Faiss==1.6.1](https://github.com/facebookresearch/faiss)
* tqdm
* dotmap
* tensorboardX

### Runtime Setup
```
source init_env.sh
```

### Dataset preparation
Prepare the dataset and change the "DIR_LIST" parameter in ./src/datasets/imagenet.py.

### Model training

This implementation currently supports LA trained ResNets. 
As LA algorithm requires training the model using IR algorithm for 10 epochs as a warm start, we first run the IR training using the following command:
```
CUDA_VISIBLE_DEVICES=0 python instance.py ./config/imagenet_ir.json
```
Then specify `instance_exp_dir` in `./config/imagenet_la.json` and run the following command to do the LA training:
```
CUDA_VISIBLE_DEVICES=0 python localagg.py ./config/imagenet_la.json
```
By default, both IR and LA are trained using a single GPU. Multi-gpu training is also supported in this implementation.
