# LinkFND : Simple Framework for False Negative Detection in Recommendation Tasks With Graph Contrastive Learning

This is our PyTorch implementation for the paper:

> Kim, Sanghun, and Hyeryung Jang. "LinkFND: Simple Framework for False Negative Detection in Recommendation Tasks With Graph Contrastive Learning." IEEE Access (2023).  [Paper link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10368024)

Author: Mr. Sanghun Kim

## Citation 

If you want to use our codes in your research, please cite:
​    
```
@article{kim2023linkfnd,
  title={LinkFND: Simple Framework for False Negative Detection in Recommendation Tasks With Graph Contrastive Learning},
  author={Kim, Sanghun and Jang, Hyeryung},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE}
}
```

## Our Proposed Overall Architecture
<p align="center">
 <img src = "./Overall Architecture.png">
</p>

## Environment Requirement

The code has been tested running under Python 3.7.6. The required packages are as follows:

- pytorch == 1.7.0
- numpy == 1.20.2
- scipy == 1.6.3
- sklearn == 0.24.1
- prettytable == 2.1.0

## Training

The instruction of commands has been clearly stated in the codes (see the parser function in utils/parser.py). Important argument:

- `gnn`
  - It specifies the graph based recommender models when training.
- `fnk`
  - It specifies the number of false negatives to be selected by top-k strategy
- `threshold`
  - It specifies the value of threshold to be selected by threshold strategy
- 


#### SimGCL without LinkFND

```
python main.py --dataset gowalla --gnn sgcl --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 -

python main.py --dataset amazon-book --gnn sgcl --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3

python main.py --dataset amazon --gnn sgcl --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3

python main.py --dataset yelp2018 --gnn sgcl --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3

python main.py --dataset ali --gnn sgcl --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3
```


#### SimGCL with LinkFND

```
python main.py --dataset gowalla --gnn linkfnd --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --tau 0.2 --lamb 0.2 --eps 0.15

python main.py --dataset amazon-book --gnn linkfnd --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --tau 0.2 --lamb 1 --eps 0.2

python main.py --dataset amazon --gnn linkfnd --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --tau 0.2 --lamb 0.1 --eps 0.15

python main.py --dataset yelp2018 --gnn linkfnd --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --tau 0.2 --lamb 0.2 --eps 0.15

python main.py --dataset ali --gnn linkfnd --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --tau 0.2 --lamb 0.1 --eps 0.15
```

## Dataset

We use three processed datasets: Alibaba, Yelp2018, Amazon, Gowalla, and Amazon-Book.

|               | Alibaba | Yelp2018  | Amazon    | Gowalla   | Amazon-Book |
| ------------- | ------- | --------- | --------- | --------- | ----------- |
| #Users        | 106,042 | 31,668    | 192,403   | 29,858    | 52,643      |
| #Items        | 53,591  | 38,048    | 63,001    | 40,981    | 91,599      |
| #Interactions | 907,407 | 1,561,406 | 1,689,188 | 1,027,370 | 2,984,108   |
| Density       | 0.00016 | 0.00130   | 0.00014   | 0.00084   | 0.00062     |

## Overall Performance Comparison

<p align="center">
 <img src = "./Performance Comparison.png">
</p>

