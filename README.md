# Visual Contrastive Explanation
This repository contains the code for the paper "VAE-CE: Visual Contrastive Explanation using Disentangled VAEs" [[IDA 2022]](https://link.springer.com/chapter/10.1007/978-3-031-01333-1_19), [[arxiv]](https://arxiv.org/abs/2108.09159). 

### Installation
Required packages are listed in `requirements.txt`. The code was developed/tested using Python 3.7.3.

### Training and testing
To train and evaluate methods you can use the supplied training scripts:
```
python train.py --type cd --epoch 50 --save True --save_name cd
python train.py --type vaece --cd_name cd --epoch 20 --save True --save_name vaece
python evaluate.py --metric_name metric --type vaece --load_name vaece --cd_name cd
```

### Data
Data can be generated by running the corresponding scripts in `data/`:
```
python -m data.synthetic.generate
python -m data.mnist.generate
```
The synthetic (training and testing) data as used in the paper can be downloaded from [here](https://surfdrive.surf.nl/files/index.php/s/hA6w0OAtJnrV229/download).

### Implementations
The implementations of all models are provided in `model/`: Shared components in `component.py`, with the training procedures in `r_model.py` and `cd_model.py`. 

Explanation generation and the explanation alignment cost (*eac*) computation are provided in `explanation/`: `explanation.py` and `evaluation.py`.

The evaluation of all metrics is provided by MetricComputation in `testing/metric.py`.

---
If you found our work useful in your research, please consider citing:
```
@inproceedings{poels2022vaece,
  title={{VAE-CE}: Visual Contrastive Explanation Using Disentangled {VAEs}},
  author={Poels, Yoeri and Menkovski, Vlado},
  booktitle={Advances in Intelligent Data Analysis XX},
  pages={237--250},
  year={2022}
}
```
