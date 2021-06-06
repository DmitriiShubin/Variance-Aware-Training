## About Explicit Variance Minimization: Training Neural Networks for Medical Imaging With Limited Data Annotations

This repository is the official implementation of the paper [About Explicit Variance Minimization: Training Neural Networks for Medical Imaging With Limited Data Annotations](https://arxiv.org/abs/2105.14117)

**Authors:** Dmitrii Shubin, Danny Eytan, Sebastian D. Goodfellow 
```
@article{Shubin2021AboutEV,
  title={About Explicit Variance Minimization: Training Neural Networks for Medical Imaging With Limited Data Annotations},
  author={Dmitrii Shubin and D. Eytan and S. Goodfellow},
  journal={ArXiv},
  year={2021},
  volume={abs/2105.14117}
}
```

### TL;DR
We propose a **Variance-Aware Training (VAT)**, the method that focuses on ***explicit*** minimization
of the variance of the model (i.e., as a part of the model loss), and providing matched or superior performance of
 self-supervised methods, but 10x faster.
 
 
![](https://github.com/DmitriiShubin/Variance-Aware-Training/assets/method.png "")

### Results
ACDC semantic segmentation (DICE marco score, without background):

| Method        | Xtr=2           | Xtr=4   |Xtr=8  | Xtr=60 (UB) |
| ------------- |:-------------:| :-----:|:-----:|:-----:|
| Baseline      | 0.700 | 0.766 | **0.834** | 0.902 |
| SimCLR      | 0.721 | 0.748  | 0.796| - |
| CP |0.743 | 0.790 | 0.799| - |
| Rotation |0.743 | 0.790 | 0.799| - |
| VAT, early agg (ours) | **0.749** | **0.814** | **0.834** | - |
| VAT, late agg (ours) |0.616 | 0.795 | 0.546| - |

APTOS 2019 ordinary regression, aka prediction of disease progression (Quadratic Weighted Cappa):

| Method        | Xtr=59           | Xtr=122   |Xtr=234  | Xtr=2343 (UB) |
| ------------- |:-------------:| :-----:|:-----:|:-----:|
| Baseline      | 0.804 | 0.851 | 0.852 | 0.909 |
| SimCLR      | 0.619 | 0.769 | 0.789| - |
| CP |0.822 | 0.847 | 0.849| - |
| Rotation |0.827 | 0.855 | 0.863| - |
| VAT, early agg (ours) | **0.851** | **0.863** | **0.868** | - |
| VAT, late agg (ours) |0.452 | 0.455 | 0.462 | - |

PCam classification (ROC-AUC score):

| Method        | Xtr=2           | Xtr=4   |Xtr=8  | Xtr=60 (UB) |
| ------------- |:-------------:| :-----:|:-----:|:-----:|
| Baseline      | 0.776 | 0.745 | 0.726 | 0.989 |
| SimCLR      | 0.781 | **0.817** | **0.860** | - |
| CP |0.706 | 0.794 | 0.814| - |
| Rotation |0.739 | 0.760 | 0.824 | - |
| VAT, early agg (ours) | **0.806** | 0.809 | 0.833 | - |
| VAT, late agg (ours) | 0.796 | 0.777 | 0.822 | - |

### Reproducing resuls

#### Environment installation

#### Training

#### Evaluation



