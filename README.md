# ACC_NAS
Accelerating multi-objective neural architecture search for remaining useful life prediction.  <br/>

## Abstract
Deep neural networks (DNNs) obtained remarkable achievements in remaining useful life (RUL) prediction of industrial components.
The architectures of these DNNs are usually determined empirically, usually with the goal of minimizing prediction error without considering
the time needed for training. However, such a design process is time-consuming as it is essentially based on trial-and-error. Moreover, this
process may be inappropriate in those industrial applications where the DNN model should take into account not only the prediction accuracy
but also the training computational cost. To address this challenge, we present a neural architecture search (NAS) technique based on an evolu-
tionary algorithm (EA) that explores the combinatorial parameter space of a one-dimensional convolutional neural network (1-D CNN) to search
for the best architectures in terms of a trade-off between RUL prediction error and number of trainable parameters. In particular, a novel way to
accelerate the NAS is introduced in this paper. We successfully shorten the lengthy training process by making use of two techniques, namely
architecture score without training and extrapolation of learning curves. We test our method on a recent benchmark dataset, the N-CMAPSS, on
which we search for trade-off solutions (in terms of prediction error vs. number of trainable parameters) using NAS. The results show that our
method considerably reduces the training time (and, as a consequence, the total time of the evolutionary search), yet successfully discovers ar-
chitectures compromising the two objectives.



## Note
```
H. Mo and G. Iacca, 
Accelerating Evolutionary Neural Architecture Search for Remaining Useful Life Prediction, 
In Bioinspired Optimization Methods and Their Applications: 10th International Conference, 
BIOMA 2022, Maribor, Slovenia, November 17–18, 2022, 
Proceedings, pp. 15-30. Cham: Springer International Publishing, 2022.
```

Bibtex entry ready to be cited
```
@inproceedings{mo2022accelerating,
  title={Accelerating Evolutionary Neural Architecture Search for Remaining Useful Life Prediction},
  author={Mo, Hyunho and Iacca, Giovanni},
  booktitle={Bioinspired Optimization Methods and Their Applications: 10th International Conference, BIOMA 2022, Maribor, Slovenia, November 17--18, 2022, Proceedings},
  pages={15--30},
  year={2022},
  organization={Springer}
}
```
