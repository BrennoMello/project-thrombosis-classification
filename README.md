# Thrombosis-Classification
## How to run: 

- `app/experiment_synclass.py`: pipeline with synclass algorithm
- `app/experiment.py`: pipeline with outlier detection 


## Results dataset V2


|              Group 1              |              Group 2               | Accuracy (class 1) | Accuracy (class 2) |  F1  | Classifier | Test Size |
| :-------------------------------: | :--------------------------------: | :----------------: | :----------------: | :--: | :--------: | --------- |
|     Thrombosis <br />(n=136)      |     Non-Thrombosis<br/>(n=283)     |        0.68        |        0.60        | 0.63 |    SVM     | 14        |
|         Type I<br/>(n=96)         | Type II (PE + HBS + RS)<br/>(n=40) |        0.65        |        0.66        | 0.60 |    KNN     | 5         |
|         Type I<br/>(n=96)         |   Type II (HBS + RS)<br/>(n=22)    |        0.76        |        0.73        | 0.67 |     RF     | 5         |
| Type I + Type II (PE)<br/>(n=114) |   Type II (HBS + RS)<br/>(n=22)    |        0.72        |        0.62        | 0.63 |    SVM     | 5         |



## Results Smote 



|              Group 1              |              Group 2               | Accuracy (class 1) | Accuracy (class 2) |  F1  | Classifier | Folds |
| :-------------------------------: | :--------------------------------: | :----------------: | :----------------: | :--: | :--------: | ----- |
|     Thrombosis <br />(n=136)      |     Non-Thrombosis<br/>(n=283)     |        0.68        |        0.60        | 0.63 |    SVM     | 10    |
|         Type I<br/>(n=96)         | Type II (PE + HBS + RS)<br/>(n=40) |        0.59        |        0.61        | 0.55 |     RF     | 10    |
|         Type I<br/>(n=96)         |   Type II (HBS + RS)<br/>(n=22)    |        0.76        |        0.61        | 0.66 |    SVM     | 10    |
| Type I + Type II (PE)<br/>(n=114) |   Type II (HBS + RS)<br/>(n=22)    |        0.75        |        0.61        | 0.66 |    SVM     | 5     |

