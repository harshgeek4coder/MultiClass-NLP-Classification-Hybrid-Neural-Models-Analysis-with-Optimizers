# Detailed Comparison of Hybrid Deep Neural Models and Optimizers for Multiclass classification in NLP

<p>Multi-class classification based on textual data refers to a supervised machine learning task, where we try to predict a certain class for input data, given the input data in raw textual format. Well, thanks to TensorFlow, we have a variety of algorithms available to perform multi-class classification via natural language processing.</p>

<p>Here , I have tried to perform a detailed analysis and comparison of various hybrid combination of deep neural networks and choosing different optimizer's impact on their learning and training of model and hence , their performance and time taken to train.</p>

#### Please have a look at <a href="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/full_main.py" >full_main.py</a> or <a href="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Final_MultiClass_NLP_Hybrid_Analysis_with_Optimizers.ipynb">IPYNB Notebook</a> for further reference. 

## CNN1-D:
### Model 1 - CNN-1D :

 <img  src= "https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/CNN1D/adam_loss_acc_comb.png" >
 

 ```
 With Adam : 

Acc: 93.71%
Precision: 0.94
Recall: 0.94 
F1 score: 0.94

Total time took for training 132.444 seconds.

Model Loss on training data:  0.000858851766679436 
Model Accuracy on training data:  1.0 
Model Loss on validation data:: 0.16400769352912903 
Model Accuracy on validation data:  0.9528089761734009
 ```

<img src="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/CNN1D/adagrad_loss_acc_comb.png">

```
With AdaGrad : 

Acc: 86.52% 
Precision: 0.87 
Recall: 0.87 
F1 score: 0.87

Total time took for training 42.859 seconds.

Model Loss on training data:  0.18429572880268097
Model Accuracy on training data:  0.9735954999923706
Model Loss on validation data 0.3753611147403717
Model Accuracy on validation data:  0.8876404762268066
```
<img src="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/CNN1D/sgd_acc_loss_comb.png">

```
With SGD : 

Acc: 85.17% 
Precision: 0.85
Recall: 0.85 
F1 score: 0.85

Total time took for training 49.209 seconds.

Model Loss on training data:  0.4065827429294586
Model Accuracy on training data:  0.9264044761657715
Model Loss on validation data 0.5402304530143738
Model Accuracy on validation data:  0.8471910357475281
```

<img src="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/CNN1D/rmsprop_acc_loss_comb.png">

```
With RMSProp : 

Acc: 92.58% 
Precision: 0.93 
Recall: 0.93 
F1 score: 0.93

Total time took for training 27.585 seconds.

Model Loss on training data:  2.9393253498710692e-05
Model Accuracy on training data:  1.0
Model Loss on validation data 0.4350118935108185
Model Accuracy on validation data:  0.934831440448761
```

<img src ="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/CNN1D/adadelta_comb.png">

```
With AdaDelta : 

Acc: 28.31% 
Precision: 0.28 
Recall: 0.28 
F1 score: 0.28

Total time took for training 23.972 seconds.

Model Loss on training data:  1.5702073574066162
Model Accuracy on training data:  0.3101123571395874
Model Loss on validation data 1.5865534543991089
Model Accuracy on validation data:  0.26966291666030884
```

### Model 2 - LSTM 

<img src ="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/LSTM/adam_comb.png" >

```
With Adam : 

Acc: 38.43% 
Precision: 0.38 
Recall: 0.38 
F1 score: 0.38

Total time took for training 119.479 seconds.

Model Loss on training data: 0.22766165435314178
Model Accuracy on training data:  0.8713483214378357
Model Loss on validation data 0.5319708585739136
Model Accuracy on validation data:  0.8157303333282471
```

<img src ="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/LSTM/adagrad_comb.png" >

```
With AdaGrad : 

Acc: 25.17% 
Precision: 0.25 
Recall: 0.25 
F1 score: 0.25

Total time took for training 32.487 seconds.

Model Loss on training data:  1.4589718580245972
Model Accuracy on training data:  0.4061797857284546
Model Loss on validation data 1.4928467273712158
Model Accuracy on validation data:  0.3685393333435058
```

<img src="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/LSTM/sgd_comb.png">

```
With SGD : 

Acc: 33.26% 
Precision: 0.33 
Recall: 0.33 
F1 score: 0.33

Total time took for training 36.672 seconds.

Model Loss on training data  1.4594911336898804
Model Accuracy on training data:  0.3292134702205658
Model Loss on validation data 1.4703514575958252
Model Accuracy on validation data:  0.33932584524154663
```

<img src="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/LSTM/rmsprop_comb.png">

```
With RMSProp : 

Acc: 92.81% 
Precision: 0.93 
Recall: 0.93 
F1 score: 0.93

Total time took for training 77.248 seconds.

Model Loss on training data  6.697150473078395e-11
Model Accuracy on training data:  1.0
Model Loss on validation data 0.7437273263931274
Model Accuracy on validation data:  0.9393258690834045
```

<img src ="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/LSTM/adadelta_comb.png">

```
With AdaDelta : 

Acc: 24.27% 
Precision: 0.24 
Recall: 0.24 
F1 score: 0.24

Total time took for training 11.980 seconds.

Model Loss on training data  1.5746873617172241
Model Accuracy on training data:  0.2550561726093292
Model Loss on validation data 1.575269103050232
Model Accuracy on validation data:  0.23146067559719086
```

### Model 3 - BiDirectional LSTM 

<img src ="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/Bi-LSTM/adam_comb.png" >

```
With Adam:

Acc: 94.61% 
Precision: 0.95 
Recall: 0.95 
F1 score: 0.95

Total time took for training 91.483 seconds.

Model Loss on training data  0.0007189955795183778
Model Accuracy on training data:  1.0
Model Loss on validation data 0.23885682225227356
Model Accuracy on validation data:  0.9460673928260803
```

<img src ="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/Bi-LSTM/adagrad_comb.png" >

```
With AdaGrad : 


Acc: 83.37% 
Precision: 0.83 
Recall: 0.83 
F1 score: 0.83

Total time took for training 99.641 seconds.

Model Loss on training data  0.37893444299697876
Model Accuracy on training data:  0.9168539047241211
Model Loss on validation data 0.5318742990493774
Model Accuracy on validation data:  0.833707869052887
```

<img src="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/Bi-LSTM/sgd_comb.png">

```
With SGD : 

Acc: 33.93% 
Precision: 0.34 
Recall: 0.34 
F1 score: 0.34

Total time took for training 66.537 seconds.

Model Loss on training data  1.4997886419296265
Model Accuracy on training data:  0.31067416071891785
Model Loss on validation data 1.504658579826355
Model Accuracy on validation data:  0.3280898928642273
```
<img src ="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/Bi-LSTM/rmsprop_comb.png">

```
With RMSProp : 

Acc: 92.81% 
Precision: 0.93 
Recall: 0.93 
F1 score: 0.93

Total time took for training 75.920 seconds.

Model Loss on training data  4.0584662741594e-08
Model Accuracy on training data:  1.0
Model Loss on validation data 0.4116312861442566
Model Accuracy on validation data:  0.934831440448761

```

<img src="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/Bi-LSTM/adadelta_comb.png">


```
With AdaDelta : 

Acc: 24.49% 
Precision: 0.24 
Recall: 0.24 
F1 score: 0.24

Total time took for training 21.339 seconds.

Model Loss on training data  1.5830999612808228
Model Accuracy on training data:  0.2595505714416504
Model Loss on validation data 1.585620403289795
Model Accuracy on validation data:  0.2404494434595108
```

### Model 4 - CNN 1-D + LSTM 

<img src="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/CNN1D%2BLSTM/adam_comb.png">

```
With Adam : 
Acc: 81.57% 
Precision: 0.82 
Recall: 0.82 
F1 score: 0.82

Total time took for training 181.622 seconds.

Model Loss on training data  0.0008503638091497123
Model Accuracy on training data:  1.0
Model Loss on validation data 0.670300304889679
Model Accuracy on validation data:  0.8719100952148438
```

<img src="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/CNN1D%2BLSTM/adagrad_comb.png">

```
With AdaGrad : 

Acc: 33.03% 
Precision: 0.33 
Recall: 0.33 
F1 score: 0.33

Total time took for training 14.808 seconds.

Model Loss on training data  1.5146769285202026
Model Accuracy on training data:  0.2657303512096405
Model Loss on validation data 1.5247085094451904
Model Accuracy on validation data:  0.2719101011753082
```

<img src="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/CNN1D%2BLSTM/sgd_comb.png">

```
With SGD : 

Acc: 34.61% 
Precision: 0.35 
Recall: 0.35 
F1 score: 0.35

Total time took for training 35.070 seconds.

Model Loss on training data  0.9556211233139038
Model Accuracy on training data:  0.5629213452339172
Model Loss on validation data 1.1251300573349
Model Accuracy on validation data:  0.5078651905059814
```

<img src="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/CNN1D%2BLSTM/rmsprop_comb.png">

```
With RMSProp : 

Acc: 91.24% 
Precision: 0.91 
Recall: 0.91 
F1 score: 0.91

Total time took for training 38.411 seconds.

Model Loss on training data  5.357720378462716e-10
Model Accuracy on training data:  1.0
Model Loss on validation data 0.5674594640731812
Model Accuracy on validation data:  0.9438202381134033
```

<img src="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/CNN1D%2BLSTM/adadelta_comb.png">

```
With AdaDelta : 

Model Loss on training data  1.5710480213165283
Model Accuracy on training data:  0.24438202381134033
Model Loss on validation data 1.5702738761901855
Model Accuracy on validation data:  0.2404494434595108
```

### Model 5 - CNN 1-D + Bi Directional LSTM 

<img src="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/CNN1D%2BBi-LSTMs/adam_comb.png">

```
With Adam : 

Acc: 94.38% 
Precision: 0.94 
Recall: 0.94 
F1 score: 0.94

Total time took for training 417.384 seconds.

Model Loss on training data  0.00014469937013927847
Model Accuracy on training data:  1.0
Model Loss on validation data 0.2845289707183838
Model Accuracy on validation data:  0.9438202381134033
```

<img src="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/CNN1D%2BBi-LSTMs/adagrad_comb.png">

```
With AdaGrad : 


Acc: 87.64% 
Precision: 0.88 
Recall: 0.88 
F1 score: 0.88

Total time took for training 871.265 seconds.

Model Loss on training data  0.009952237829566002
Model Accuracy on training data:  0.9994382262229919
Model Loss on validation data 0.5138413906097412
Model Accuracy on validation data:  0.8764045238494873
```

<img src="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/CNN1D%2BBi-LSTMs/sgd_comb.png">

```
With SGD : 

Acc: 69.89% 
Precision: 0.70 
Recall: 0.70
F1 score: 0.70

Total time took for training 1121.185 seconds.

Model Loss on training data  0.8237250447273254
Model Accuracy on training data:  0.748314619064331
Model Loss on validation data 1.093525767326355
Model Accuracy on validation data:  0.6988763809204102
```
<img src="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/CNN1D%2BBi-LSTMs/rmsprop_comb.png">

```
With RMSProp : 


Acc: 94.83% 
Precision: 0.95 
Recall: 0.95 
F1 score: 0.95

Total time took for training 222.419 seconds.

Model Loss on training data  1.339430094615679e-10
Model Accuracy on training data:  1.0
Model Loss on validation data 0.4486222565174103
Model Accuracy on validation data:  0.9483146071434021
```

<img src="https://github.com/harshgeek4coder/MultiClass-NLP-Classification-Hybrid-Neural-Models-Analysis-with-Optimizers/blob/main/Images/CNN1D%2BBi-LSTMs/adadelta_combo.png">

```
With AdaDelta : 

Acc: 24.49%
Precision: 0.24
Recall: 0.24
F1 score: 0.24

Total time took for training 123.594 seconds.

Model Loss on training data  1.5946524143218994
Model Accuracy on training data:  0.2606741487979889
Model Loss on validation data 1.595898985862732
Model Accuracy on validation data:  0.24494382739067078
 ```


## Summarised Analysis:

```
+------------------------------------------+----------+----------------------------+
|          Model [With Optimiser]          | Accuracy | Time Taken To Train (secs) |
+------------------------------------------+----------+----------------------------+
|              CNN-1D [Adam]               |  95.28   |           179.62           |
|             CNN-1D [AdaGrad]             |  88.76   |           307.4            |
|               CNN-1D [SGD]               |  84.72   |           182.02           |
|             CNN-1D [RMSProp]             |  93.48   |           56.43            |
|            CNN-1D [AdaDelta]             |  26.97   |           107.23           |
|               LSTM [Adam]                |  81.57   |           797.8            |
|              LSTM [AdaGrad]              |  36.85   |          1106.78           |
|                LSTM [SGD]                |  33.93   |           665.59           |
|              LSTM [RMSProp]              |  93.93   |           494.15           |
|             LSTM [AdaDelta]              |  23.15   |           207.85           |
|        Bidirectional LSTM [Adam]         |  94.61   |           490.36           |
|       Bidirectional LSTM [AdaGrad]       |  83.37   |          1439.13           |
|         Bidirectional LSTM [SGD]         |  32.81   |           613.55           |
|       Bidirectional LSTM [RMSProp]       |  93.48   |           432.31           |
|      Bidirectional LSTM [AdaDelta]       |  24.04   |           460.73           |
|           CNN-1D + LSTM  [Adam]          |  87.19   |           359.42           |
|         CNN-1D + LSTM  [AdaGrad]         |  27.19   |           160.44           |
|           CNN-1D + LSTM  [SGD]           |  50.79   |           597.64           |
|         CNN-1D + LSTM  [RMSProp]         |  94.38   |           211.52           |
|         CNN-1D + LSTM  [AdaDelta]        |  24.04   |           136.47           |
|    CNN-1D + Bidirectional LSTM  [Adam]   |  94.38   |           417.38           |
|  CNN-1D + Bidirectional LSTM  [AdaGrad]  |  87.64   |           871.27           |
|    CNN-1D + Bidirectional LSTM  [SGD]    |  69.89   |          1121.19           |
|  CNN-1D + Bidirectional LSTM  [RMSProp]  |  94.83   |           222.42           |
|  CNN-1D + Bidirectional LSTM  [AdaDelta] |  24.49   |           123.59           |
+------------------------------------------+----------+----------------------------+
```

### Let's Sort this table based on Highest Accuracy :

```
+------------------------------------------+----------+----------------------------+
|          Model [With Optimiser]          | Accuracy | Time Taken To Train (secs) |
+------------------------------------------+----------+----------------------------+
|    CNN-1D + Bidirectional LSTM  [Adam]   |  95.28   |           144.17           |
|        Bidirectional LSTM [Adam]         |  94.61   |           91.48            |
|              CNN-1D [Adam]               |  93.71   |           132.44           |
|              LSTM [RMSProp]              |  92.81   |           77.25            |
|       Bidirectional LSTM [RMSProp]       |  92.81   |           75.92            |
|  CNN-1D + Bidirectional LSTM  [RMSProp]  |  92.81   |           48.54            |
|             CNN-1D [RMSProp]             |  92.58   |           27.58            |
|         CNN-1D + LSTM  [RMSProp]         |  91.24   |           38.41            |
|  CNN-1D + Bidirectional LSTM  [AdaGrad]  |  89.66   |           103.6            |
|             CNN-1D [AdaGrad]             |  86.52   |           42.86            |
|               CNN-1D [SGD]               |  85.17   |           49.21            |
|       Bidirectional LSTM [AdaGrad]       |  83.37   |           99.64            |
|           CNN-1D + LSTM  [Adam]          |  81.57   |           181.62           |
|    CNN-1D + Bidirectional LSTM  [SGD]    |  68.99   |           90.17            |
|               LSTM [Adam]                |  38.43   |           119.48           |
|           CNN-1D + LSTM  [SGD]           |  34.61   |           35.07            |
|         Bidirectional LSTM [SGD]         |  33.93   |           66.54            |
|                LSTM [SGD]                |  33.26   |           36.67            |
|         CNN-1D + LSTM  [AdaGrad]         |  33.03   |           14.81            |
|            CNN-1D [AdaDelta]             |  28.31   |           23.97            |
|              LSTM [AdaGrad]              |  25.17   |           32.49            |
|      Bidirectional LSTM [AdaDelta]       |  24.49   |           21.34            |
|         CNN-1D + LSTM  [AdaDelta]        |  24.49   |           14.91            |
|             LSTM [AdaDelta]              |  24.27   |           11.98            |
|  CNN-1D + Bidirectional LSTM  [AdaDelta] |   22.7   |           16.33            |
+------------------------------------------+----------+----------------------------+
```


### Let's Sort this table based Maximum Time Taken By A Model :

```
+------------------------------------------+----------+----------------------------+
|          Model [With Optimiser]          | Accuracy | Time Taken To Train (secs) |
+------------------------------------------+----------+----------------------------+
|           CNN-1D + LSTM  [Adam]          |  81.57   |           181.62           |
|    CNN-1D + Bidirectional LSTM  [Adam]   |  95.28   |           144.17           |
|              CNN-1D [Adam]               |  93.71   |           132.44           |
|               LSTM [Adam]                |  38.43   |           119.48           |
|  CNN-1D + Bidirectional LSTM  [AdaGrad]  |  89.66   |           103.6            |
|       Bidirectional LSTM [AdaGrad]       |  83.37   |           99.64            |
|        Bidirectional LSTM [Adam]         |  94.61   |           91.48            |
|    CNN-1D + Bidirectional LSTM  [SGD]    |  68.99   |           90.17            |
|              LSTM [RMSProp]              |  92.81   |           77.25            |
|       Bidirectional LSTM [RMSProp]       |  92.81   |           75.92            |
|         Bidirectional LSTM [SGD]         |  33.93   |           66.54            |
|               CNN-1D [SGD]               |  85.17   |           49.21            |
|  CNN-1D + Bidirectional LSTM  [RMSProp]  |  92.81   |           48.54            |
|             CNN-1D [AdaGrad]             |  86.52   |           42.86            |
|         CNN-1D + LSTM  [RMSProp]         |  91.24   |           38.41            |
|                LSTM [SGD]                |  33.26   |           36.67            |
|           CNN-1D + LSTM  [SGD]           |  34.61   |           35.07            |
|              LSTM [AdaGrad]              |  25.17   |           32.49            |
|             CNN-1D [RMSProp]             |  92.58   |           27.58            |
|            CNN-1D [AdaDelta]             |  28.31   |           23.97            |
|      Bidirectional LSTM [AdaDelta]       |  24.49   |           21.34            |
|  CNN-1D + Bidirectional LSTM  [AdaDelta] |   22.7   |           16.33            |
|         CNN-1D + LSTM  [AdaDelta]        |  24.49   |           14.91            |
|         CNN-1D + LSTM  [AdaGrad]         |  33.03   |           14.81            |
|             LSTM [AdaDelta]              |  24.27   |           11.98            |
+------------------------------------------+----------+----------------------------+
```
