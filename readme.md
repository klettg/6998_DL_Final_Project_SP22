# Ethereum Price Prediction

**Griffin Klett (gk2591)**

**Nathan Cuevas (njc2150)**

---

**Professor Parijat Dube | Deep Learning and Systems Performance COMS 6998 | Spring 2022 | Columbia University**

---

### Project Description

The goal of this project is to implement and backtest deep learning models that build and expand upon approaches in existing literature on Bitcoin prediction. We will then then translate and apply approaches from previous literature to create a classifier that when given previous data, can predict whether the price will rise or fall the following day. We hope to leverage our domain knowledge in blockchains to design this model and carefully select an ideal feature set to create a profitable model. We will test the performance of the model by measuring test and validation accuracy and by creating a trading simulation that trades an initial investment on previous data. 

### Repository Description

This repository's directory structure is as follows:

```
├── code
│   ├── model.py
│   ├── notebooks
│   │   ├── CNN.ipynb
│   │   └── DNN.ipynb
│   └── scripts
│       ├── reverse.py
│       └── script.py
├── data
│   ├── ActiveERC20Addrs.csv
│   ├── BTC-USD.csv
│   ├── BlockDifficulty.csv
|  ...
│   └── data_v2.csv
├── documents
│   ├── Final_Project_Proposal.docx
│   └── Midterm_Seminar Presentation.pptx
├── readme.md
└── references
    ├── A_Comparative_Study_of_Bitcoin_Price_Prediction_Using_Deep_Learning_Ji_Et_Al.pdf
    ├── Applying_Deep_Learning_to_Better_Predict_Cryptocurrency_Trends.pdf
    └── Forecasting_the_Price_of_Bitcoin_using_deep_learning.pdf
```

The top level directory has the following folders:

* Code: 
  * The folder `notebooks` contains the notebooks that were used in implementing and testing the model alongside the plots. 
  * The folder `scripts` contains the scripts that were used to consolidate data on blockchians found on the web as well as generate the plots for the report. 
  * `model.py` contains most of our code in a singlular file. See instructions below on how to use this file. 

*  Data:
  * Data that was gathered from the web. `data_v2.csv` aggragates all of this data and is currently the dataset being used in our expriments. 
  * Contains the experiment data in the `experiment_data` directory. 

* Documents: 
  * Includes our slides for the presentation as well as the proposal and slides from our related midterm seminar presentation. 

* References
  * contains the top papers that we considered for this project

### Example Commands to run Project

**dependencies:**

Before running any of our code, please make sure the following dependencies are installed

* python 3.8.1+
* pandas 1.4.1+
* numpy 1.21.5+
* tensorflow 2.8.0+
* matplotlib 3.5.1+

**model.py**

Navigate to `code/model.py` to run the code we used for this project. Set the following flags to use. For examples, refer to end of this section. 

* `--model`: This is a required flag. Options are `CNN` and `DNN` which runs the convolutional neural network model and the deep neural network respectively. 
* `--m`: The number of days that each training example is allowed to look back. See the slides in `documents` folder for more details. This is an optional flag, default is 30. 
* `--epochs`: The number of epochs to run the model. This is an optional flag, default is 100.
* `--simulate`: Will run our simulation that gives a trading bot a $10,000 inital investment. The integer that is passed after this flag will be how many times the simulation is run. The average end investment will be determined and printed after all the simulation runs. This is an optional flag, default is 0. 

To simply check if the model runs, run the following:

```
python model.py --model CNN
```

To run our simulation 10 times for 100 epochs with $m=35$ using the deep neural network, run the following:

```
python model.py --model DNN --epochs 150 --m 35 --simulate 10
```

### Project Results

**summary**

Overall,  we were largely unsuccessful in building a profitable, deployable trading algorithm for ethereum prices; our models were unable to generalize to unseen data, regardless of how we normalized or sliced test/train, and regardless of how much localized information was fed into the model. Further, our model did not account for trading fees or associated liquidity costs with executing the algorithm.

For example, unlike previous research on Bitcoin, our performance did not depend in any meaningful way on the size of m per training/testing The two possible consequences of these results suggest an improvement is needed in the data collection/exploration domain, or that Ethereum’s market price is not subject to any discernible pattern by our DNN/CNN Models. Further research could incorporate using some better scaling of the network or restricting the test/training time to the period after which ethererum had 1000x’d network size; this was impossible for us as the dataset remains too small but will be possible in the future.

**experiment results/plots**

Validation accuracy for $m=30$. We notice that validation accuracy does not meaningfully improve over epoch, regardless of model architecture.

<img width="437" alt="image" src="https://user-images.githubusercontent.com/42158119/167318187-c8104796-143e-45d5-9fa0-818bee191662.png">

Contrary to Ji et Al's research, there was no significant improvement when adjusting the value of $m$ from $10$ to $70$. 

<img width="761" alt="image" src="https://user-images.githubusercontent.com/42158119/167318237-cec47e29-ae9f-4e3e-9f4a-1be31c46fe75.png">

CNN and DNN training and validation accuracies amongst several values of $m$. There is impovement in training accuracy per epoch; however the same is not seen for validation accuracy. This is due to our model's not being able to generalize effectively. 

<img width="900" alt="image" src="https://user-images.githubusercontent.com/42158119/167318281-931a41fb-50be-44fb-af69-dfee99561d52.png">

The results of of trading algorithm using our DNN and CNN models after given an initial investment of $10,000 and running on all of the uneen days. The results were averaged over 5 runs. 

<img width="669" alt="image" src="https://user-images.githubusercontent.com/42158119/167318350-7b600017-8b6a-4e74-9477-ff4abb56d30a.png">

