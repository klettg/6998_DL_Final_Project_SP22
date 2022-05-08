# Ethereum Price Prediction

**Griffin Klett (gk2591)**

**Nathan Cuevas (njc2150)**

---

**Professor Parijat Dube | Deep Learning and Systems Performance COMS 6998 | Spring 2022 | Columbia University**

---

**Project Description**

The goal of this project is to implement and backtest deep learning models that build and expand upon approaches in existing literature on Bitcoin prediction. We will then then translate and apply approaches from previous literature to create a classifier that when given previous data, can predict whether the price will rise or fall the following day. We hope to leverage our domain knowledge in blockchains to design this model and carefully select an ideal feature set to create a profitable model. We will test the performance of the model by measuring test and validation accuracy and by creating a trading simulation that trades an initial investment on previous data. 

**Repository Description**

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
