# DSI-KPMG-Capstone

## WEEK 8 Updates
### Progress
- update feature datasets
- binary classification model
- modify LSTM and CNN model, implement forward forecasting

### Problem
- CIQ database access
- Include previous target variable or not? if include, prediction looks like a simple shifting on CNN

### Plan
- emsemble LSTM, CNN, Random forest classifier 

---

## WEEK 7 Updates
### Progress
- First report: 1. Topic overview and progress overview; 2. Feature selection; 3. EDA on both target and predictors; 4. Model Setup, VAR, CNN, LSTM
- Automate CNN model (pipeline)
- Automate LSTM model (pipeline)
- 
### Problem

### Plan
- Modify LSTM archetecture
- CNN model fine tunning
- LSTM model fine tunning
- Scenrio based model set up

---

## WEEK 6 Updates
### Progress
- Feature selection: identified causation relationship within predictors, added new features that represent government fiscal and monetary policies 
- Feature engineering: constructed rolling window, normalized feature data, normalized time (assume seasonality exist)
- Implemented CNN model
- In process of implementing LSTM model

### Problem
- How to deal with important feature that contains many NA values?
- Is it useful to normalize date?

### Plan
- Feature selection
- Refine CNN model
- Implement LSTM model
- gradient-boosting tree/forest

---

## WEEK 5 Updates
### Progress
- Implemented VAR baseline model. We dropped Unemployment rate and GS10 when building the model based on p-value of the Granger causality test. Also, we used data from 1973-1 to 2017-4 to fit the model and picked maxlags=13 based on AIC. Predicted target from 2017-7 to 2018-4. RMSE = 101.75. 
- Implemented logistic regression model to classify the expanding and contracting trends. Accuracy rate is lower than 50%, which proved to be not applicable. 
- We also tried to implement auto-arima model to forecast the S&P 500 EPS-Index.

### Problem
- Our accuracy rate was low due to the black swan event(covid) in recent two years. 
- If our number of features is enough?

### Plan
- Implement Sliding window on our data and put it in our ML model.
- LSTM

---

## WEEK 4 Updates
### Progress
- Gathered predictive features dataset and performed data wrangling work. We chose the variables mentioned in this doc: https://www.investopedia.com/articles/personal-finance/020215/top-ten-us-economic-indicators.asp. We will consider add and remove some features later. (from 1969.08.01 - 2021-03-01): <br>
 [Unemployment Rate(16 and over):] (https://fred.stlouisfed.org/series/UNRATE) <br>
 [Median Sales Price for New Houses Sold in the United States:] (https://fred.stlouisfed.org/series/MSPNHSUS) <br>
 [10-Year Treasury Constant Maturity Rate:] (https://fred.stlouisfed.org/series/GS10) <br>
 [Personal Consumption Expenditures:] (https://fred.stlouisfed.org/series/PCE) <br>
 [Manufacturing New Orders:] (https://fred.stlouisfed.org/series/AMTMNO) <br>
 [Industrial Production: Manufacturing (NAICS):] (https://fred.stlouisfed.org/series/IPMAN) <br>
 [Producer Price Index by Commodity: All Commodities（PPIACO）] (https://fred.stlouisfed.org/series/PPIACO) <br>
 [Consumer Price Index for All Urban Consumers: Food and Beverages in U.S. City Average (CPIFABSL)] (https://fred.stlouisfed.org/series/CPIFABSL) <br>
 [Construction Spending:] (https://fred.stlouisfed.org/series/TTLCONS) <br>
 [MedianUsualWeeklyRealEarnings：] (https://fred.stlouisfed.org/series/LES1252881600Q) <br>
 [VolumeOfTotalRetailTradeSales：] (https://fred.stlouisfed.org/series/SLRTTO01USQ657S) <br>
- Correlation check: Found multiple predictors are highly correlated with each other. 
- <del>Co-integration test(p-value): WIP 🚧 </del>
- <del>reasoning: All features are not stationary, so cointegration which is probably a more robust measure of linkage between two features.</del>

### Problem
- Data length not consistent.
- Too many features are strongly correlated, and how should we keep these featurs?

### Plan
- Label expansion and contraction with 0 and 1 respectively, turning the ML problem into a classification problem.

---

## WEEK 3 Updates
### Progress
- Located unique peaks and troughs of index change. And looked for the historical events, variables(GDP, unemployment rate, ...) associated with these changes.
- Found long-term growth and short-term contraction, aka cycles, in the index change.

### Problem
- Clarificaiton of target variable.
- Data not consistent with the website.
- CIQ data access.
- Power calculation for feature selection.


