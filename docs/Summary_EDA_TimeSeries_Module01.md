

## **Exploratory Data Analysis (EDA) Report - Time Series**

### **1. Project Scope and Objectives**

The primary objective of the project is **to optimize inventory and purchasing management to reduce excess stock by 20%** within six months.

The two target variables analyzed are:

1.  **Inventory Optimization (Target: `stock_quantity`)**: Focus on analyzing daily stock patterns to reduce excess *stock*.
2.  **Demand Forecasting (Target: `sales_volume`)**: Focus on forecasting daily sales to align inventory levels with expected demand.

### **2. Data Preparation and Structure**

* **Frequency:** The target variables (`stock_quantity` and `sales_volume`) were aggregated and resampled to a **daily** (`D`) frequency.
* **Missing Values Treatment:** Days with no recorded activity had their values filled with **zero**.
* **Predictors:** The complete *DataFrame* contains 24 columns, including static predictors (such as `moq`, `lead_time`, `min_stock`, `max_stock`) and dynamic ones (such as `weather_severity`, `is_holiday`, `in_season`).

### **3. Time Series Analysis (Overview and Decomposition)**

#### **Trend, Cyclicality, and Seasonality**

Visual analysis of the raw series, along with decomposition (using an additive model with a 7-day period), revealed the following:

* **Variability:** Both time series exhibit a **high degree of daily or weekly variability**.
* **Trend:** No **clear long-term** growth or decline **trend** was observed for either metric during the analyzed period.
* **Seasonality:** There is a **strong weekly seasonal component** in both variables, indicating patterns that repeat every 7 days.
* **Cyclicality:** The patterns appear to be cyclical throughout the years, although without perfect regularity.

### **4. Stationarity Analysis (ADF and KPSS)**

Stationarity is a crucial property for time series modeling. The stationarity test was performed for both variables:

| Variable | ADF Test (p-value) | KPSS Test (p-value) | Conclusion |
| :--- | :--- | :--- | :--- |
| **`sales_volume`** | 0.000 | 0.100 | **Stationary** |
| **`stock_quantity`** | 0.000 | 0.100 | **Stationary** |

* **Conclusion:** The `sales_volume` and `stock_quantity` series are considered **stationary**, which simplifies the modeling stage, as it will not be necessary to apply differencing to remove trend or seasonality (although the weekly seasonality should be addressed in the model).

#### **Autocorrelation (t vs. t-1)**

The analysis of the correlation with a 1-day lag (`t` vs. `t-1`) showed a **strong positive autocorrelation** for both variables. This means that today's sales or stock value is highly correlated with the value from the previous day, typical behavior of non-random time series.

### **5. Correlation with Predictor Variables**

The correlation analysis between the stock variable (`stock_quantity`) and the numerical predictor variables revealed important relationships:

* **Strongest Positive Correlation:**
    * `max_stock` (Maximum Stock): **0.909**
    * `min_stock` (Minimum Stock): **0.908**
    * **Implication:** The stock quantity is, as expected, strongly influenced by the predefined stock limits (`min_stock` and `max_stock`).
* **Moderate Negative Correlation:**
    * `maximum_days_on_sale`: **-0.447**
    * `shelf_life_days` (Shelf Life): **-0.414**
    * **Implication:** Products with a shorter shelf life or a shorter maximum days on sale tend to have a more controlled or lower `stock_quantity`, which makes sense for managing the risk of expiration (perishable stocks).