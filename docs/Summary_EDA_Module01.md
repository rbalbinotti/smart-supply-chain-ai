

## **Exploratory Data Analysis (EDA) Report \- Module 01: Demand and Inventory**

---

### **1\. Executive Summary**

This report summarizes the Exploratory Data Analysis (EDA) conducted for Module 01 of the **"Intelligent System for Supply Chain Management"** project. The primary project goal is to **optimize inventory and purchasing management**, with a target of **reducing overstocking by 20%** within six months.

The initial analysis confirms that **overstocking is the most prevalent inventory issue** in the database, representing approximately **50%** of all classified entries. The **Pantry** category is the largest contributor to overstocking, while the **Fresh Foods** category faces the biggest challenge with understock and expired stock.

---

### **2\. Overview and Data Quality**

The initial dataset contains 3000 entries and 29 columns, covering the period from 2022-12-09 to 2025-09-22.

| Key Metric | Average Value | Minimum | Maximum | Insight |
| :---- | :---- | :---- | :---- | :---- |
| **Sales Volume** | \~361 units | 3 units | 4068 units | High dispersion, indicating products with both low and very high demand. |
| **Lead Time** | \~4.7 days | 2 days | 12 days | Short lead time, which is essential for optimizing **Just-in-Time** processes. |
| **Shelf Life** | \~260 days | 2 days | 1825 days (5 years) | Large variation, highlighting the need for distinct stocking strategies for perishable vs. non-perishable goods. |

**Data Quality:** The dataset is complete, with no missing values, and categorical and date columns have been correctly processed for analysis.

---

### **3\. Key Findings and Inventory Analysis**

#### **3.1. Inventory Classification Distribution**

Inventory analysis, based on current stock levels (stock\_quantity) relative to min\_stock and max\_stock limits, reveals the following breakdown (approximate values):

* **Overstock:** Approximately **1500** entries (the largest group).  
* **Understock:** Approximately **800** entries.  
* **Safe:** Approximately **700** entries.

**Conclusion:** The biggest opportunity for improvement lies in **Overstock** management, which directly aligns with the project's goal.

#### **3.2. Expired Stock and Turnover**

* **Expired Stock:** A total of **134** items were classified as potentially expired. The **Fresh Foods** category (39 items) is the most affected, due to its very short shelf life (average of 5 days).  
* **Inventory Turnover Rate:** The average turnover rate is **\~1.04**, with the median at **\~0.84**. A median below 1 suggests that more than half of the items have slow turnover, reinforcing the overall stock inefficiency and the overstocking problem.

#### **3.3. Analysis by Category**

| Category | Most Frequent | Most Common Category for Overstock |
| :---- | :---- | :---- |
| **General** | Pantry | Pantry (470 items) |
| Sub-Category | Vegetables | \- |

**Category/Stock Relationship:**

* The **Pantry** category is the primary focus area for **Overstock** reduction.  
* The **Fresh Foods** category shows the highest number of items in **Understock** (85 entries).

#### **3.4. External Influence (Weather)**

The analysis of sales demand in relation to weather severity indicates that the highest volume of demand classified as **High** occurs under **Moderate** weather severity conditions. This is an important factor to consider in demand forecasting models.

---

### **4\. Conclusion and Next Steps**

The EDA results clearly establish that **Overstocking** is the main inefficiency point, concentrated primarily in **Pantry** products.

The next steps in the project should include utilizing the created features, such as inventory\_turnover\_rate and the stock level classifications (stock\_level\_classification), in machine learning models to predict optimal demand and optimize reorder points.