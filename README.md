# AI-Driven Store Footfall Impact Analysis on Sales

## Project Overview

The **AI-Driven Store Footfall Impact Analysis on Sales** project focuses on analyzing the relationship between customer footfall and store sales performance using data analysis, regression modeling, and interactive visualizations.

The project uses store-related factors such as foot traffic, advertising expenditure, average price, staff count, promotions, and inventory availability to understand their influence on sales performance.

An interactive Streamlit dashboard presents the analysis results and helps users explore footfall patterns, sales trends, regression performance, and store-level impact.

## Objectives

- Analyze the relationship between customer footfall and sales.
- Identify important factors influencing store revenue.
- Perform exploratory data analysis using visualizations.
- Build regression models to understand and predict sales performance.
- Evaluate model performance using R², MAE, and RMSE.
- Present insights through an interactive Streamlit dashboard.

## Key Features

### 1. Data Cleaning
- Load and prepare the store footfall and sales dataset.
- Handle missing values and prepare data for analysis.
- Organize relevant features for regression modeling.

### 2. Exploratory Data Analysis (EDA)
- Sales trends over time.
- Footfall trends over time.
- Store-wise performance comparison.
- Relationship between footfall and sales.

### 3. Regression Modeling
- Linear Regression.
- Multiple Regression.
- Model evaluation using:
  - R² Score
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)

### 4. Interactive Streamlit Dashboard

The dashboard contains seven sections:

- **Overview:** Summary of the project and key information.
- **Footfall Analysis:** Explore customer traffic patterns.
- **Sales Analysis:** Understand sales performance.
- **Footfall vs Sales:** Examine the relationship between footfall and sales.
- **Regression Model:** View regression modeling results.
- **Impact Analysis:** Analyze the impact of footfall on sales.
- **Insights:** Present important findings from the analysis.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Git and GitHub
- Git Large File Storage (Git LFS)

## Project Structure

```text
AI_Store_Footfall_Analysis/
│
├── app.py
│
├── data/
│   └── clean_footfall_sales.csv
│
├── models/
│   ├── model2_multiple_regression.pkl
│   ├── model_summary.json
│   └── store_level_impact.csv
│
├── notebooks/
│   ├── analysis.ipynb
│   └── figures/
│
├── stage1_data_understanding.py
│
├── .gitattributes
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/varshitha211/HCL-CAPSTONE.git
```

### 2. Navigate to the Project Folder

```bash
cd HCL-CAPSTONE
```

### 3. Install Git LFS

The cleaned dataset is stored using Git Large File Storage because of its large file size.

```bash
git lfs install
git lfs pull
```

### 4. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The dashboard will open in your browser, usually at:

```text
http://localhost:8501
```

## Model Evaluation

The regression models are evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| R² Score | Measures how well the model explains variation in sales |
| MAE | Measures the average absolute prediction error |
| RMSE | Measures prediction error while giving larger errors more weight |

The model results are stored in the `models/` directory.

## Dataset Information

The project uses a cleaned store footfall and sales dataset.

The dataset includes store-related attributes such as:

- Store ID
- Foot traffic
- Advertising expenditure
- Average price
- Staff count
- Store size
- Promotional days
- Inventory fill rate
- Returns rate
- Competitor presence
- Local income index
- Parking spaces
- Store age
- Holiday-related information

The cleaned dataset is maintained using Git LFS.

## Expected Outcome

The project aims to provide insights into how customer footfall and other store-related factors influence sales performance.

The interactive dashboard makes it easier to explore store performance, understand relationships between variables, and interpret regression model results.
## 🚀 Live Project Demo

**Deployed Application:** [AI-Driven Store Footfall Impact Analysis](https://hcl-capstone-8hdvtrhzk8kcoknqnn6mj3.streamlit.app/)

**GitHub Repository:** [HCL-CAPSTONE](https://github.com/varshitha211/HCL-CAPSTONE)

## Author

**Amara Krupa Varshitha**

## Project Type

Individual Capstone Project

**Project Title:** AI-Driven Store Footfall Impact Analysis on Sales
