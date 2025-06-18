# Insurance Risk Analytics & Predictive Modeling

## Project Overview

This repository hosts the codebase for the 10 Academy: Artificial Intelligence Mastery project (June 11–17, 2025), focusing on End-to-End Insurance Risk Analytics & Predictive Modeling for AlphaCare Insurance Solutions (ACIS). The objective is to analyze historical car insurance data (Feb 2014–Aug 2015) to identify low-risk customer segments, optimize premium pricing, and enhance marketing strategies in South Africa.

---

## Objectives

- Conduct Exploratory Data Analysis (EDA) to uncover patterns in risk and profitability.

- Implement Data Version Control (DVC) for reproducible data pipelines.

- Perform A/B Hypothesis Testing to validate risk differences across provinces, zip codes, and genders.

- Build predictive models to estimate claim severity and optimize premiums.

---

## Repository Structure

'''bash 
insurance-risk-analytics/
├── data/                    # Raw and processed datasets
├── notebooks/               # Jupyter notebooks for EDA and analysis
├── scripts/                 # Python scripts for tasks
├── .github/workflows/       # CI/CD workflows
├── README.md                # Project documentation
├── .gitignore               # Git ignore file
└── requirements.txt         # Python dependencies

'''

---

## Setup Instructions

1. Clone the Repository:'''bash git clone https://github.com/your-username/insurance-risk-analytics.git
cd insurance-risk-analytics

2. Set Up Python Environment:'''bash python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

3. Install Dependencies:

   - Python 3.8+

   - Required packages:'''bash pip install pandas numpy matplotlib seaborn scipy dvc

4. Version Control:

   - Create a branch for each task:''' bash git checkout -b task-1

   - Commit changes frequently with descriptive messages.

   - Push branches to GitHub and create Pull Requests for merging.

## CI/CD with GitHub Actions

A GitHub Actions workflow is configured to lint Python code and ensure code quality. See '''bash .github/workflows/python-lint.yml for details.      
