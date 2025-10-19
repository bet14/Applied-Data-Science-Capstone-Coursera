# SpaceX Launch Analysis Capstone Project

This repository contains the full project code and analysis for evaluating the viability of a hypothetical new space company, Space Y, to compete with SpaceX. The project focuses on analyzing SpaceX launch data to predict successful rocket landings, estimate launch costs, and identify optimal launch sites.

---

## Project Overview

This capstone project uses data science methodologies including data collection, exploratory data analysis (EDA), interactive visual analytics, and machine learning prediction to analyze SpaceX launch data. The objective is to provide insights and predictive models that can guide strategic decisions for rocket launches.

---

## Contents

- **Data Collection**  
  Code to collect launch data from SpaceX API and Wikipedia using web scraping techniques.

- **Data Wrangling & EDA**  
  Scripts for cleaning, summarizing, and visualizing data using Python, SQL queries, and plotting libraries.

- **Interactive Visual Analytics**  
  Folium maps and Plotly Dash dashboards to explore launch sites, payloads, and success rates interactively.

- **Predictive Modeling**  
  Machine learning classification models (Logistic Regression, SVM, Decision Tree, KNN) to predict successful landings.

- **Results & Conclusions**  
  Summary of findings including best launch sites, payload success rates, and model accuracy.

---

## Key Findings

- Data was successfully collected and enriched from public sources (SpaceX API and Wikipedia).  
- Decision Tree Classifier achieved the highest accuracy (>87%) in predicting successful landings.  
- The best launch site identified is KSC LC-39A, with a high success ratio and good logistics.  
- Payloads above 7,000kg show lower risk for launch failure.  
- Success rates have improved over time, reflecting technological advancements.

---

## Repository Structure

```
/data_collection/
  - spacex_api_collection.ipynb
  - web_scraping_wikipedia.ipynb

/data_wrangling/
  - data_wrangling.ipynb
  - eda_visualization.ipynb
  - eda_sql_queries.ipynb

/interactive_visuals/
  - folium_maps.ipynb
  - spacex_dash_app.py

/predictive_analysis/
  - machine_learning_prediction.ipynb

/appendix/
  - additional_notes.md
```

---

## How to Use

1. **Data Collection**: Run notebooks to fetch and prepare raw data.  
2. **Data Wrangling & EDA**: Explore and clean data, generate visualizations and SQL summaries.  
3. **Interactive Visuals**: Use Folium and Dash scripts to analyze launch site geography and payload relationships.  
4. **Predictive Modeling**: Train and evaluate classification models to predict landing success.  
5. **Review Results**: Check summary notebooks for insights and conclusions.

---

## Dependencies

- Python 3.x  
- Libraries: pandas, numpy, matplotlib, seaborn, plotly, folium, scikit-learn, SQLAlchemy, requests, beautifulsoup4

---

## References

- SpaceX API: https://api.spacexdata.com/v4/rockets/  
- Wikipedia Falcon 9 Launches: https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches



---

For detailed explanations and code comments, please refer to each notebook/script in the repository.
