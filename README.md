# 🏎️ F1 PitStop Analyzer

A machine learning-driven analysis of Formula 1 pit stop data (2018–2024) to explore how variables like driver aggression, tire usage, and pit stop efficiency impact race performance—especially the likelihood of finishing in the top 5.

---

## 📊 Dataset

- **Source:** Formula 1 Pit Stops (2018–2024)
- **Size:** 7374 rows × 30 columns
- **Features Include:**
  - Driver, Constructor, Circuit, Race Details
  - Weather and environmental conditions
  - Performance strategy metrics (e.g., aggression scores, lap time variation)
  - Pit stop statistics and tire compound data

---

## 🧹 Preprocessing

- Cleaned column names to snake_case
- Removed duplicate records
- Replaced missing values with:
  - Median for numerical values like aggression scores, lap time variation
  - 0 or "Unknown" for categorical or optional fields (e.g., stint, tire compound)
- Converted data types for accurate modeling:
  - Date, time, and numeric conversions applied
- Target variable `top5` created:
  - **1** if driver finished in Top 5
  - **0** otherwise

---

## 📈 Visualizations

- **Driver Aggression vs. Position Change:** Understand how risky strategies correlate with gains or losses in race rank
- **Pit Time Distribution:** Reveal common pit durations and outliers
- **Tire Compound Usage:** Frequency of each tire type to analyze team strategy
- **Heatmap of Feature Correlation:** Highlight strongest relationships among performance variables
- **F1 Score Comparison Bar Plot:** Compare effectiveness of models
- **Confusion Matrices:** For all models, showing TP, FP, TN, FN

*Visual outputs are available in the `/visualizations` directory.*

---

## 🧠 ML Models Trained

We trained and evaluated 4 classification models to predict Top 5 finishes:

| Model              | Accuracy | F1 Score |
|-------------------|----------|----------|
| Logistic (Ridge)  | 99.8%    | 0.9986   |
| Random Forest      | 100%     | 1.0000   |
| KNN                | 98.9%    | 0.9893   |
| Naive Bayes        | 98.7%    | 0.9879   |

🎯 **Best Performing Model:**  
**✅ Random Forest** – Achieved perfect classification with 100% accuracy and F1 Score, showcasing robustness in identifying top-performing drivers using available features.

---

## 🔧 How to Run

1. Open the R script located in `notebooks/f1_analysis_modeling.R`
2. Make sure to install required R packages:

```r
install.packages(c("tidyverse", "lubridate", "janitor", "hms", "caret", "randomForest", "class", "e1071", "ggplot2"))
```

Run the script to:

- ✅ Clean the dataset  
- 📊 Visualize key relationships  
- 🧠 Train & evaluate ML models  
- 🏁 Output F1 scores and confusion matrices  

---

## 🙌 Acknowledgements

Special thanks to the open-source motorsport analytics community and Kaggle Datasets.  
This project would not be possible without public data sharing and the amazing R community.

---

## 🧠 Author

**Shanzay** 

