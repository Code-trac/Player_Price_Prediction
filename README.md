# 🏏 IPL Auction Price Prediction using Machine Learning

**Author**: Satyam Singh  
**Project Type**: Data Science / Sports Analytics  
**Tech Stack**: Python, pandas, scikit-learn, matplotlib

---

## 🚩 Overview
This project predicts IPL player auction prices using a combination of player performance statistics and machine learning models. By integrating datasets from multiple IPL seasons, we explore the influence of both player stats and "brand value" on final auction prices, and demonstrate best practices in end-to-end data science and predictive modeling.

## 👨‍💻 Project Workflow

### 📊 Data Collection & Cleaning
- Auction and player stats collected for IPL 2022 & 2025.
- Extensive cleaning, including custom mapping to resolve player name mismatches between sources.

### 🛠️ Feature Engineering & Merging
- Batting and bowling stats (e.g., runs, strike rate, wickets, economy) merged with auction price.
- Feature selection and log transformation are used to address highly skewed price distributions.

### 🧠 Modeling & Evaluation
- Linear Regression and Random Forest tested, with and without log transformation.
- Train/test splits ensure robust, out-of-sample evaluation.
- Metrics: RMSE and R² on both original and log-transformed price.

### 📉 Visualization
- Actual vs. predicted price plots to reveal prediction strengths and weaknesses.
- Tabulated output of top predictions and real prices.

### 📌 Insights & Business Analysis
- Analysis of which features (stats) most strongly affect predicted prices.
- Discussion of "brand value", auction psychology, and outliers.

---

## 🛠️ Installation & Usage

1. **Clone this repo (or download as zip)**  
2. **Install dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib
```
3. **Place data files in the working directory:**
```
ipl_2022_dataset.csv, ipl_2025_dataset.csv  
bat_2022.csv, bowl_2022.csv, bat_2025.csv, bowl_2025.csv
```
4. **Run the main script:**
```bash
python app.py
```

### Outputs:
- Model training logs (RMSE, R² for each experiment)
- Actual vs. predicted price plots (saved or displayed)
- Top player price predictions and residuals (in console or file)

---

## 📈 Results & Insights
- **Linear Regression** was ineffective (RMSE ~9, R² near -1): IPL price is not a linear function of stats.
- **Random Forest with log-transformed price** greatly improved predictions (R² > 0.98 for bowlers, >0.97 for batsmen).
- Player performance influences prices, but **brand value and hype create big outliers** (especially for star batsmen and captains).
- Actual vs. predicted plots show good fit for most, but **underestimate “blockbuster” signings** — reminding us that in auctions, data tells much but not all of the story.

---

## 📋 Key Learnings
- The importance of careful data cleaning in multi-source sports datasets.
- The power of **log transformation** in stabilizing ML regression problems with skewed targets.
- **Random Forests** can capture non-linear, high-variance patterns missed by linear models.
- Non-statistical factors (star power, captaincy, “brand”) are significant in cricket auctions.

---

## 🚀 Next Steps
- Try new model types (e.g., XGBoost, ensemble methods)
- Build interpretability / SHAP analysis for feature importances
- Include more granular, recent performance features
- Deploy as a web app or dashboard for real-time auction analysis

---

## 🤝 Contact
Interested in discussing sports analytics, data science, or this project?  
Connect with me on LinkedIn:linkedin.com/in/satyam-singh-108486300
Or open an issue.
