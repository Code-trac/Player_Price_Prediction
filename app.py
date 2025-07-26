import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#Data Loading

df_2022 = pd.read_csv('ipl_2022_dataset.csv')
df_2025 = pd.read_csv('ipl_2025_dataset.csv')
bat_2022 = pd.read_csv('bat_2022.csv')
bowl_2022 = pd.read_csv('bowl_2022.csv')
bat_2025 = pd.read_csv('bat_2025.csv')
bowl_2025 = pd.read_csv('bowl_2025.csv')


#Name Normalization as names in datasets are different

name_mapping = {
    'JC Buttler': 'Jos Buttler',
    'Q de Kock': 'Quinton de Kock',
    'HH Pandya': 'Hardik Pandya',
    'DA Miller': 'David Miller',
    'F du Plessis': 'Faf Du Plessis',
    'S Dhawan': 'Shikhar Dhawan',
    'SV Samson': 'Sanju Samson',
    'DJ Hooda': 'Deepak Hooda',
    'LS Livingstone': 'Liam Livingstone',
    'DA Warner': 'David Warner',
    'RA Tripathi': 'Rahul Tripathi',
    'SS Iyer': 'Shreyas Iyer',
    'NT Tilak Varma': 'N. Tilak Varma',
    'AK Markram': 'Aiden Markram',
    'RD Gaikwad': 'Ruturaj Gaikwad',
    'V Kohli': 'Virat Kohli',
    'RR Pant': 'Rishabh Pant',
    'AD Russell': 'Andre Russell',
    'KD Karthik': 'Dinesh Karthik',
    'SO Hetmyer': 'Shimron Hetmyer',
    'N Pooran': 'Nicholas Pooran',
    'SA Yadav': 'Suryakumar Yadav',
    'GJ Maxwell': 'Glenn Maxwell',
    'S Dube': 'Shivam Dube',
    'PP Shaw': 'Prithvi Shaw',
    'AT Rayudu': 'Ambati Rayudu',
    'RG Sharma': 'Rohit Sharma',
    'YBK Jaiswal': 'Yashasvi Jaiswal',
    'JM Bairstow': 'Jonny Bairstow',
    'DP Conway': 'Devon Conway',
    'MR Marsh': 'Mitchell Marsh',
    'R Powell': 'Rovman Powell',
    'MM Ali': 'Moeen Ali',
    'JM Sharma': 'Jitesh Sharma',
    'MS Dhoni': 'MS Dhoni',
    'RV Uthappa': 'Robin Uthappa',
    'Shahbaz Ahmed': 'Shahbaz Ahmed',
    'R Tewatia': 'Rahul Tewatia',
    'KS Williamson': 'Kane Williamson',
    'PBB Rajapaksa': 'Bhanuka Rajapaksa',
    'MA Agarwal': 'Mayank Agarwal',
    'R Ashwin': 'R. Ashwin',
    'TH David': 'Tim David',
    'R Parag': 'Riyan Parag',
    'KH Pandya': 'Krunal Pandya',
    'AR Patel': 'Axar Patel',
    'VR Iyer': 'Venkatesh Iyer',
    'RK Singh': 'Rinku Singh',
    'SW Billings': 'Sam Billings',
    'D Brevis': 'Dewald Brevis',
    'A Badoni': 'Ayush Badoni',
    'Lalit Yadav': 'Lalit Yadav',
    'MS Wade': 'Matthew Wade',
    'MP Stoinis': 'Marcus Stoinis',
    'B Sai Sudharsan': 'Sai Sudharsan',
    'KA Pollard': 'Kieron Pollard',
    'AM Rahane': 'Ajinkya Rahane',
    'Anuj Rawat': 'Anuj Rawat',
    'SN Thakur': 'Shardul Thakur',
    'M Shahrukh Khan': 'Shahrukh Khan',
    'RA Jadeja': 'Ravindra Jadeja',
    'A Manohar': 'Abhinav Manohar',
    'Washington Sundar': 'Washington Sundar',
    'Rashid Khan': 'Rashid Khan',
    'SN Khan': 'Sarfaraz Khan',
    'MK Pandey': 'Manish Pandey',
    'MK Lomror': 'Mahipal Lomror',
    'AJ Finch': 'Aaron Finch',
    'RD Chahar': 'Rahul Chahar',
    'E Lewis': 'Evin Lewis',
    'SP Narine': 'Sunil Narine',
    'Shashank Singh': 'Shashank Singh',
    'SS Prabhudessai': 'Suyash Prabhudessai',
    'PJ Cummins': 'Pat Cummins',
    'JD Unadkat': 'Jaydev Unadkat',
    'R Shepherd': 'Romario Shepherd',
    'JO Holder': 'Jason Holder',
    'UT Yadav': 'Umesh Yadav',
    'OF Smith': 'Odean Smith',
    'K Rabada': 'Kagiso Rabada',
    'Kuldeep Yadav': 'Kuldeep Yadav',
    'PK Garg': 'Priyam Garg',
    'Ramandeep Singh': 'Ramandeep Singh',
    'D Pretorius': 'Dwaine Pretorius',
    'PVD Chameera': 'Dushmantha Chameera',
    'HV Patel': 'Harshal Patel',
    'HR Shokeen': 'Hrithik Shokeen',
    'TA Boult': 'Trent Boult',
    'N Jagadeesan': 'N Jagadeesan',
    'DR Sams': 'Daniel Sams',
    'PW Hasaranga': 'Wanindu Hasaranga',
    'R Dhawan': 'Rishi Dhawan',
    'DJ Mitchell': 'Daryl Mitchell',
    'SE Rutherford': 'Sherfane Rutherford',
    'JDS Neesham': 'James Neesham',
    'Mohammed Siraj': 'Mohammed Siraj',
    'TL Seifert': 'Tim Seifert',
    'B Kumar': 'Bhuvneshwar Kumar',
    'Mohsin Khan': 'Mohsin Khan',
    'DJ Bravo': 'Dwayne Bravo',
    'SP Jackson': 'Sheldon Jackson',
    'Avesh Khan': 'Avesh Khan',
    'HE van der Dussen': 'Rassie van der Dussen',
    'Harpreet Brar': 'Harpreet Brar',
    'MJ Santner': 'Mitchell Santner',
    'Arshdeep Singh': 'Arshdeep Singh',
    'B Indrajith': 'Baba Indrajith',
    'M Vohra': 'Manan Vohra',
    'V Shankar': 'Vijay Shankar',
    'Mandeep Singh': 'Mandeep Singh',
    'JR Hazlewood': 'Josh Hazlewood',
    'DJ Willey': 'David Willey',
    'KK Nair': 'Karun Nair',
    'Prabhsimran Singh': 'Prabhsimran Singh',
    'Anmolpreet Singh': 'Anmolpreet Singh',
    'M Ashwin': 'Murugan Ashwin',
    'CV Varun': 'Varun Chakravarthy',
    'CJ Jordan': 'Chris Jordan',
    'RA Bawa': 'Raj Bawa',
    'M Jansen': 'Marco Jansen',
    'S Gopal': 'Shreyas Gopal',
    'OC McCoy': 'Obed McCoy',
    'FA Allen': 'Fabian Allen',
    'KS Bharat': 'KS Bharat',
    'SA Abbott': 'Sean Abbott',
    'Rasikh Salam': 'Rasikh Salam',
    'Kartik Tyagi': 'Kartik Tyagi',
    'M Theekshana': 'Maheesh Theekshana',
    'Simarjeet Singh': 'Simarjeet Singh',
    'RV Patel': 'Ripal Patel',
    'Mukesh Choudhary': 'Mukesh Choudhary',
    'M Prasidh Krishna': 'Prasidh Krishna',
    'Aman Hakim Khan': 'Aman Khan',
    'LH Ferguson': 'Lockie Ferguson',
    'AS Joseph': 'Alzarri Joseph',
    'YS Chahal': 'Yuzvendra Chahal',
    'VG Arora': 'Vaibhav Arora',
    'PN Mankad': 'Prerak Mankad',
    'KS Sharma': 'Karan Sharma',
    'Abdul Samad': 'Abdul Samad',
    'A Tomar': 'Ashok Tomar',
    'Umran Malik': 'Umran Malik',
    "YS Chahal": "Yuzvendra Chahal",
    "PW Hasaranga": "Wanindu Hasaranga",
    "K Rabada": "Kagiso Rabada",
    "Umran Malik": "Umran Malik",
    "Kuldeep Yadav": "Kuldeep Yadav",
    "Mohammed Shami": "Mohammed Shami",
    "JR Hazlewood": "Josh Hazlewood",
    "Rashid Khan": "Rashid Khan",
    "HV Patel": "Harshal Patel",
    "M Prasidh Krishna": "Prasidh Krishna",
    "Avesh Khan": "Avesh Khan",
    "T Natarajan": "Thangarasu Natarajan",
    "AD Russell": "Andre Russell",
    "UT Yadav": "Umesh Yadav",
    "TA Boult": "Trent Boult",
    "KK Ahmed": "Khaleel Ahmed",
    "DJ Bravo": "Dwayne Bravo",
    "Mukesh Choudhary": "Mukesh Choudhary",
    "JJ Bumrah": "Jasprit Bumrah",
    "SN Thakur": "Shardul Thakur",
    "Mohsin Khan": "Mohsin Khan",
    "RD Chahar": "Rahul Chahar",
    "TG Southee": "Tim Southee",
    "JO Holder": "Jason Holder",
    "Ravi Bishnoi": "Ravi Bishnoi",
    "DR Sams": "Daniel Sams",
    "B Kumar": "Bhuvneshwar Kumar",
    "M Theekshana": "Maheesh Theekshana",
    "R Ashwin": "Ravichandran Ashwin",
    "LH Ferguson": "Lockie Ferguson",
    "OC McCoy": "Obed McCoy",
    "Yash Dayal": "Yash Dayal",
    "KH Pandya": "Krunal Pandya",
    "Arshdeep Singh": "Arshdeep Singh",
    "SP Narine": "Sunil Narine",
    "M Ashwin": "Murugan Ashwin",
    "PVD Chameera": "Dushmantha Chameera",
    "A Nortje": "Anrich Nortje",
    "Mohammed Siraj": "Mohammed Siraj",
    "MM Ali": "Moeen Ali",
    "HH Pandya": "Hardik Pandya",
    "Mustafizur Rahman": "Mustafizur Rahman",
    "RP Meredith": "Riley Meredith",
    "KR Sen": "Kuldeep Sen",
    "J Suchith": "Jagadeesha Suchith",
    "M Jansen": "Marco Jansen",
    "AS Joseph": "Alzarri Joseph",
    "PJ Cummins": "Pat Cummins",
    "GJ Maxwell": "Glenn Maxwell",
    "AR Patel": "Axar Patel",
    "R Sai Kishore": "Ravisrinivasan Sai Kishore",
    "R Dhawan": "Rishi Dhawan",
    "CV Varun": "Varun Chakravarthy",
    "Washington Sundar": "Washington Sundar",
    "LS Livingstone": "Liam Livingstone",
    "Ramandeep Singh": "Ramandeep Singh",
    "JD Unadkat": "Jaydev Unadkat",
    "D Pretorius": "Dwaine Pretorius",
    "TS Mills": "Tymal Mills",
    "OF Smith": "Odean Smith",
    "RA Jadeja": "Ravindra Jadeja",
    "K Kartikeya": "Kumar Kartikeya",
    "K Gowtham": "Krishnappa Gowtham",
    "Basil Thampi": "Basil Thampi",
    "Shivam Mavi": "Shivam Mavi",
    "Akash Deep": "Akash Deep",
    "MJ Santner": "Mitchell Santner",
    "Simarjeet Singh": "Simarjeet Singh",
    "Lalit Yadav": "Lalit Yadav",
    "MR Marsh": "Mitchell Marsh",
    "KA Pollard": "Kieron Pollard",
    "Harpreet Brar": "Harpreet Brar",
    "Shahbaz Ahmed": "Shahbaz Ahmed",
    "MP Stoinis": "Marcus Stoinis",
    "PJ Sangwan": "Pradeep Sangwan",
    "C Sakariya": "Chetan Sakariya",
    "NT Ellis": "Nathan Ellis",
    "VG Arora": "Vaibhav Arora",
    "R Shepherd": "Romario Shepherd",
    "NA Saini": "Navdeep Saini",
    "A Badoni": "Ayush Badoni",
    "PH Solanki": "Prashant Solanki",
    "M Pathirana": "Matheesha Pathirana",
    "Sandeep Sharma": "Sandeep Sharma",
    "HR Shokeen": "Hrithik Shokeen",
    "Fazalhaq Farooqi": "Fazalhaq Farooqi",
    "AJ Tye": "Andrew Tye",
    "VR Aaron": "Varun Aaron",
    "CJ Jordan": "Chris Jordan",
    "DG Nalkande": "Darshan Nalkande",
    "DJ Willey": "David Willey",
    "AS Roy": "Anukul Roy",
    "M Markande": "Mayank Markande",
    "TU Deshpande": "Tushar Deshpande",
    "Kartik Tyagi": "Kartik Tyagi",
    "Harshit Rana": "Harshit Rana",
    "AK Markram": "Aiden Markram",
    "DJ Hooda": "Deepak Hooda",
    "S Gopal": "Shreyas Gopal",
    "FA Allen": "Fabian Allen",
}

def clean_names(df, key='Player'):
    df[key] = df[key].apply(lambda x: x.split(' (')[0])
    df[key] = df[key].replace(name_mapping)
    return df

bat_2022 = clean_names(bat_2022, key="Player")
bowl_2022 = clean_names(bowl_2022, key="Player")
bat_2025 = clean_names(bat_2025, key="Player")
bowl_2025 = clean_names(bowl_2025, key="Player")

bat_types = ["BAT"]
bowl_types = ["BOWL"]

df_2022_bat = df_2022[df_2022['Type'].isin(bat_types)].copy()
df_2025_bat = df_2025[df_2025['Type'].isin(bat_types)].copy()
df_2022_bowl = df_2022[df_2022['Type'].isin(bowl_types)].copy()
df_2025_bowl = df_2025[df_2025['Type'].isin(bowl_types)].copy()

train_bat_2022 = pd.merge(df_2022_bat, bat_2022, on='Player', how='inner')
train_bowl_2022 = pd.merge(df_2022_bowl, bowl_2022, on='Player', how='inner')
train_bat_2025 = pd.merge(df_2025_bat, bat_2025, on='Player', how='inner')
train_bowl_2025 = pd.merge(df_2025_bowl, bowl_2025, on='Player', how='inner')

common_cols_bat = list(set(train_bat_2022.columns).intersection(train_bat_2025.columns))
train_bat_2022_common = train_bat_2022[common_cols_bat]
train_bat_2025_common = train_bat_2025[common_cols_bat]
train_bat_all = train_bat_2022_common.copy()

common_cols_bowl = list(set(train_bowl_2022.columns).intersection(train_bowl_2025.columns))
train_bowl_2022_common = train_bowl_2022[common_cols_bowl]
train_bowl_2025_common = train_bowl_2025[common_cols_bowl]
train_bowl_all = train_bowl_2022_common.copy()


for df in [train_bat_all, train_bowl_all]:
    sx = [c for c in df.columns if "Season_x" in c]
    sy = [c for c in df.columns if "Season_y" in c]
    if sx and sy:
        df["Season"] = df[sx[0]].combine_first(df[sy[0]])
        df.drop(columns=sx + sy, inplace=True)


def random_f(data, target_col, drop_cols=None, model_name="Model"):
    if drop_cols is None:
        drop_cols = []
    drop_cols = [c for c in drop_cols if c in data.columns]
    X = data.drop(columns=[target_col] + drop_cols)
    y = data[target_col]
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=42
        ))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")
    return pipeline, X_train.columns.tolist(), categorical_cols, (X_test, y_test, y_pred)

#Testing Linear Regression 

'''
def train_model(data, target_col, drop_cols=None, model_name="Model"):
    
    if drop_cols is None:
        drop_cols = []
        
    X = data.drop(columns=[target_col] + drop_cols)
    y = data[target_col]
    
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'  
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")
    
    plt.figure(figsize=(5,5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual Price (Cr)")
    plt.ylabel("Predicted Price (Cr)")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.show()
    
    return pipeline

#drop_cols_bat = ["Player", "Type", "Season"]



model_bat = train_model(
    data=train_bat_all,
    target_col="Price",
    drop_cols=drop_cols_bat,
    model_name="Batsmen Price Prediction"
)

drop_cols_bowl = ["Player", "Type"]

if "Season" in train_bowl_all.columns:
    drop_cols_bowl.append("Season")

model_bowl = train_model(
    data=train_bowl_all,
    target_col="Price",
    drop_cols=drop_cols_bowl,
    model_name="Bowlers Price Prediction"
)

'''

train_bat_all["Price_log"] = np.log1p(train_bat_all["Price"])
train_bowl_all["Price_log"] = np.log1p(train_bowl_all["Price"])
target_bat = "Price_log"
target_bowl = "Price_log"
drop_cols_bat = ["Player", "Type", "Team", "Mat", "Inns"]
drop_cols_bowl = ["Player", "Type", "Team", "Mat", "Inns"]


model_bat_rf, bat_features, bat_cat_cols, (Xb_test, yb_test, yb_pred_log) = random_f(
    data=train_bat_all,
    target_col=target_bat,
    drop_cols=drop_cols_bat,
    model_name="Batsmen Price Prediction (Random Forest, Log Target)"
)


model_bowl_rf, bowl_features, bowl_cat_cols, (Xl_test, yl_test, yl_pred_log) = random_f(
    data=train_bowl_all,
    target_col=target_bowl,
    drop_cols=drop_cols_bowl,
    model_name="Bowlers Price Prediction (Random Forest, Log Target)"
)


def plot(y_true_log, y_pred_log, title):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual Price (Cr)")
    plt.ylabel("Predicted Price (Cr)")
    plt.title(title)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)],
             color='red', linestyle='--')
    plt.grid(which='both', alpha=0.2)
    plt.tight_layout()
    plt.show()

plot(yb_test, yb_pred_log, "Batsmen: Actual vs Predicted Auction Price")
plot(yl_test, yl_pred_log, "Bowlers: Actual vs Predicted Auction Price")


def get_features(df, cols):
    missing_cols = set(cols) - set(df.columns)
    for c in missing_cols:
        df[c] = 0  
    return df[cols]

test_bat_2025 = get_features(train_bat_2025, bat_features)
bat_log_preds_2025 = model_bat_rf.predict(test_bat_2025)
train_bat_2025["PredictedPrice"] = np.expm1(bat_log_preds_2025)


test_bowl_2025 = get_features(train_bowl_2025, bowl_features)
bowl_log_preds_2025 = model_bowl_rf.predict(test_bowl_2025)
train_bowl_2025["PredictedPrice"] = np.expm1(bowl_log_preds_2025)

print("\nTop Predicted Batsmen:")
print(train_bat_2025[["Player", "Price", "PredictedPrice"]].sort_values(
    by="PredictedPrice", ascending=False).head(10))
print("\nTop Predicted Bowlers:")
print(train_bowl_2025[["Player", "Price", "PredictedPrice"]].sort_values(
    by="PredictedPrice", ascending=False).head(10))
