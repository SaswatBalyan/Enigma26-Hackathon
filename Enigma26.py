import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


train_profiles_path = 'F:/train.xlsx'

test_profiles_path = 'F:/test.xlsx'   

target_path = 'F:/target.csv'       


train_users = pd.read_excel(train_profiles_path)
test_users = pd.read_excel(test_profiles_path)
train_pairs = pd.read_csv(target_path)


def clean_cols(df):
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    return df

train_users = clean_cols(train_users)
test_users = clean_cols(test_users)
train_pairs = clean_cols(train_pairs)


text_cols = ['Business_Interests', 'Business_Objectives', 'Constraints', 'Role', 'Industry', 'Location_City']
for col in text_cols:
    if col in train_users.columns:
        train_users[col] = train_users[col].fillna('').astype(str).str.lower()
        test_users[col] = test_users[col].fillna('').astype(str).str.lower()


all_users = pd.concat([train_users, test_users], ignore_index=True)
all_users = all_users.drop_duplicates(subset=['Profile_ID'])

print(f"Total Unique Users: {len(all_users)}")
print(f"Training Pairs: {len(train_pairs)}")

all_text_data = pd.concat([all_users['Business_Interests'], all_users['Business_Objectives']], axis=0)

tfidf = TfidfVectorizer(max_features=500, stop_words='english')
tfidf.fit(all_text_data)  


interest_matrix = tfidf.transform(all_users['Business_Interests'])
objective_matrix = tfidf.transform(all_users['Business_Objectives'])

svd = TruncatedSVD(n_components=50, random_state=42)

interest_svd = svd.fit_transform(interest_matrix)
objective_svd = svd.fit_transform(objective_matrix)


user_id_to_idx = {uid: idx for idx, uid in enumerate(all_users['Profile_ID'])}

def get_features(pairs_df, users_df):
 

    user_info = users_df.set_index('Profile_ID').to_dict('index')
    
    features = []
    
    print(f"Generating features for {len(pairs_df)} pairs...")
    
    for _, row in pairs_df.iterrows():
        u1, u2 = row['src_user_id'], row['dst_user_id']
        
        if u1 not in user_id_to_idx or u2 not in user_id_to_idx:
            features.append([0]*8) 
            continue
            
        idx1 = user_id_to_idx[u1]
        idx2 = user_id_to_idx[u2]

        info1 = user_info[u1]
        info2 = user_info[u2]
        
        sim_interest = cosine_similarity(
        interest_svd[idx1:idx1+1],
        interest_svd[idx2:idx2+1]
        )[0, 0]

        sim_objective = cosine_similarity(
        objective_svd[idx1:idx1+1],
        objective_svd[idx2:idx2+1]
        )[0, 0]

        cross_match = 0.5 * (
        (objective_matrix[idx1] @ interest_matrix[idx2].T)[0, 0] +
        (objective_matrix[idx2] @ interest_matrix[idx1].T)[0, 0]
        )

        
        b_profile_text = f"{info2['Role']} {info2['Industry']} {info2['Location_City']} {info2['Company_Size_Employees']}"

        cons1_words = set(str(info1['Constraints']).split())
        b_words = set(str(b_profile_text).split())
        constraint_clash_score = len(cons1_words.intersection(b_words))
        
        same_location = 1 if info1['Location_City'] == info2['Location_City'] else 0
        same_industry = 1 if info1['Industry'] == info2['Industry'] else 0
        
        age1 = info1.get("Age", 0) or 0
        age2 = info2.get("Age", 0) or 0

        features.append([
        sim_interest,
        sim_objective,
        cross_match,
        constraint_clash_score,
        same_location,
        same_industry,
        abs(age1 - age2)
        ])

        
    return np.array(features)

X = get_features(train_pairs, all_users)
y = train_pairs['compatibility_score'].values
train_pairs['compatibility_score'].describe()


print("Feature generation complete. No Dimension Mismatch!")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("Training Model...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='mse',
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

print("Training Complete.")

test_user_ids = test_users['Profile_ID'].unique()
test_pairs_list = []

for u1 in test_user_ids:
    for u2 in test_user_ids:
        test_pairs_list.append([u1, u2])



test_pairs_df = pd.DataFrame(test_pairs_list, columns=['src_user_id', 'dst_user_id'])
print(f"Generated {len(test_pairs_df)} test pairs.")

X_test = get_features(test_pairs_df, all_users)

test_preds = model.predict(X_test)

test_preds = np.clip(test_preds, 0, 1)

submission = pd.DataFrame()

submission['ID'] = test_pairs_df['src_user_id'].astype(str) + '_' + test_pairs_df['dst_user_id'].astype(str)
submission['compatibility_score'] = test_preds

submission.to_csv('submission.csv', index=False)

print("SUCCESS: 'submission.csv' saved!")
print(submission.head())
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

val_preds = model.predict(X_val)

mse_score = mean_squared_error(y_val, val_preds)
rmse_score = np.sqrt(mse_score)

print("="*30)
print(f" FINAL VALIDATION MSE: {mse_score:.6f}")
print(f"   (Lower is better. Target is < 0.05 approx)")
print(f"   Root MSE (RMSE):      {rmse_score:.6f}")
print("="*30)

print("\n--- Feature Importance Graph ---")
plt.figure(figsize=(10, 5))
lgb.plot_importance(model, max_num_features=10, height=0.5)
plt.title("What drives Compatibility?")
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(y_val, val_preds, alpha=0.2, color='blue')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.xlabel("Actual Score (Truth)")
plt.ylabel("Predicted Score (Model)")
plt.title("Prediction Accuracy Check")
plt.legend()
plt.grid(True)
plt.show()
import xgboost as xgb

print("Initializing XGBoost...")

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,          
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=50
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

xgb_test_preds = xgb_model.predict(X_test)
xgb_test_preds = np.clip(xgb_test_preds, 0, 1)

sub_xgb = submission.copy()
sub_xgb['compatibility_score'] = xgb_test_preds
sub_xgb.to_csv('submission_xgboost.csv', index=False)

print("XGBoost Model Trained & Saved as 'submission_xgboost.csv'")

