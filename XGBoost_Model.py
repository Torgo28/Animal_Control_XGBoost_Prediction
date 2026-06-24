import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import time

print("=== Final Optimized XGBoost Model ===")
print("Log-transformed XGBoost - Hyperparameter Optimized & Industry Standard")

# Load data
with open('modeling_data.pkl', 'rb') as f:
    modeling_data = pickle.load(f)

animal_data = modeling_data['animal_data']
modeling_features = modeling_data['modeling_features']

print(f"\nDataset: {animal_data.shape[0]:,} samples, {len(modeling_features)} features")
print(f"Features: {modeling_features}")

# Prepare data with log transformation
X = animal_data[modeling_features]
y_original = animal_data['RESPONSE_TIME_MINUTES']
y_log = np.log(y_original + 1)  # Log transform to handle skewness

print(f"\nTarget transformation:")
print(f"  Original skewness: {y_original.skew():.2f} (highly right-skewed)")
print(f"  Log-transformed skewness: {y_log.skew():.2f} (much more normal)")

# Create train/validation/test splits (60/20/20)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_log, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"\nData splits:")
print(f"  Training:   {X_train.shape[0]:,} samples (60%)")
print(f"  Validation: {X_val.shape[0]:,} samples (20%)")
print(f"  Test:       {X_test.shape[0]:,} samples (20%)")

# Optimal XGBoost model (from hyperparameter tuning)
print(f"\nTraining optimized XGBoost...")
start_time = time.time()

final_model = XGBRegressor(
    n_estimators=300,           # Optimal from hyperparameter tuning
    max_depth=10,              # Optimal from hyperparameter tuning  
    learning_rate=0.05,        # Conservative learning rate for stability
    subsample=0.9,             # 90% row sampling (prevents overfitting)
    colsample_bytree=1.0,      # Use all features (optimal for this dataset)
    reg_alpha=0.1,             # L1 regularization
    reg_lambda=2,              # L2 regularization  
    random_state=42,
    n_jobs=-1                  # Use all CPU cores
)

# Fit the model
final_model.fit(X_train, y_train)
training_time = time.time() - start_time

# Make predictions (in log space)
train_pred_log = final_model.predict(X_train)
val_pred_log = final_model.predict(X_val) 
test_pred_log = final_model.predict(X_test)

# Transform back to original scale
train_pred = np.exp(train_pred_log) - 1
val_pred = np.exp(val_pred_log) - 1
test_pred = np.exp(test_pred_log) - 1

train_actual = np.exp(y_train) - 1
val_actual = np.exp(y_val) - 1
test_actual = np.exp(y_test) - 1

# Calculate performance metrics
train_mae = mean_absolute_error(train_actual, train_pred)
val_mae = mean_absolute_error(val_actual, val_pred)
test_mae = mean_absolute_error(test_actual, test_pred)

train_rmse = np.sqrt(mean_squared_error(train_actual, train_pred))
val_rmse = np.sqrt(mean_squared_error(val_actual, val_pred))
test_rmse = np.sqrt(mean_squared_error(test_actual, test_pred))

train_r2 = r2_score(train_actual, train_pred)
val_r2 = r2_score(val_actual, val_pred)
test_r2 = r2_score(test_actual, test_pred)

print(f"Training completed in {training_time:.2f} seconds")

# Performance Analysis
print(f"\n" + "="*60)
print("PERFORMANCE ANALYSIS")
print("="*60)

print(f"\nMAE (Mean Absolute Error):")
print(f"  Training:   {train_mae:.2f} minutes")
print(f"  Validation: {val_mae:.2f} minutes")
print(f"  Test:       {test_mae:.2f} minutes")

print(f"\nRMSE (Root Mean Squared Error):")
print(f"  Training:   {train_rmse:.2f} minutes")
print(f"  Validation: {val_rmse:.2f} minutes") 
print(f"  Test:       {test_rmse:.2f} minutes")

print(f"\nR² (Coefficient of Determination):")
print(f"  Training:   {train_r2:.3f}")
print(f"  Validation: {val_r2:.3f}")
print(f"  Test:       {test_r2:.3f}")

# Overfitting/Underfitting Analysis
train_val_gap = val_mae - train_mae
gap_percent = (train_val_gap / train_mae) * 100

print(f"\n" + "="*60)
print("OVERFITTING/UNDERFITTING ANALYSIS")
print("="*60)

print(f"\nTrain-Validation Gap Analysis:")
print(f"  Training MAE:   {train_mae:.2f} minutes")
print(f"  Validation MAE: {val_mae:.2f} minutes")
print(f"  Gap:            {train_val_gap:+.2f} minutes ({gap_percent:+.1f}%)")

if gap_percent > 20:
    print(f"  ❌ OVERFITTING DETECTED (Gap > 20%)")
    fitting_status = "OVERFITTING"
elif gap_percent < 5 and train_mae > 6:
    print(f"  ❌ UNDERFITTING DETECTED (Small gap but poor performance)")
    fitting_status = "UNDERFITTING"
elif gap_percent < 0:
    print(f"  ⚠️  UNUSUAL: Validation better than training")
    fitting_status = "UNUSUAL"
else:
    print(f"  ✅ GOOD FIT: Healthy generalization gap")
    fitting_status = "GOOD_FIT"

# Accuracy Analysis
test_errors = np.abs(test_actual - test_pred)
within_5 = (test_errors <= 5).mean() * 100
within_10 = (test_errors <= 10).mean() * 100
within_15 = (test_errors <= 15).mean() * 100

print(f"\n" + "="*60)
print("PREDICTION ACCURACY ANALYSIS")
print("="*60)

print(f"\nAccuracy Bands:")
print(f"  Within 5 minutes:  {within_5:.1f}%")
print(f"  Within 10 minutes: {within_10:.1f}%")
print(f"  Within 15 minutes: {within_15:.1f}%")

print(f"\nError Distribution:")
print(f"  Mean error:        {test_errors.mean():.2f} minutes")
print(f"  Median error:      {np.median(test_errors):.2f} minutes")
print(f"  90th percentile:   {np.percentile(test_errors, 90):.2f} minutes")

# Feature Importance
print(f"\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': modeling_features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    print(f"  {i:2d}. {row['feature']:<25}: {row['importance']:.4f} ({row['importance']*100:.1f}%)")

# Business Impact
original_baseline_mae = 7.44  # From initial model without any optimization
improvement_minutes = original_baseline_mae - test_mae
improvement_percent = (improvement_minutes / original_baseline_mae) * 100

print(f"\n" + "="*60)
print("BUSINESS IMPACT ANALYSIS")
print("="*60)

print(f"\nImprovement vs Original Baseline:")
print(f"  Original MAE:     {original_baseline_mae:.2f} minutes")
print(f"  Optimized MAE:    {test_mae:.2f} minutes")
print(f"  Improvement:      {improvement_minutes:.2f} minutes ({improvement_percent:.1f}%)")

annual_incidents = animal_data.shape[0]  # Approximate annual incidents
time_saved_hours = (improvement_minutes * annual_incidents) / 60

print(f"\nEstimated Annual Impact:")
print(f"  Incidents per year:    ~{annual_incidents:,}")
print(f"  Time saved per year:   ~{time_saved_hours:,.0f} hours")
print(f"  Value proposition:     More accurate resource allocation")

# Model Summary
print(f"\n" + "="*60)
print("FINAL MODEL SUMMARY")
print("="*60)

print(f"\n🏆 RECOMMENDED MODEL: Optimized XGBoost Regressor")
print(f"   • Test MAE: {test_mae:.2f} minutes")
print(f"   • {within_5:.1f}% predictions within 5 minutes")
print(f"   • {within_10:.1f}% predictions within 10 minutes")
print(f"   • {improvement_percent:.1f}% improvement over baseline")
print(f"   • Industry standard for tabular data")
print(f"   • Optimal balance of performance and simplicity")
print(f"   • Fitting status: {fitting_status}")
print(f"   • Training time: {training_time:.2f} seconds")

print(f"\n✅ Key Advantages:")
print(f"   • Gradient boosting learns from errors sequentially")
print(f"   • Built-in regularization prevents overfitting")
print(f"   • Handles missing values natively (no imputation needed)")
print(f"   • Excellent performance on mixed-type features")
print(f"   • Industry standard for tabular data problems")
print(f"   • Optimal hyperparameters from systematic tuning")

# Save the final model
model_package = {
    'model': final_model,
    'features': modeling_features,
    'test_mae': test_mae,
    'test_rmse': test_rmse,
    'test_r2': test_r2,
    'feature_importance': feature_importance,
    'hyperparameters': final_model.get_params(),
    'use_log_transform': True,
    'fitting_status': fitting_status,
    'training_time': training_time,
    'accuracy_within_5min': within_5,
    'accuracy_within_10min': within_10
}

with open('final_optimized_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print(f"\n💾 Model saved as 'final_optimized_model.pkl'")
print(f"📋 Ready for assignment submission and deployment!")

# Example prediction
print(f"\n" + "="*60)
print("EXAMPLE PREDICTION")
print("="*60)

sample_idx = 0
sample_features = X_test.iloc[sample_idx:sample_idx+1]
sample_actual = test_actual.iloc[sample_idx]

# Make prediction
sample_pred_log = final_model.predict(sample_features)[0]
sample_pred = np.exp(sample_pred_log) - 1

print(f"\nSample Incident Prediction:")
print(f"  Actual response time:    {sample_actual:.1f} minutes")
print(f"  Predicted response time: {sample_pred:.1f} minutes") 
print(f"  Prediction error:        {abs(sample_actual - sample_pred):.1f} minutes")

print(f"\n🎉 Final model analysis complete!")
print(f"🚀 Model is ready for production deployment!")