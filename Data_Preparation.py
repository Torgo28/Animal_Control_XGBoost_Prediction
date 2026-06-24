import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

###################################################
#============Load and initial cleanup=============
###################################################

#Define data types for categorical columns
animal_dtypes = {
    "REQUEST TYPE": "category",
    "SPECIES": "category",
    "BREED": "category",
    "SEX": "category",
    "AGE": "Int64",
    "DISPOSITION": "category"
}
#Load dataset with dtypes defined above
animal_data = pd.read_csv("Animal_Control_Incidents_dataset.csv", dtype=animal_dtypes, low_memory=False)
# Define required attributes
attributes = ["INCIDENT TIME", "DISPATCHED TIME", "ARRIVAL TIME", "AVAILABLE TIME", "REQUEST TYPE", "SPECIES", "BREED", "SEX", "AGE", "DISPOSITION"]
categorical_attributes = ["REQUEST TYPE", "SPECIES", "BREED", "SEX", "DISPOSITION"]
numerical_attributes = ["AGE"]
time_attributes = ["INCIDENT TIME", "DISPATCHED TIME", "ARRIVAL TIME", "AVAILABLE TIME"]

#Remove duplicates
duplicates = animal_data.duplicated().sum()
animal_data = animal_data.drop_duplicates()
post_dupe_removal_count = animal_data.shape[0]

animal_data = animal_data[attributes]

###################################################
#==================Normalize data==================
###################################################

#Normalize categorical data for consistent formatting
animal_data[categorical_attributes] = animal_data[categorical_attributes].apply(
    lambda x: x.str.lower()           # Convert to lowercase
              .str.strip()             # Remove leading/trailing spaces
              .str.replace(r'\s+', ' ', regex=True)  # Replace multiple spaces with single space
              .str.normalize('NFKD')   # Normalize unicode characters
)
#Normalize time data for consistent formatting
animal_data[time_attributes] = animal_data[time_attributes].apply(
    lambda x: x.str.strip()                              # Strip leading/trailing spaces
              .str.replace(r'\s+', ' ', regex=True)      # Replace multiple spaces with single space
              .replace("", np.nan)                       # Replace empty strings with NaN
)

#Normalize numerical data for consistent formatting  
for col in numerical_attributes:
    if animal_data[col].dtype == 'Int64':
        # AGE column is already properly formatted as nullable integer
        pass
    else:
        # Handle any other numerical columns that might have string representations
        animal_data[col] = pd.to_numeric(animal_data[col], errors='coerce')

#Convert back to categorical dtype
for col in categorical_attributes:
    animal_data[col] = animal_data[col].astype('category')

######################################################################################
#===========================CATEGORICAL DATA CLEANUP==================================
######################################################################################

###################################################
#============Fix errors in "SEX" column============
###################################################

#Replace unknown and unspecified entries under "SEX" with NaN
animal_data["SEX"] = animal_data["SEX"].astype(str).replace(["u", "1", "n", "d", "k", "s", "0"], np.nan).astype('category')

# Keep NaN values as-is for Random Forest (can handle missing values natively)
# Missing SEX data may be informative (e.g., wildlife calls more likely to have missing sex info)

###################################################
#============Fix errors in "BREED" column============
###################################################

#Define corrections for breed names
breed_corrections = {
    "bulldog french": "bulldog (french)",
    "bulldog english": "bulldog (english)",
    "english bulldog": "bulldog (english)",
    "english buldog": "bulldog (english)",
    "american bulldog": "bulldog (american)",
    "alaskan  malamute": "alaskan malamute",
    "poodle min/toy": "poodle (min/toy)",
    "poodle -standard": "poodle (standard)",
    "poodle standard": "poodle (standard)",
    "healer": "heeler",
    "xhealer": "xheeler",
    "pitbull": "pit bull",
    "pit": "pit bull",
    "pit bull terrier": "pit bull",
    "x german shepherd": "xgerman shepherd",
    "x retriever": "xretriever",
    "duck domestic": "duck (domestic)",
    "snake domestic non poisonous": "snake (domestic, non-poisonous)",
    "snake domestic poisonous": "snake (domestic, poisonous)",
    "other": "unknown"
    }

#Apply corrections to the "BREED" column
animal_data["BREED"] = animal_data["BREED"].astype(str).replace(breed_corrections).replace("unknown", np.nan).astype('category')
# Keep missing BREED values as NaN for Random Forest (will be handled later)

# Rebuild categorical dtype with sorted categories after corrections (preserve NaN)
animal_data["BREED"] = animal_data["BREED"].astype('category')

###################################################
#============Fix errors for "SPECIES" column============
###################################################

#Define corrections for species names
species_corrections = {
    "birds": "bird",
    "other": "unknown"
}

#Apply corrections to the "SPECIES" column
animal_data["SPECIES"] = animal_data["SPECIES"].astype(str).replace(species_corrections).astype('category')
# Keep missing SPECIES values as NaN for Random Forest (will be handled later)

#Clean species-breed mismatches

#Define species-to-breeds mapping for reclassification. Note: goose is classified as a bird and rabbit is classified as wildlife, not livestock as there is no distinction between wild and domestic goose in the dataset. Fish are classified exotic since they are listed as abondoned animals in the dataset.
species_breeds = {
    "bird": ["parrot", "hawk", "goose", "peacock", "owl", "crow"],
    "cat": ["dsh", "dmh", "siamese", "persian"],
    "dog": ["pit bull", "terrier", "dachshund", "chihuahua", "xchihuahua", "xgerman shepherd", "cairn terrier", "catahoula hound", "great dane", "hound", "husky", "labrador retriever", "german shepherd", "mastiff", "poodle (min/toy)", "poodle (standard)", "saint bernard", "shar pei", "bulldog (english)","beagle", "foxhound", "pointer", "pomeranian", "pug", "retriever", "springer spaniel", "walker hound", "samoyed", "schipperke"],
    "exotic": ["chinchilla", "hamster", "ferret", "guinea pig", "fish"],
    "fowl": ["duck (domestic)", "chicken", "rooster"],
    "livestock": ["duck (domestic)", "chicken", "donkey", "pig", "goat", "horse"],
    "reptile": ["iguana", "alligator", "snake (domestic, non-poisonous)", "snake (domestic, poisonous)", "turtle"],
    "wildlife": ["bat", "possum", "raccoon", "rabbit", "skunk", "armadillo", "squirrel", "mink"]
}

# Apply the species-breeds mapping to correct misclassifications
for species, breeds in species_breeds.items():
    for breed in breeds:
        mask = animal_data["BREED"] == breed
        if mask.any():  # Only update if the breed exists
            animal_data.loc[mask, "SPECIES"] = species
        else:
            print(f"Warning: Breed '{breed}' not found in data")

# Assign all breeds starting with "x" to dog species
x_breed_mask = animal_data["BREED"].str.startswith("x", na=False)
if x_breed_mask.any():
    animal_data.loc[x_breed_mask, "SPECIES"] = "dog"

#Replace "unknown" with NaN for Random Forest
animal_data["SPECIES"] = animal_data["SPECIES"].astype(str).replace("unknown", np.nan).astype('category')

#Rebuild categorical dtype after corrections (preserve NaN)
animal_data["SPECIES"] = animal_data["SPECIES"].astype('category')

###########################################################
#===========Fix errors in "REQUEST TYPE" column============
###########################################################

#Define corrections for request types
request_type_corrections = {
    "barking dog 2nd": "barking dog",
    "barking dog 1st": "barking dog",
    "loose live stock": "loose livestock",
    "pitbull": "pit bull",
    "wild life": "wildlife",
    "stray": "stray animal",
    "other": "unknown",
    "vet transfer": "vet pickup",
    "trap": "trapped animal", #Based on disposition of cases
    "recheck": "cruelty recheck"
}

#Apply corrections to the "REQUEST TYPE" column
animal_data["REQUEST TYPE"] = animal_data["REQUEST TYPE"].astype(str).replace(request_type_corrections).replace("unknown", np.nan).astype('category')

# Keep missing "REQUEST TYPE" values as NaN for Random Forest

#Remove training data rows
training_mask = animal_data["REQUEST TYPE"].str.contains("training example call", case=False, na=False)
training_removed = training_mask.sum()
animal_data = animal_data[~training_mask]

#Rebuild categorical dtype after corrections (preserve NaN)
animal_data["REQUEST TYPE"] = animal_data["REQUEST TYPE"].astype('category')

###########################################################
#===========Fix errors in "DISPOSITION" column=============
###########################################################

# Keep missing "DISPOSITION" values as NaN for Random Forest
animal_data["DISPOSITION"] = animal_data["DISPOSITION"].astype(str).replace("unknown", np.nan).astype('category')

###########################################################
#=======================Inference==========================
###########################################################

#Infer species from request type when species is NaN
inferred_count = 0
unknown_species_mask = animal_data["SPECIES"].isna()

#Infer dog from request type
dog_request_mask = animal_data["REQUEST TYPE"].str.contains("dog", na=False, case=False)
infer_dog = unknown_species_mask & dog_request_mask
if infer_dog.any():
    animal_data.loc[infer_dog, "SPECIES"] = "dog"
    inferred_count += infer_dog.sum()

#Infer cat from request type
cat_request_mask = animal_data["REQUEST TYPE"].str.contains("cat", na=False, case=False)
infer_cat = unknown_species_mask & cat_request_mask
if infer_cat.any():
    animal_data.loc[infer_cat, "SPECIES"] = "cat"
    inferred_count += infer_cat.sum()

#Infer livestock from request type
livestock_request_mask = animal_data["REQUEST TYPE"].str.contains("livestock", na=False, case=False)
infer_livestock = unknown_species_mask & livestock_request_mask
if infer_livestock.any():
    animal_data.loc[infer_livestock, "SPECIES"] = "livestock"
    inferred_count += infer_livestock.sum()

#Infer wildlife from request type
wildlife_request_mask = animal_data["REQUEST TYPE"].str.contains("wildlife", na=False, case=False)
infer_wildlife = unknown_species_mask & wildlife_request_mask
if infer_wildlife.any():
    animal_data.loc[infer_wildlife, "SPECIES"] = "wildlife"
    inferred_count += infer_wildlife.sum()

#Infer fowl from request type
fowl_request_mask = animal_data["REQUEST TYPE"].str.contains("fowl", na=False, case=False)
infer_fowl = unknown_species_mask & fowl_request_mask
if infer_fowl.any():
    animal_data.loc[infer_fowl, "SPECIES"] = "fowl"
    inferred_count += infer_fowl.sum()

#Infer wildlife from disposition "return to wild" if species is still unknown after previous inferences.
return_to_wild_mask = animal_data["DISPOSITION"].str.contains("return to wild", na=False, case=False)
infer_wildlife_disposition = unknown_species_mask & return_to_wild_mask
if infer_wildlife_disposition.any():
    still_unknown = animal_data["SPECIES"].isna()
    final_wildlife_inference = infer_wildlife_disposition & still_unknown
    if final_wildlife_inference.any():
        animal_data.loc[final_wildlife_inference, "SPECIES"] = "wildlife"
        inferred_count += final_wildlife_inference.sum()

print(f"Inferred species from request type for {inferred_count} rows")

#Remove extreme request type vs species mismatches (data errors)
before_mismatch_filter = animal_data.shape[0]
mismatch_count = 0

#Snake requests should only have reptile species
snake_mismatch = (animal_data["REQUEST TYPE"].str.contains("snake", case=False, na=False)) & \
                 (~animal_data["SPECIES"].isin(["reptile"])) & \
                 (animal_data["SPECIES"].notna())  # Only flag non-NaN mismatches
mismatch_count += snake_mismatch.sum()
animal_data = animal_data[~snake_mismatch]

#Pit bull requests should only have dog species  
pitbull_mismatch = (animal_data["REQUEST TYPE"].str.contains("pit bull", case=False, na=False)) & \
                   (~animal_data["SPECIES"].isin(["dog"])) & \
                   (animal_data["SPECIES"].notna())  # Only flag non-NaN mismatches
mismatch_count += pitbull_mismatch.sum()
animal_data = animal_data[~pitbull_mismatch]

print(f"Removed {mismatch_count} records with clear request type vs species mismatches")

###########################################################
#=====Consolidate request types after inference============
###########################################################

#Consolidate trap-related request types (after species inference)
request_type_post_inference_corrections = {
    "deliver cat trap": "deliver trap",
    "deliver dog trap": "deliver trap",
}

animal_data["REQUEST TYPE"] = animal_data["REQUEST TYPE"].astype(str).replace(request_type_post_inference_corrections).astype('category')

#Rebuild categorical dtype after corrections (preserve NaN)
animal_data["REQUEST TYPE"] = animal_data["REQUEST TYPE"].astype('category')

######################################################################################
#============================NUMERICAL DATA CLEANUP===================================
######################################################################################

###################################################
#============Fix errors in "AGE" column============
###################################################

#As 58% of the cleaned data is NaN for age and we are using a RFR model, imputation is not performed. Instead, we will leave NaN values as is for modeling.

###################################################
#============Fix errors in TIME columns============
###################################################

#Clean time columns - replace invalid entries with NaN
# Treating all 0:00 as invalid placeholder data (only 1 potentially legitimate case out of thousands)
time_corrections = {
    "INCIDENT TIME": ["0:00"],             
    "DISPATCHED TIME": ["0:00", ":"],            
    "ARRIVAL TIME": ["0:00", ":"],                
    "AVAILABLE TIME": ["0:00", "0.099305556", "93:33:00"]              
}

# Apply corrections
for col, invalid_values in time_corrections.items():
    animal_data[col] = animal_data[col].replace(invalid_values, np.nan)

#Remove rows where incident times are NaN, as they are the most important factor for response time analysis
before_time_nan_filter = animal_data.shape[0]
animal_data = animal_data.dropna(subset=["INCIDENT TIME"])
after_time_nan_filter = animal_data.shape[0]
time_nan_removed = before_time_nan_filter - after_time_nan_filter
print(f"Removed {time_nan_removed} rows with missing incident times")

# Convert time columns to datetime and extract features for modeling
print("Converting time data and extracting features...")

# Convert to datetime time format and extract hour features
for col in time_attributes:
    # Convert to datetime (will be NaN for invalid times)
    datetime_col = pd.to_datetime(animal_data[col], format='%H:%M', errors='coerce')

#################################################
#============Calculate response times============
#################################################
    
    # Extract hour as numeric feature for modeling
    hour_col_name = col.replace(" ", "_").upper() + "_HOUR" 
    animal_data[hour_col_name] = datetime_col.dt.hour

# Add new time features to numerical attributes list
time_hour_columns = [col.replace(" ", "_").upper() + "_HOUR" for col in time_attributes]
numerical_attributes.extend(time_hour_columns)

print("Converted time data and extracted hour features")

# Calculate response time (target variable for prediction)
print("Calculating response times...")

def calculate_time_difference_minutes(start_col, end_col):
    start_time = pd.to_datetime(animal_data[start_col], format='%H:%M', errors='coerce')
    end_time = pd.to_datetime(animal_data[end_col], format='%H:%M', errors='coerce')
    
    # Calculate difference in minutes
    diff = (end_time - start_time).dt.total_seconds() / 60
    
    # Handle day rollover (negative times assume next day)
    diff = np.where(diff < 0, diff + 1440, diff)  # Add 24 hours (1440 minutes)
    
    return diff

# Create response time metrics
animal_data["RESPONSE_TIME_MINUTES"] = calculate_time_difference_minutes("DISPATCHED TIME", "ARRIVAL TIME")
animal_data["TOTAL_INCIDENT_TIME_MINUTES"] = calculate_time_difference_minutes("INCIDENT TIME", "AVAILABLE TIME") 

# Add response time to numerical attributes (this will be your main target variable)
numerical_attributes.extend(["RESPONSE_TIME_MINUTES", "TOTAL_INCIDENT_TIME_MINUTES"])

print("Calculated response time metrics")

###########################################################
#============DATA QUALITY: Remove Outliers===============
###########################################################

print("\n=== DATA QUALITY: OUTLIER REMOVAL ===")

# Log data quality statistics before outlier removal
before_outlier_removal = animal_data.shape[0]
response_times = animal_data['RESPONSE_TIME_MINUTES']

print(f"Before outlier removal: {before_outlier_removal:,} records")
print(f"Response time statistics:")
print(f"  Mean: {response_times.mean():.2f} minutes")
print(f"  Median: {response_times.median():.2f} minutes")  
print(f"  Max: {response_times.max():.2f} minutes")
print(f"  99th percentile: {response_times.quantile(0.99):.2f} minutes")

# 1. REMOVE IMPOSSIBLE ZERO RESPONSE TIMES
zero_response_mask = (animal_data['RESPONSE_TIME_MINUTES'] == 0)
zero_response_count = zero_response_mask.sum()
animal_data = animal_data[~zero_response_mask]

print(f"\n🧹 Removed {zero_response_count:,} impossible zero response times")

# 2. REMOVE EXTREME OUTLIERS (>12 hours = 720 minutes)
# Based on investigation: 75 cases of 1439 minutes (data entry errors)
extreme_outlier_mask = (animal_data['RESPONSE_TIME_MINUTES'] > 720)
extreme_outlier_count = extreme_outlier_mask.sum()
animal_data = animal_data[~extreme_outlier_mask]

print(f"🚩 Removed {extreme_outlier_count:,} extreme outliers (>12 hours)")

# 3. CHECK FOR NEGATIVE RESPONSE TIMES (should be impossible)
negative_response_mask = (animal_data['RESPONSE_TIME_MINUTES'] < 0)
negative_response_count = negative_response_mask.sum()
if negative_response_count > 0:
    animal_data = animal_data[~negative_response_mask]
    print(f"🚩 Removed {negative_response_count:,} negative response times")

# Log final outlier removal results
after_outlier_removal = animal_data.shape[0]
total_outliers_removed = before_outlier_removal - after_outlier_removal

print(f"\n📊 OUTLIER REMOVAL SUMMARY:")
print(f"  Records before: {before_outlier_removal:,}")
print(f"  Records after:  {after_outlier_removal:,}")
print(f"  Total removed: {total_outliers_removed:,} ({(total_outliers_removed/before_outlier_removal)*100:.1f}%)")

# Show improved data quality statistics
clean_response_times = animal_data['RESPONSE_TIME_MINUTES']
print(f"\n✅ IMPROVED DATA QUALITY:")
print(f"  Mean: {clean_response_times.mean():.2f} minutes")
print(f"  Median: {clean_response_times.median():.2f} minutes")
print(f"  Max: {clean_response_times.max():.2f} minutes") 
print(f"  99th percentile: {clean_response_times.quantile(0.99):.2f} minutes")

improvement_ratio = response_times.mean() / clean_response_times.mean()
print(f"  Mean response time improved by {((improvement_ratio-1)*100):+.1f}%")

# Remove rows without complete time data needed for response time calculation
before_time_filter = animal_data.shape[0]
# Keep only records that have the essential time data for calculating response times
time_complete_mask = (animal_data["INCIDENT TIME"].notna() & 
                     animal_data["DISPATCHED TIME"].notna() & 
                     animal_data["ARRIVAL TIME"].notna())
animal_data = animal_data[time_complete_mask]
after_time_filter = animal_data.shape[0]
time_removed = before_time_filter - after_time_filter
print(f"Removed {time_removed} rows without complete time data for response time calculation")

# Remove rows without response time data (can't train without target variable)
before_response_filter = animal_data.shape[0]
animal_data = animal_data[animal_data["RESPONSE_TIME_MINUTES"].notna()]
after_response_filter = animal_data.shape[0]
response_removed = before_response_filter - after_response_filter
print(f"Removed {response_removed} rows without valid response time data for modeling")

# Final data preprocessing for modeling
print("\nStarting categorical encoding...")

# Use Label Encoding for high-cardinality categorical variables
# (Label Encoding is preferred over One-Hot for Random Forest with many categories)

# Store original categorical data for reference
categorical_original = animal_data[categorical_attributes].copy()

# Use Label Encoding for high-cardinality categorical variables (excluding BREED which will be addressed via clustering)
label_encoders = {}
for col in categorical_attributes:
    if col == "BREED":
        print(f"Skipped label encoding for {col}: Using BREED_COMPLEXITY_CLUSTER instead")
        continue
        
    # Convert categorical to string first, then encode
    temp_data = animal_data[col].astype(str).fillna('MISSING')
    
    # Create label encoder and fit
    le_col = LabelEncoder()
    animal_data[col + '_ENCODED'] = le_col.fit_transform(temp_data)
    
    # Replace encoded 'MISSING' back to NaN for Random Forest (only if MISSING exists)
    if 'MISSING' in le_col.classes_:
        missing_encoded_value = le_col.transform(['MISSING'])[0]
        animal_data[col + '_ENCODED'] = animal_data[col + '_ENCODED'].replace(missing_encoded_value, np.nan)
    
    label_encoders[col] = le_col
    print(f"Label encoded {col}: {len(le_col.classes_)} unique categories")

###########################################################
#============CLUSTERING-BASED FEATURE ENGINEERING========
###########################################################

print("\nStarting clustering-based feature engineering...")

# CLUSTERING 1: Breed Complexity Clustering
print("Creating breed complexity clusters...")

# Calculate breed response statistics for clustering
breed_stats = animal_data.groupby('BREED', observed=True).agg({
    'RESPONSE_TIME_MINUTES': ['mean', 'median', 'std', 'count']
}).round(2)

breed_stats.columns = ['mean_response', 'median_response', 'std_response', 'incident_count']
# Only use breeds with sufficient data for reliable clustering
breed_stats = breed_stats[breed_stats['incident_count'] >= 10]

print(f"Clustering {len(breed_stats)} breeds with 10+ incidents...")

# Prepare clustering features (mean, median, std of response times)
clustering_features = breed_stats[['mean_response', 'median_response', 'std_response']].fillna(0)

# Standardize features for clustering
scaler = StandardScaler()
clustering_scaled = scaler.fit_transform(clustering_features)

# Perform K-means clustering to group breeds by complexity
n_breed_clusters = 4  # Easy, Moderate, Complex, Very Complex
breed_kmeans = KMeans(n_clusters=n_breed_clusters, random_state=42, n_init=10)
breed_clusters = breed_kmeans.fit_predict(clustering_scaled)

# Add cluster labels to breed stats
breed_stats['complexity_cluster'] = breed_clusters

# Create breed complexity mapping
breed_complexity_map = dict(zip(breed_stats.index, breed_stats['complexity_cluster']))

# Add complexity cluster to main dataset
animal_data['BREED_COMPLEXITY_CLUSTER'] = animal_data['BREED'].map(breed_complexity_map)
animal_data['BREED_COMPLEXITY_CLUSTER'] = animal_data['BREED_COMPLEXITY_CLUSTER'].fillna(-1)  # Unknown breeds

# Print cluster analysis
print("Breed complexity clusters created:")
for cluster_id in range(n_breed_clusters):
    cluster_breeds = breed_stats[breed_stats['complexity_cluster'] == cluster_id]
    avg_response = cluster_breeds['mean_response'].mean()
    breed_count = len(cluster_breeds)
    sample_breeds = cluster_breeds.head(3).index.tolist()
    print(f"  Cluster {cluster_id}: {breed_count} breeds, {avg_response:.1f} min avg response")
    print(f"    Sample breeds: {', '.join(sample_breeds)}")

# CLUSTERING 2: Temporal Pattern Clustering  
print("\nCreating temporal pattern clusters...")

# Calculate hourly response patterns
hourly_stats = animal_data.groupby('INCIDENT_TIME_HOUR', observed=True).agg({
    'RESPONSE_TIME_MINUTES': ['mean', 'median', 'count']
}).round(2)

hourly_stats.columns = ['mean_response', 'median_response', 'incident_count']

# Prepare temporal clustering features
temporal_features = hourly_stats[['mean_response', 'incident_count']].fillna(0)

# Standardize and cluster
temporal_scaler = StandardScaler()
temporal_scaled = temporal_scaler.fit_transform(temporal_features)
n_time_clusters = 3  # Peak, Normal, Off-Peak
temporal_kmeans = KMeans(n_clusters=n_time_clusters, random_state=42, n_init=10)
temporal_clusters = temporal_kmeans.fit_predict(temporal_scaled)

hourly_stats['time_pattern_cluster'] = temporal_clusters

# Create temporal pattern mapping
hour_pattern_map = dict(zip(hourly_stats.index, hourly_stats['time_pattern_cluster']))

# Add temporal pattern to main dataset
animal_data['TIME_PATTERN_CLUSTER'] = animal_data['INCIDENT_TIME_HOUR'].map(hour_pattern_map)

# Print temporal cluster analysis
print("Temporal pattern clusters created:")
for cluster_id in range(n_time_clusters):
    cluster_hours = hourly_stats[hourly_stats['time_pattern_cluster'] == cluster_id]
    avg_response = cluster_hours['mean_response'].mean()
    total_incidents = cluster_hours['incident_count'].sum()
    hour_range = f"{cluster_hours.index.min()}-{cluster_hours.index.max()}"
    print(f"  Cluster {cluster_id}: Hours {hour_range}, {avg_response:.1f} min avg, {total_incidents:,.0f} total incidents")

# Add clustering features to numerical attributes
clustering_features = ['BREED_COMPLEXITY_CLUSTER', 'TIME_PATTERN_CLUSTER']
numerical_attributes.extend(clustering_features)

print(f"✅ Added {len(clustering_features)} clustering-based features")
print("Clustering-based feature engineering complete!")

# Create final feature list for modeling (include encoded categorical features except BREED)
encoded_categorical = [col + '_ENCODED' for col in categorical_attributes if col != "BREED"]
modeling_features = numerical_attributes + encoded_categorical

# Remove response time features from feature list (these are target variables)
modeling_features = [col for col in modeling_features if 'TIME_MINUTES' not in col]

print(f"\nFinal feature count for modeling: {len(modeling_features)} features")
print("Features included:")
for i, feature in enumerate(modeling_features, 1):
    print(f"  {i}. {feature}")
print(f"Target variable: RESPONSE_TIME_MINUTES")

# Display final data summary
print(f"\nFinal dataset shape: {animal_data.shape}")
print(f"Response time statistics:")
print(animal_data['RESPONSE_TIME_MINUTES'].describe())

print(f"\nSpecies/Breed distribution in final dataset:")
print(f"SPECIES - Known: {animal_data['SPECIES'].notna().sum()}, Missing: {animal_data['SPECIES'].isna().sum()}")
print(f"BREED - Known: {animal_data['BREED'].notna().sum()}, Missing: {animal_data['BREED'].isna().sum()}")


#########################
#Reviews
#########################

#Identify breed mismatches after correction
species_breed_counts = animal_data.groupby(["SPECIES", "BREED"], observed=True).size().reset_index(name='count')
# Sort by species and breed alphabetically
species_breed_counts = species_breed_counts.sort_values(["SPECIES", "BREED"])

#Identify request-breed-species mismatches after correction
request_breed_species_counts = animal_data.groupby(["REQUEST TYPE", "BREED", "SPECIES"], observed=True).size().reset_index(name='count')
#Sort by request type, breed, and species alphabetically
request_breed_species_counts = request_breed_species_counts.sort_values(["REQUEST TYPE", "BREED", "SPECIES"])

#Prints

# Check for NaN values in categorical columns
print("\n=== NaN Check for Categorical Data ===")
for col in categorical_attributes:
    nan_count = animal_data[col].isna().sum()
    print(f"{col}: {nan_count} NaN values")

print("Number of duplicates removed:", duplicates)
print("Number of records after all cleaning steps:", animal_data.shape[0])

#######################
# Data preparation complete - moving to model training phase


##################################################
#============Export TIME data for analysis========
##################################################

#Export time data for analysis before any cleaning

print("Creating time analysis CSVs...")

# Data validation complete - time analysis performed

# Export records with 0:00 times for inspection  
zero_time_records = animal_data[
    (animal_data["INCIDENT TIME"] == "0:00") |
    (animal_data["DISPATCHED TIME"] == "0:00") | 
    (animal_data["ARRIVAL TIME"] == "0:00") |
    (animal_data["AVAILABLE TIME"] == "0:00")
]["REQUEST TYPE"]

# Create summary statistics
time_summary = []
for col in time_attributes:
    zero_count = (animal_data[col] == "0:00").sum()
    null_count = animal_data[col].isna().sum()
    total = len(animal_data)
    
    time_summary.append({
        "Column": col,
        "Zero_Count": zero_count, 
        "Null_Count": null_count,
        "Valid_Times": total - zero_count - null_count,
        "Total_Records": total,
        "Percent_Zero": round(zero_count/total*100, 2),
        "Percent_Null": round(null_count/total*100, 2)
    })

# Create summary statistics for analysis
time_summary_df = pd.DataFrame(time_summary)

# Enhanced final summary with species/breed distribution
print(f"\nENHANCED FINAL DATASET SUMMARY:")
print(f"Final dataset shape: {animal_data.shape}")
print(f"Species/Breed distribution in final dataset:")
print(f"SPECIES - Known: {animal_data['SPECIES'].notna().sum()}, Missing (NaN): {animal_data['SPECIES'].isna().sum()}")
print(f"BREED - Known: {animal_data['BREED'].notna().sum()}, Missing (NaN): {animal_data['BREED'].isna().sum()}")

##################################################
#============Export prepared data for modeling===
##################################################

print("\nSaving prepared data for modeling...")

# Save the cleaned dataset
# Create a dictionary with all the modeling variables
modeling_data = {
    'animal_data': animal_data,
    'modeling_features': modeling_features,
    'target_variable': 'RESPONSE_TIME_MINUTES',
    'label_encoders': label_encoders,
    'categorical_attributes': categorical_attributes,
    'numerical_attributes': numerical_attributes
}

# Save to pickle file
with open('modeling_data.pkl', 'wb') as f:
    pickle.dump(modeling_data, f)

print("✅ Data preparation complete - saved modeling_data.pkl")
print("✅ Ready for model training phase")

