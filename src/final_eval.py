import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder

#Load Data
@st.cache_data
def load_all_data():
    # Load test and train data
    test_df = pd.read_pickle("fashion-dataset/output/fashion_test_embeddings.pkl")
    train_df = pd.read_pickle("fashion-dataset/output/fashion_train_embeddings.pkl")
    
    #Convert product_id to string for consistency
    test_df['product_id'] = test_df['product_id'].astype(str)
    train_df['product_id'] = train_df['product_id'].astype(str)
    
    # Define string columns to normalize (same as recommender system)
    str_cols = [
        'category', 'sub_category', 'article_type', 'color', 'neck_type', 'season',
        'pattern', 'fabric', 'sleeve_length', 'brand', 'sizes', 'dress_length', 
        'gender', 'age_group', 'usage_group'
    ]
    
    # Normalize string columns (same as recommender system)
    for col in str_cols:
        if col in test_df.columns:
            test_df[col] = test_df[col].astype(str).str.lower().str.strip()  # Fixed typo: astype instead of ast
            test_df[col] = test_df[col].replace('nan', pd.NA)
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str).str.lower().str.strip()
            train_df[col] = train_df[col].replace('nan', pd.NA)
    
    # Load feedback logs and profiles
    feedback_logs = {}
    if os.path.exists("feedback_logs.json"):
        with open("feedback_logs.json", "r") as f:
            feedback_logs = json.load(f)
    
    saved_profiles = {}
    if os.path.exists("saved_profiles.json"):
        with open("saved_profiles.json", "r") as f:
            saved_profiles = json.load(f)
    
    # Load models if available
    user_models = {}
    model_dir = "user_models"
    if os.path.exists(model_dir):
        for user_id in saved_profiles.keys():
            model_path = os.path.join(model_dir, f"{user_id}_model.pkl")
            encoder_path = os.path.join(model_dir, f"{user_id}_encoder.pkl")
            if os.path.exists(model_path) and os.path.exists(encoder_path):
                user_models[user_id] = {
                    "model": joblib.load(model_path),
                    "encoder": joblib.load(encoder_path)
                }
    
    return test_df, train_df, feedback_logs, saved_profiles, user_models

#Cosine similarity
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Custom cosine similarity function (same as recommender system)"""
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# Build User Vector
def build_user_vector(user_id, train_df, feedback_logs):
    # Get liked product IDs (keep as strings)
    liked_ids = [pid for pid, val in feedback_logs.get(user_id, {}).items() if val == 1]
    
    # Get liked rows
    liked_rows = train_df[train_df['product_id'].isin(liked_ids)]

    if liked_rows.empty:
        st.error(f"No liked product embeddings for user: {user_id}")
        return None, None

    # Stack embeddings and calculate mean
    embeddings = [row['image_embedding'] for _, row in liked_rows.iterrows() 
                 if row['image_embedding'] is not None]
    
    if not embeddings:
        st.error(f"No valid embeddings found for user: {user_id}")
        return None, None
        
    user_vector = np.mean(np.stack(embeddings), axis=0)
    return user_vector, liked_rows

# Content-Based Recommendations
def get_content_based_recommendations(user_vector, test_df, k=2):
    # Calculate similarity scores using our custom function
    similarity_scores = []
    for _, row in test_df.iterrows():
        if row['image_embedding'] is not None:
            sim = cosine_similarity(user_vector, row['image_embedding'])
        else:
            sim = 0.0
        similarity_scores.append(sim)
    
    test_df = test_df.copy()
    test_df['similarity'] = similarity_scores
    return test_df.sort_values(by='similarity', ascending=False).head(k)

# Model-Based Recommendations
def get_model_based_recommendations(user_id, test_df, user_models, k=2):
    # Check if model exists for user
    if user_id not in user_models:
        st.warning(f"No trained model found for user: {user_id}")
        return None
    
    model = user_models[user_id]["model"]
    encoder = user_models[user_id]["encoder"]
    
    # Prepare test data
    str_cols = [
        'category', 'sub_category', 'article_type', 'color', 'neck_type', 'season', 
        'pattern', 'fabric', 'sleeve_length', 'brand', 'sizes', 
        'dress_length', 'gender', 'age_group', 'usage_group'
    ]
    
    # Create a copy to avoid modifying original
    test_copy = test_df.copy()
    
    # Handle missing columns
    for col in str_cols + ['price']:
        if col not in test_copy.columns:
            test_copy[col] = 'unknown'
    
    # Fill missing values
    test_copy[str_cols] = test_copy[str_cols].fillna('unknown')
    test_copy['price'] = test_copy['price'].fillna(0)
    
    # Encode features
    try:
        X_test = encoder.transform(test_copy[str_cols + ['price']])
    except Exception as e:
        st.error(f"Error encoding test data: {e}")
        return None
    
    # Predict probabilities
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            if probs.shape[1] >= 2:  # Check if we have at least 2 classes
                test_copy['score'] = probs[:, 1]  # Probability of positive class
            else:
                test_copy['score'] = probs[:, 0]
        else:
            test_copy['score'] = model.predict(X_test)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None
    
    return test_copy.sort_values(by='score', ascending=False).head(k)




def precision_at_k(recommendations, selected_article_type, selected_usage_group):
    """
    Precision@K: fraction of recommended items that match BOTH selected article_type AND usage_group.
    """
    if recommendations.empty:
        return 0.0
    
    hits = 0
    for _, rec_row in recommendations.iterrows():
        article_match = (selected_article_type and rec_row['article_type'] == selected_article_type)
        usage_match = (selected_usage_group and rec_row['usage_group'] == selected_usage_group)
        
        if article_match and usage_match:
            hits += 1
    
    return hits / len(recommendations)
def recall_at_k(recommendations, selected_article_type, selected_usage_group, k, test_df):
    """
    Recall@K: fraction of relevant items retrieved in top-K out of all relevant items in the test set.
    Relevant = matches selected_article_type OR selected_usage_group.
    """
    if recommendations.empty:
        return 0.0


    total_relevant = sum(
        bool(selected_article_type) and (row['article_type'] == selected_article_type) and
        bool(selected_usage_group) and (row['usage_group'] == selected_usage_group)
        for _, row in test_df.iterrows()
    )
    if total_relevant == 0:
        return 0.0

    # Count relevant items in top-K
    top_k = recommendations.head(k)
    hits = sum(
        bool(selected_article_type and row['article_type'] == selected_article_type) and
        bool(selected_usage_group and row['usage_group'] == selected_usage_group)
        for _, row in top_k.iterrows()
    )

    return hits / total_relevant

# F1@K
def f1_at_k(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

#MAMS
def multi_attribute_match_score(recommendations, user_preferences):
    scores = []
    total_features = len([v for v in user_preferences.values() if v])  # count non-empty filters

    for _, row in recommendations.iterrows():
        match_count = 0
        for attr, pref_value in user_preferences.items():
            if not pref_value:
                continue
                
            if attr == "price":
                if "<" in str(pref_value):
                    try:
                        limit = float(pref_value.replace("<", "").strip())
                        if row[attr] < limit:
                            match_count += 1
                    except ValueError:
                        pass
                elif ">" in str(pref_value):
                    try:
                        limit = float(pref_value.replace(">", "").strip())
                        if row[attr] > limit:
                            match_count += 1
                    except ValueError:
                        pass
                else:
                    try:
                        if float(row[attr]) == float(pref_value):
                            match_count += 1
                    except (ValueError, TypeError):
                        pass

            else:
                if str(row[attr]).lower() == str(pref_value).lower():
                    match_count += 1

        score = match_count / total_features if total_features > 0 else 0
        scores.append(score)

    recommendations['match_score'] = scores
    return recommendations

#Dropdown Options
def get_filtered_dropdowns(df, selected_article_type, selected_usage_group):
    # Filter by both article_type and usage_group if provided
    filtered_df = df.copy()
    if selected_article_type:
        filtered_df = filtered_df[filtered_df['article_type'] == selected_article_type]
    if selected_usage_group:
        filtered_df = filtered_df[filtered_df['usage_group'] == selected_usage_group]

    dropdowns = {}
    attributes = ["color", "pattern", "sleeve_length", "dress_length", "fabric", "neck_type"]

    for attr in attributes:
        unique_values = filtered_df[attr].dropna().unique()
        unique_values = [v for v in unique_values if v and str(v).strip()]
        unique_values = sorted(set(unique_values))
        
        if unique_values:
            dropdowns[attr] = [""] + unique_values

    return dropdowns

#Streamlit UI
def main():
    st.title("👗 Fashion Recommendation Evaluation")
    st.markdown("Evaluate **Precision@K**, **Recall@K**, and **Multi-Attribute Match Score** for personalized recommendations.")

    # Load data with consistent preprocessing
    test_df, train_df, feedback_logs, saved_profiles, user_models = load_all_data()
    
    #User ID selection
    user_id = st.selectbox("Select User ID:", list(saved_profiles.keys()))
    
    #Recommendation Method
    method = st.radio("Recommendation Method:", 
                     ["Content-Based (Image Embeddings)", "Model-Based (LightGBM/Random Forest)"],
                     index=1)

    #Top-K
    k = st.slider("Top-K Recommendations:", 1, 200, 10)

    #Step 1: Article Type Selection
    article_types = sorted(test_df['article_type'].dropna().unique())
    selected_article = st.selectbox("Select Article Type:", article_types)

    #Step 1b: Usage Group Selection
    usage_groups = sorted(test_df['usage_group'].dropna().unique())
    selected_usage_group = st.selectbox("Select Usage Group:", [""] + usage_groups)

    #Step 2: Dynamic Attribute Dropdowns
    st.subheader("🎛 Select Preferences (based on article type and usage group)")
    preferences = {"article_type": selected_article, "usage_group": selected_usage_group}

    if selected_article or selected_usage_group:
        dropdowns = get_filtered_dropdowns(test_df, selected_article, selected_usage_group)

        # Create dropdowns for each attribute
        for attr, options in dropdowns.items():
            label = attr.replace('_', ' ').title()
            preferences[attr] = st.selectbox(f"Select {label}:", options)

        # Price filter
        price_input = st.text_input("Price Filter (e.g., <1000 or >500):", "")
        preferences["price"] = price_input.strip()

    #Run Evaluation
    if st.button("Evaluate Recommendations"):
        user_vector, liked_rows = build_user_vector(user_id, train_df, feedback_logs)
        
        if user_vector is None:
            st.error("Cannot build user vector. Not enough liked items.")
            return
            
        # Select recommendation method
        if method == "Content-Based (Image Embeddings)":
            recommendations = get_content_based_recommendations(user_vector, test_df, k=k)
            method_name = "Content-Based"
        else:
            recommendations = get_model_based_recommendations(user_id, test_df, user_models, k=k)
            method_name = "Model-Based"
            
            if recommendations is None:
                st.error("Could not generate model-based recommendations.")
                return

        # Calculate metrics with updated definitions
        precision = precision_at_k(recommendations, selected_article, selected_usage_group)
        # recall = recall_at_k(recommendations, selected_article, selected_usage_group)
        recall = recall_at_k(recommendations, selected_article, selected_usage_group, k, test_df)
        f1 = f1_at_k(precision, recall)

        # Multi-attribute score
        recommendations = multi_attribute_match_score(recommendations, preferences)
        avg_match_score = recommendations['match_score'].mean()

        # Display results
        st.subheader(f"Evaluation Results ({method_name} Method)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"Precision@{k}", f"{precision:.2%}")
        with col2:
            st.metric(f"Recall@{k}", f"{recall:.2%}")
        with col3:
            st.metric("F1 Score", f"{f1:.2%}")
        with col4:
            st.metric("Attribute Match", f"{avg_match_score:.2%}")
        
        # Show recommendations
        st.subheader(f"Top {k} Recommendations")
        display_cols = ['product_id', 'article_type', 'usage_group', 'color', 'fabric', 
                       'pattern', 'price', 'match_score']
        
        # Add method-specific score column
        if method == "Content-Based (Image Embeddings)":
            display_cols.insert(1, 'similarity')
        else:
            display_cols.insert(1, 'score')
            
        st.dataframe(recommendations[display_cols].sort_values(display_cols[1], ascending=False))

#Run App
if __name__ == "__main__":
    main()
