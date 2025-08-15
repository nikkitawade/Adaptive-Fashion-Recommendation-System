import pandas as pd
import numpy as np
import streamlit as st
import os
import json
import csv
import time
import joblib
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score

#Create directory for models
MODEL_DIR = "user_models"
os.makedirs(MODEL_DIR, exist_ok=True)

#Load training set
df = pd.read_pickle("fashion-dataset/output/fashion_train_embeddings.pkl")
df['product_id'] = df['product_id'].astype(str)

str_cols = [
    'category', 'sub_category', 'article_type', 'color', 'neck_type', 'season',
    'pattern', 'fabric', 'sleeve_length', 'brand', 'sizes', 'dress_length', 'gender', 'age_group', 'usage_group'
]
for col in str_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().str.strip()
        df[col] = df[col].replace('nan', pd.NA)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

#Complementary Colours
def get_complementary_colors(skin_tone, hair_color):
    color_palettes = {
        'warm': ['red', 'orange', 'yellow', 'olive', 'white', 'gold', 'brown'],
        'cool': ['blue', 'purple', 'green', 'black', 'grey', 'silver', 'pink'],
        'neutral': ['white', 'black', 'navy', 'taupe', 'beige', 'grey', 'cream']
    }
    return list(set(color_palettes.get(skin_tone.lower(), [])) | {hair_color.lower()})

#ML Training Function
def prepare_ml_data():
    data, labels = [], []
    for product_id, label in st.session_state.feedback_log.items():
        pid = str(product_id)
        product = df[df['product_id'] == pid]
        if not product.empty:
            data.append(product[str_cols + ['price']].iloc[0])
            labels.append(label)
        else:
            st.warning(f"Product ID {pid} not found in dataset.")
    if not data:
        return None, None, None
    df_train = pd.DataFrame(data)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X = encoder.fit_transform(df_train.fillna('unknown'))

    return X, labels, encoder

SCORE_LOG_FILE = "recommendation_scores.csv"

def log_recommendation_scores(profile_name, df_scores):
    if df_scores.empty:
        return
    log_entries = []
    timestamp = time.time()
    for _, row in df_scores.iterrows():
        log_entries.append({
            'timestamp': timestamp,
            'profile': profile_name,
            'product_id': row['product_id'],
            'score': row['score']
        })

    write_header = not os.path.exists(SCORE_LOG_FILE)
    with open(SCORE_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'profile', 'product_id', 'score'])
        if write_header:
            writer.writeheader()
        writer.writerows(log_entries)
     

def train_model():
    X, y, encoder = prepare_ml_data()
    if X is not None and len(y) >= 10 and len(set(y)) > 1:
        print("nikita")
        # model = lgb.LGBMClassifier(
        #     n_estimators=100,
        #     learning_rate=0.1,
        #     num_leaves=7,
        #     min_data_in_leaf=1,
        #     min_split_gain=0.0,
        #     force_col_wise=True,
        #     random_state=42
        # )
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            max_depth=5,
            min_child_samples=5,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            force_col_wise=True
        )


        st.info(f"Training on {X.shape[0]} samples with {X.shape[1]} features")
        class_counts = pd.Series(y).value_counts().to_dict()
        st.info(f"Class distribution: {class_counts}")

        from sklearn.model_selection import StratifiedKFold, cross_val_score

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        #Optional: evaluate recall or precision
        precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision')
        recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall')

        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)

        st.info(f"Cross-Validated Precision: {avg_precision:.3f}")
        st.info(f"Cross-Validated Recall: {avg_recall:.3f}")

        # Train on full data after evaluation
        model.fit(X, y)

        st.session_state.model = model
        st.session_state.encoder = encoder
        
        # Save model for current profile
        if st.session_state.current_profile_name:
            model_path = os.path.join(MODEL_DIR, f"{st.session_state.current_profile_name}_model.pkl")
            encoder_path = os.path.join(MODEL_DIR, f"{st.session_state.current_profile_name}_encoder.pkl")
            joblib.dump(model, model_path)
            joblib.dump(encoder, encoder_path)
            st.success(f"Model saved for profile: {st.session_state.current_profile_name}")

BODY_TYPE_RULES = {
    'hourglass': {'article_type': ['dresses', 'tops', 'jeans'], 'neck_type': ['v-neck', 'sweetheart','Boat neck','scoop neck'], 'priority': 3.0},
    'pear': {'article_type': ['skirts', 'tops', 'trousers'], 'neck_type': ['scoop neck', 'boat neck', 'square neck'], 'priority': 2.5},
    'apple': {'article_type': ['tshirts', 'track pants', 'jackets'], 'neck_type': ['v-neck', 'Round Neck', 'Cowl neck','one shoulder', 'shoulder straps','strapless'], 'priority': 2.0},
    'rectangle': {'article_type': ['dresses', 'jeans', 'blazers'], 'neck_type': ['High neck', 'Mandarin collar', 'tie-up neck', 'Shirt collar'], 'priority': 1.5}
}


PROFILE_FILE = "saved_profiles.json"
FEEDBACK_FILE = "feedback_logs.json"

def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else {}

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

#Session Init
st.set_page_config(page_title="ML Fashion Advisor", layout="wide")
if 'feedback_log' not in st.session_state:
    st.session_state.feedback_log = {}
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}
if 'current_profile_name' not in st.session_state:
    st.session_state.current_profile_name = None
if 'feedback_actions' not in st.session_state:
    st.session_state.feedback_actions = 0
if 'model' not in st.session_state:
    st.session_state.model = None
if 'encoder' not in st.session_state:
    st.session_state.encoder = None
if 'saved_profiles' not in st.session_state:
    st.session_state.saved_profiles = load_json(PROFILE_FILE)
if 'feedback_logs' not in st.session_state:
    st.session_state.feedback_logs = load_json(FEEDBACK_FILE)
if 'shown_product_ids' not in st.session_state:
    st.session_state.shown_product_ids = set()
if 'recommendation_refresh_count' not in st.session_state:
    st.session_state.recommendation_refresh_count = 0
if 'total_likes' not in st.session_state:
    st.session_state.total_likes = 0
if 'total_shown' not in st.session_state:
    st.session_state.total_shown = 0
if 'prev_pas' not in st.session_state:
    st.session_state.prev_pas = 0.0
if 'current_recommendation_set' not in st.session_state:
    st.session_state.current_recommendation_set = None
if 'current_recommendation_data' not in st.session_state:
    st.session_state.current_recommendation_data = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

# --- [SIDEBAR FORM] ---
with st.sidebar:
    st.header("Personal Style Profile")

    if st.session_state.saved_profiles:
        load_name = st.selectbox("Load Saved Profile", list(st.session_state.saved_profiles.keys()))
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("Load Profile"):
                profile = st.session_state.saved_profiles[load_name]
                feedback = st.session_state.feedback_logs.get(load_name, {})
                st.session_state.user_profile = profile
                st.session_state.feedback_log = {str(k): v for k, v in feedback.items()}
                st.session_state.current_profile_name = load_name
                st.session_state.feedback_actions = 0
                st.session_state.current_recommendation_set = None
                st.session_state.current_recommendation_data = None
                st.session_state.feedback_submitted = False

                if 'profile_metrics' not in st.session_state:
                    st.session_state.profile_metrics = load_json("profile_metrics.json")
                
                metrics = st.session_state.profile_metrics.get(load_name, {})
                st.session_state.total_likes = metrics.get('total_likes', 0)
                st.session_state.total_shown = metrics.get('total_shown', 0)
                st.session_state.prev_pas = metrics.get('prev_pas', 0.0)
                st.session_state.feedback_actions = metrics.get('feedback_actions', 0)
                
                # Try to load saved model
                model_path = os.path.join(MODEL_DIR, f"{load_name}_model.pkl")
                encoder_path = os.path.join(MODEL_DIR, f"{load_name}_encoder.pkl")
                
                if os.path.exists(model_path) and os.path.exists(encoder_path):
                    try:
                        st.session_state.model = joblib.load(model_path)
                        st.session_state.encoder = joblib.load(encoder_path)
                        st.success("Loaded saved model for profile")
                    except Exception as e:
                        st.error(f"Error loading model: {e}")
                        train_model()
                else:
                    st.info("No saved model found, training new model")
                    train_model()
                    
                st.experimental_rerun()

        with col2:
            if st.button("Save Feedback"):
                pname = st.session_state.current_profile_name
                if pname:
                    st.session_state.feedback_logs[pname] = st.session_state.feedback_log.copy()
                    save_json(FEEDBACK_FILE, st.session_state.feedback_logs)
                    
                    # Save model if it exists
                    if st.session_state.get('model') and st.session_state.get('encoder'):
                        model_path = os.path.join(MODEL_DIR, f"{pname}_model.pkl")
                        encoder_path = os.path.join(MODEL_DIR, f"{pname}_encoder.pkl")
                        joblib.dump(st.session_state.model, model_path)
                        joblib.dump(st.session_state.encoder, encoder_path)
                    
                    st.success(f"Feedback and model for '{pname}' saved!")
                else:
                    st.warning("No profile loaded to save feedback.")
        with col3:
            if st.button("❌ Delete", key=f"delete_{load_name}"):
                if load_name in st.session_state.saved_profiles:
                    del st.session_state.saved_profiles[load_name]
                if load_name in st.session_state.feedback_logs:
                    del st.session_state.feedback_logs[load_name]
                if st.session_state.current_profile_name == load_name:
                    st.session_state.current_profile_name = None
                    st.session_state.user_profile = {}
                    st.session_state.feedback_log = {}
                
                # Delete saved models
                model_path = os.path.join(MODEL_DIR, f"{load_name}_model.pkl")
                encoder_path = os.path.join(MODEL_DIR, f"{load_name}_encoder.pkl")
                if os.path.exists(model_path):
                    os.remove(model_path)
                if os.path.exists(encoder_path):
                    os.remove(encoder_path)
                
                save_json(PROFILE_FILE, st.session_state.saved_profiles)
                save_json(FEEDBACK_FILE, st.session_state.feedback_logs)
                st.success(f"Profile '{load_name}' and models deleted!")
                st.experimental_rerun()

    else:
        st.info("No saved profiles yet.")

    st.markdown("---")

    body_shape = st.selectbox("Body Shape", list(BODY_TYPE_RULES.keys())).lower()
    skin_tone = st.selectbox("Skin Tone", ['warm', 'cool', 'neutral'])
    hair_color = st.selectbox("Hair Color", ['black', 'brown', 'blonde', 'red'])
    gender = st.selectbox("Gender", ['women', 'men', 'girls', 'boys', 'unisex']).lower()
    max_price = st.slider("Max Price", 0, 10000, 5000)
    height = st.number_input("Height (cm)", 140, 220, 165)
    weight = st.number_input("Weight (kg)", 40, 150, 60)
    usage_group = st.selectbox("Usage Group", df['usage_group'].dropna().unique())

    if st.button("Generate Recommendations", type="primary"):
        st.session_state.user_profile = {
            'body_shape': body_shape,
            'skin_tone': skin_tone,
            'hair_color': hair_color,
            'gender': gender,
            'max_price': max_price,
            'height': height,
            'weight': weight,
            'usage_group': usage_group if 'usage_group' in df.columns else None
        }
        st.session_state.feedback_actions = 0
        st.session_state.current_recommendation_set = None
        st.session_state.current_recommendation_data = None
        st.session_state.feedback_submitted = False
        for pid_key in list(st.session_state.keys()):
            if pid_key.startswith("fb_"):
                del st.session_state[pid_key]
        st.experimental_rerun()

    st.markdown("Save Current Profile")
    profile_name = st.text_input("Profile Name")
    if st.button("Save Profile"):
        if profile_name.strip():
            name = profile_name.strip()
            st.session_state.saved_profiles[name] = st.session_state.user_profile
            st.session_state.feedback_logs[name] = {
                str(k): v for k, v in st.session_state.feedback_log.items()
            }
            profile_metrics = {
                'total_likes': st.session_state.get('total_likes', 0),
                'total_shown': st.session_state.get('total_shown', 0),
                'prev_pas': st.session_state.get('prev_pas', 0.0),
                'feedback_actions': st.session_state.get('feedback_actions', 0)
            }
            if 'profile_metrics' not in st.session_state:
                st.session_state.profile_metrics = {}
            st.session_state.profile_metrics[name] = profile_metrics
            st.session_state.current_profile_name = name
            
            # Save model if it exists
            if st.session_state.get('model') and st.session_state.get('encoder'):
                model_path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
                encoder_path = os.path.join(MODEL_DIR, f"{name}_encoder.pkl")
                joblib.dump(st.session_state.model, model_path)
                joblib.dump(st.session_state.encoder, encoder_path)
            
            save_json(PROFILE_FILE, st.session_state.saved_profiles)
            save_json(FEEDBACK_FILE, st.session_state.feedback_logs)
            save_json("profile_metrics.json", st.session_state.profile_metrics)
            st.success(f"Profile '{name}' with model and feedback saved!")
        else:
            st.warning("Enter a valid profile name.")

def calculate_pas():
    if st.session_state.total_shown == 0:
        return 0.0
    return st.session_state.total_likes / st.session_state.total_shown

def get_recommendations(df):
    user = st.session_state.user_profile
    feedback_log = st.session_state.feedback_log
    rules = BODY_TYPE_RULES.get(user['body_shape'], {})
    preferred_colors = get_complementary_colors(user['skin_tone'], user['hair_color'])

    df_filtered = df[df['price'] <= user['max_price']].copy()
    gender = user.get('gender', 'unisex')
    df_filtered = df_filtered[df_filtered['gender'].str.lower().isin([gender, 'unisex'])]

    disliked_ids = {str(pid) for pid, label in feedback_log.items() if label == 0}
    shown_ids = st.session_state.shown_product_ids
    df_filtered = df_filtered[~df_filtered['product_id'].isin(disliked_ids | shown_ids)]
    
    if 'usage_group' in df_filtered.columns and user.get('usage_group'):
        df_filtered = df_filtered[df_filtered['usage_group'] == user['usage_group']]
    
    df_filtered['score'] = 0.0

    df_filtered['score'] += (
        df_filtered['neck_type'].isin(rules.get('neck_type', []))
    ).astype(float) * rules.get('priority', 1.0) * 0.8

    df_filtered['score'] += (
        df_filtered['color'].isin(preferred_colors)
    ).astype(float) * 2.0

    if 'weight' in user:
        weight = user['weight']
        size_pref = 'S|M' if weight < 60 else 'M|L' if weight < 80 else 'L|XL'
        df_filtered['score'] += (
            df_filtered['sizes'].str.contains(size_pref, na=False, case=False)
        ).astype(float) * 0.5

    df_filtered['score'] += (
        df_filtered['article_type'].isin(rules.get('article_type', []))
    ).astype(float) * rules.get('priority', 1.0) * 0.8

    model = st.session_state.get('model')
    encoder = st.session_state.get('encoder')
    if model and encoder:
        try:
            # Use the exact same features as during training
            required_features = str_cols + ['price']
            
            # Create temp df with required features
            temp_df = df_filtered[required_features].copy()
            
            # Fill missing values
            temp_df = temp_df.fillna('unknown')
            
            # Transform using encoder
            X_test = encoder.transform(temp_df)
            
            # Predict probabilities
            probs = model.predict_proba(X_test)
            like_probs = probs[:, 1] if probs.shape[1] == 2 else [1.0] * len(X_test)
            df_filtered['score'] += like_probs * 3.0
        except Exception as e:
            st.warning(f"Model prediction error: {e}")

    liked_embeddings = []
    for pid, label in feedback_log.items():
        if label == 1:
            row = df[df['product_id'] == str(pid)]
            if not row.empty and row.iloc[0].get('image_embedding') is not None:
                liked_embeddings.append(row.iloc[0]['image_embedding'])
    if liked_embeddings:
        user_img_emb = np.mean(np.stack(liked_embeddings), axis=0)
        sim_scores = []
        for emb in df_filtered['image_embedding']:
            sim_scores.append(cosine_similarity(user_img_emb, emb) if emb is not None else 0.0)
        df_filtered['score'] += np.array(sim_scores) * 2.0

    return df_filtered.sort_values(by='score', ascending=False).head(10)

#Main Diasplay
if st.session_state.user_profile:
    #Calculate and display PAS
    pas_score = calculate_pas()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Personalization Accuracy", 
                f"{pas_score:.0%}",
                help="Percentage of shown items you've liked")
    
    #Calculate and display FUE
    if st.session_state.feedback_actions > 0:
        fue = (pas_score - st.session_state.prev_pas) / st.session_state.feedback_actions
        with col2:
            st.metric("Feedback Efficiency", 
                    f"{fue:.3f}",
                    delta="Improving" if fue > 0 else None,
                    help="Accuracy improvement per feedback action")    
    st.subheader("Recommended For You")
    
    #Generate new recommendations if needed
    if st.session_state.current_recommendation_set is None or st.session_state.feedback_submitted:
        recs = get_recommendations(df)
        st.session_state.current_recommendation_set = recs['product_id'].tolist()
        st.session_state.current_recommendation_data = recs
        st.session_state.feedback_submitted = False
        st.session_state.shown_product_ids.update(st.session_state.current_recommendation_set)
    
    recs = st.session_state.current_recommendation_data
    
    #Calculate rated count for progress bar
    rated_count = sum(1 for pid in st.session_state.current_recommendation_set 
                      if str(pid) in st.session_state.feedback_log)
    progress_value = rated_count / 10
    st.progress(progress_value)
    st.caption(f"Rated {rated_count} of 10 items")
    
    #Display recommendations
    for idx, row in recs.iterrows():
        pid = row['product_id']
        key = f"fb_{pid}"
        if key not in st.session_state:
            if pid in st.session_state.feedback_log:
                if st.session_state.feedback_log[pid] == 1:
                    st.session_state[key] = "Like"
                elif st.session_state.feedback_log[pid] == 0:
                    st.session_state[key] = "Dislike"
                else:
                    st.session_state[key] = "No Opinion"
            else:
                st.session_state[key] = "No Opinion"

        with st.container():
            cols = st.columns([1, 3, 1])
            with cols[0]:
                if pd.notna(row['local_image_path']) and os.path.exists(row['local_image_path']):
                    st.image(row['local_image_path'], use_column_width=True)
                else:
                    st.warning("Image not available")
            with cols[1]:
                st.markdown(f"### {row['product_name']}")
                st.markdown(
                    f"**Brand**: {row['brand']}  \n"
                    f"**Type**: {row['article_type']}  \n"
                    f"**Color**: {row['color']}  \n"
                    f"**Price**: ${row['price']}"
                )
                st.markdown(
                    f"**Fabric**: {row.get('fabric', 'N/A')}  \n"
                    f"**Pattern**: {row.get('pattern', 'N/A')}  \n"
                    f"**Neckline**: {row.get('neck_type', 'N/A')}  \n"
                    f"**Dress length**: {row.get('dress_length', 'N/A')}"
                )
            with cols[2]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("👍", key=f"like_{pid}"):
                        st.session_state.feedback_log[pid] = 1
                        st.session_state.total_likes += 1
                        st.session_state.total_shown += 1
                        st.session_state.feedback_actions += 1
                        st.session_state[key] = "Like"
                        st.experimental_rerun()
                with col2:
                    if st.button("👎", key=f"dislike_{pid}"):
                        st.session_state.feedback_log[pid] = 0
                        st.session_state.total_shown += 1
                        st.session_state.feedback_actions += 1
                        st.session_state[key] = "Dislike"
                        st.experimental_rerun()
                with col3:
                    if st.button("➖", key=f"noop_{pid}"):
                        if pid in st.session_state.feedback_log:
                            if st.session_state.feedback_log[pid] == 1:
                                st.session_state.total_likes -= 1
                            st.session_state.total_shown -= 1
                            st.session_state.feedback_actions -= 1
                            del st.session_state.feedback_log[pid]
                        st.session_state[key] = "No Opinion"
                        st.experimental_rerun()
                
                #Show current selection
                if st.session_state[key] == "Like":
                    st.success("Liked")
                elif st.session_state[key] == "Dislike":
                    st.error("Disliked")
                else:
                    st.info("No opinion")

    #Submit All Feedback button
    disabled_state = rated_count < 5
    if st.button("Submit All Feedback", disabled=disabled_state, type="primary"):
        st.session_state.feedback_submitted = True
        train_model()
        pname = st.session_state.current_profile_name
        if pname:
            st.session_state.feedback_logs[pname] = st.session_state.feedback_log.copy()
            save_json(FEEDBACK_FILE, st.session_state.feedback_logs)
        st.success("Feedback submitted! Generating new recommendations...")
        st.experimental_rerun()