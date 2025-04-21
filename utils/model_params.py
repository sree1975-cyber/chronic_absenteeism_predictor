"""
Model Training App with Streamlit
Complete working version with Gradient Boosting support
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_models(data, models_to_train, params):
    """
    Train machine learning models and return results
    Args:
        data: Pandas DataFrame with features and target
        models_to_train: List of model types to train (e.g., ['random_forest'])
        params: Dictionary of model parameters
    Returns:
        Dictionary containing trained models and evaluation reports
    """
    if data.empty:
        raise ValueError("No training data provided")
    
    # Assume last column is target, others are features
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    trained_models = {}
    reports = {}
    
    for model_type in models_to_train:
        try:
            if model_type == 'random_forest':
                model = RandomForestClassifier(**params['random_forest'])
            elif model_type == 'gradient_boosting':
                model = GradientBoostingClassifier(**params['gradient_boosting'])
            elif model_type == 'logistic_regression':
                model = LogisticRegression(**params['logistic_regression'])
            elif model_type == 'neural_network':
                model = MLPClassifier(**params['neural_network'])
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            reports[model_type] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
            trained_models[model_type] = model
            
        except Exception as e:
            st.error(f"Error training {model_type}: {str(e)}")
            continue
    
    return {
        'models': trained_models,
        'reports': reports,
        'feature_names': list(X.columns)
    }

def render_model_params_tab():
    """Render the Model Parameters tab content"""
    st.markdown("<div class='card-title'>⚙️ Model Parameters</div>", unsafe_allow_html=True)
    
    # Model selection and parameters form
    with st.form(key="model_params_form"):
        st.markdown("### Select Model Type")
        
        # Model selection
        model_type = st.selectbox(
            "Machine Learning Model",
            options=[
                "Random Forest",
                "Gradient Boosting",
                "Logistic Regression",
                "Neural Network"
            ],
            index=0
        )
        
        # Convert display name to internal name
        model_type_key = model_type.lower().replace(" ", "_")
        
        # Store the selected model type
        if 'active_model' not in st.session_state or st.session_state.active_model != model_type_key:
            st.session_state.active_model = model_type_key
        
        # Model parameters - we'll show different parameters based on the model type
        st.markdown("### Configure Model Parameters")
        
        # Default parameters based on model type
        if model_type_key == "random_forest":
            n_estimators = st.slider("Number of Trees", 10, 500, 100, 10)
            max_depth = st.slider("Maximum Tree Depth", 2, 30, 10, 1)
            min_samples_split = st.slider("Minimum Samples to Split", 2, 20, 2, 1)
            
            model_params = {
                'random_forest': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'random_state': 42
                }
            }
        
        elif model_type_key == "gradient_boosting":
            n_estimators = st.slider("Number of Boosting Stages", 10, 500, 100, 10)
            learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
            max_depth = st.slider("Maximum Tree Depth", 2, 10, 3, 1)
            
            model_params = {
                'gradient_boosting': {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'random_state': 42
                }
            }
        
        elif model_type_key == "logistic_regression":
            c_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.01)
            max_iter = st.slider("Maximum Iterations", 100, 2000, 1000, 100)
            solver = st.selectbox("Solver", ["lbfgs", "liblinear", "newton-cg", "sag"], index=0)
            
            model_params = {
                'logistic_regression': {
                    'C': c_value,
                    'max_iter': max_iter,
                    'solver': solver,
                    'random_state': 42
                }
            }
        
        elif model_type_key == "neural_network":
            hidden_layer_sizes = st.text_input("Hidden Layer Sizes", "50, 25")
            activation = st.selectbox("Activation Function", ["relu", "tanh", "logistic"], index=0)
            alpha = st.slider("L2 Regularization", 0.0001, 0.01, 0.0001, 0.0001)
            max_iter = st.slider("Maximum Iterations", 100, 2000, 1000, 100)
            
            # Parse the hidden layer sizes
            try:
                hidden_layers = tuple(int(x.strip()) for x in hidden_layer_sizes.split(","))
            except:
                hidden_layers = (50, 25)
            
            model_params = {
                'neural_network': {
                    'hidden_layer_sizes': hidden_layers,
                    'activation': activation,
                    'alpha': alpha,
                    'max_iter': max_iter,
                    'random_state': 42
                }
            }
        
        # Training parameters
        st.markdown("### Training Parameters")
        test_size = st.slider("Test Set Size (%)", 10, 50, 20, 5) / 100
        
        # Train button
        train_button = st.form_submit_button(label="Train Model")
    
    # Handle training process if button is clicked
    if train_button:
        # Check if we have training data (replace with your actual data)
        if 'historical_data' not in st.session_state:
            # Sample data if none exists (replace with your data loading logic)
            st.warning("Using sample data - replace with your actual dataset")
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
            st.session_state.historical_data = pd.DataFrame(X).assign(target=y)
        
        # Set training status
        st.session_state.training_status = "in_progress"
        
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Status message
        status_text = st.empty()
        status_text.text("Starting training process...")
        
        # Simulate training progress
        import time
        for i in range(101):
            if i < 20:
                status_text.text("Preprocessing data...")
            elif i < 40:
                status_text.text("Splitting training and test sets...")
            elif i < 70:
                status_text.text(f"Training {model_type} model...")
            elif i < 90:
                status_text.text("Evaluating model performance...")
            else:
                status_text.text("Finalizing model...")
            
            progress_bar.progress(i)
            time.sleep(0.05)
        
        # Train the model
        status_text.text(f"Training {model_type} model with {len(st.session_state.historical_data)} records...")
        
        # Call train_models with the selected model type and parameters
        training_results = train_models(
            st.session_state.historical_data, 
            models_to_train=[model_type_key],
            params=model_params
        )
        
        if training_results and 'models' in training_results and model_type_key in training_results['models']:
            # Store the model and related information in the session state
            st.session_state.model = training_results['models'][model_type_key]
            st.session_state.feature_names = training_results['feature_names']
            st.session_state.training_report = training_results['reports'][model_type_key]
            
            # Set training status to complete
            st.session_state.training_status = "complete"
            
            # Success message
            st.success(f"✅ {model_type} model trained successfully!")
            
            # Show key metrics
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("Accuracy", f"{training_results['reports'][model_type_key]['accuracy']:.4f}")
            with metrics_col2:
                st.metric("Precision", f"{training_results['reports'][model_type_key]['precision']:.4f}")
            with metrics_col3:
                st.metric("Recall", f"{training_results['reports'][model_type_key]['recall']:.4f}")
            with metrics_col4:
                st.metric("F1 Score", f"{training_results['reports'][model_type_key]['f1_score']:.4f}")
            
        else:
            # Set training status to failed
            st.session_state.training_status = "failed"
            
            # Error message
            st.error("❌ Model training failed. Please check your data and parameters.")

# Run the app
if __name__ == "__main__":
    st.set_page_config(page_title="Model Trainer", layout="wide")
    render_model_params_tab()
