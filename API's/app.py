# # Backend Logics - Flask for API Routes

import os
import pandas as pd
import numpy as np
import csv
from io import StringIO, BytesIO
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt 
import seaborn as sns
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def dfSummary(data):
    summary = pd.DataFrame(data.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Column'] = summary['index']
    summary = summary[['Column','dtypes']]
    summary['non-null'] = data.notnull().sum().values
    summary['Missing'] = data.isnull().sum().values 
    summary['Missing (%)'] = data.isnull().sum().values * 100 / len(data) 
    summary['Uniques'] = data.nunique().values  
    
    summary['dtypes'] = summary['dtypes'].astype(str)
    
    return summary.to_dict(orient='records')

def plot_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return image_base64

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if filename.endswith('.csv'):
            try:
                df = pd.read_csv(file_path, engine='python')
            except Exception as e:
                # Different Encodings
                encodings = ['utf-8', 'latin-1', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except:
                        continue
                else:
                    return jsonify({'error': f'Could not read CSV: {str(e)}'}), 400
        
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        
        summary = dfSummary(df)
        descriptive_stats = {col: {stat: float(val) for stat, val in stats.items()} 
                     for col, stats in df.describe().to_dict().items()}
     
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        fig, axes = plt.subplots(nrows=max(1, len(numeric_columns)//3 + 1), 
                                 ncols=3, 
                                 figsize=(15, 5 * (len(numeric_columns)//3 + 1)))
        axes = axes.flatten() if len(numeric_columns) > 3 else [axes]
        
        for i, col in enumerate(numeric_columns):
            if i < len(axes):
                df[col].hist(ax=axes[i])
                axes[i].set_title(col)
        
        plt.tight_layout()
        num_dist_plot = plot_to_base64(fig)
        
        # Categorical Distribution 
        cat_columns = df.select_dtypes(include=['object']).columns
        cat_plots = {}
        for col in cat_columns:
            fig, ax = plt.subplots(figsize=(10,5))
            df[col].value_counts().plot(kind='bar', ax=ax)
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=90)
            cat_plots[col] = plot_to_base64(fig)
        
        # Correlation Heatmap 
        
        if len(numeric_columns) > 1:
            fig, ax = plt.subplots(figsize=(10,8))
            corr_matrix = df[numeric_columns].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
            plt.title('Correlation Heatmap')
            correlation_plot = plot_to_base64(fig)
        else:
            correlation_plot = None
        
        return jsonify({
            'filename': filename,
            'shape': df.shape,
            'summary': summary,
            'descriptive_stats': descriptive_stats,
            'numerical_distribution': num_dist_plot,
            'categorical_distributions': cat_plots,
            'correlation_plot': correlation_plot
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000) # Invoking the API Routes

# import os
# import pandas as pd
# import numpy as np
# import csv
# from io import StringIO, BytesIO
# from werkzeug.utils import secure_filename
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS

# # Preprocessing and Model Libraries
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# import xgboost as xgb

# # Metrics
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import joblib

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# import os
# import pandas as pd
# import numpy as np
# import csv
# from io import StringIO, BytesIO
# import matplotlib
# matplotlib.use('Agg')  
# import matplotlib.pyplot as plt 
# import seaborn as sns
# import base64
# from werkzeug.utils import secure_filename
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS

# # Preprocessing and Model Libraries
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# import xgboost as xgb

# # Metrics
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import joblib

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def plot_to_base64(fig):
#     buffer = BytesIO()
#     fig.savefig(buffer, format='png')
#     buffer.seek(0)
#     image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
#     plt.close(fig)
#     return image_base64

# def dfSummary(data):
#     summary = pd.DataFrame(data.dtypes, columns=['dtypes'])
#     summary = summary.reset_index()
#     summary['Column'] = summary['index']
#     summary = summary[['Column','dtypes']]
#     summary['non-null'] = data.notnull().sum().values
#     summary['Missing'] = data.isnull().sum().values 
#     summary['Missing (%)'] = data.isnull().sum().values * 100 / len(data) 
#     summary['Uniques'] = data.nunique().values  
    
#     summary['dtypes'] = summary['dtypes'].astype(str)
    
#     return summary.to_dict(orient='records')

# @app.route('/eda', methods=['POST'])
# def perform_eda():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         if filename.endswith('.csv'):
#             try:
#                 df = pd.read_csv(file_path, engine='python')
#             except Exception as e:
#                 # Different Encodings
#                 encodings = ['utf-8', 'latin-1', 'iso-8859-1']
#                 for encoding in encodings:
#                     try:
#                         df = pd.read_csv(file_path, encoding=encoding)
#                         break
#                     except:
#                         continue
#                 else:
#                     return jsonify({'error': f'Could not read CSV: {str(e)}'}), 400
        
#         elif filename.endswith(('.xlsx', '.xls')):
#             df = pd.read_excel(file_path)
        
#         summary = dfSummary(df)
#         descriptive_stats = {col: {stat: float(val) for stat, val in stats.items()} 
#                      for col, stats in df.describe().to_dict().items()}
     
#         # Numerical Distribution
#         numeric_columns = df.select_dtypes(include=[np.number]).columns
#         fig, axes = plt.subplots(nrows=max(1, len(numeric_columns)//3 + 1), 
#                                  ncols=3, 
#                                  figsize=(15, 5 * (len(numeric_columns)//3 + 1)))
#         axes = axes.flatten() if len(numeric_columns) > 3 else [axes]
        
#         for i, col in enumerate(numeric_columns):
#             if i < len(axes):
#                 df[col].hist(ax=axes[i])
#                 axes[i].set_title(col)
        
#         plt.tight_layout()
#         num_dist_plot = plot_to_base64(fig)
        
#         # Categorical Distribution 
#         cat_columns = df.select_dtypes(include=['object']).columns
#         cat_plots = {}
#         for col in cat_columns:
#             fig, ax = plt.subplots(figsize=(10,5))
#             df[col].value_counts().plot(kind='bar', ax=ax)
#             plt.title(f'Distribution of {col}')
#             plt.xticks(rotation=90)
#             cat_plots[col] = plot_to_base64(fig)
        
#         # Correlation Heatmap 
#         if len(numeric_columns) > 1:
#             fig, ax = plt.subplots(figsize=(10,8))
#             corr_matrix = df[numeric_columns].corr()
#             sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
#             plt.title('Correlation Heatmap')
#             correlation_plot = plot_to_base64(fig)
#         else:
#             correlation_plot = None
        
#         return jsonify({
#             'filename': filename,
#             'shape': df.shape,
#             'summary': summary,
#             'descriptive_stats': descriptive_stats,
#             'numerical_distribution': num_dist_plot,
#             'categorical_distributions': cat_plots,
#             'correlation_plot': correlation_plot
#         })
    
#     return jsonify({'error': 'File type not allowed'}), 400

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def preprocess_data(df, target_column, drop_threshold=40):
#     # Summary for identifying columns to drop
#     def dfSummary(data):
#         summary = pd.DataFrame(data.dtypes, columns=['dtypes'])
#         summary = summary.reset_index()
#         summary['Column'] = summary['index']
#         summary = summary[['Column']]
#         summary['Missing'] = data.isnull().sum().values * 100 / len(data) 
#         summary['Uniques'] = data.nunique().values * 100 / len(data)
#         return summary
    
#     # Identify columns to drop
#     dfsum = dfSummary(df)
#     dropped = dfsum[(dfsum['Missing'] >= drop_threshold) | (dfsum['Uniques'] == 100)]
#     list_dropped = dropped.Column.to_list()
    
#     # Drop identified columns
#     new_df = df.drop(list_dropped, axis=1)
    
#     # Handle missing values
#     int_columns = new_df.select_dtypes(np.number).columns
#     obj_columns = new_df.select_dtypes(exclude=np.number).columns
#     new_df[int_columns] = new_df[int_columns].apply(lambda x: x.fillna(x.mean()))
#     new_df[obj_columns] = new_df[obj_columns].apply(lambda x: x.fillna(x.mode()[0]))
    
#     # Strip spaces from object columns
#     for stripSpaces in obj_columns:
#         new_df[stripSpaces] = new_df[stripSpaces].str.replace(' ', '')
    
#     # Encode categorical variables
#     encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
#     objToInt = encoder.fit_transform(new_df[list(obj_columns)])
#     new_df[list(obj_columns)] = pd.DataFrame(objToInt, columns=list(obj_columns))
    
#     return new_df, encoder, list_dropped

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         if filename.endswith('.csv'):
#             try:
#                 df = pd.read_csv(file_path, engine='python')
#             except Exception as e:
#                 # Different Encodings
#                 encodings = ['utf-8', 'latin-1', 'iso-8859-1']
#                 for encoding in encodings:
#                     try:
#                         df = pd.read_csv(file_path, encoding=encoding)
#                         break
#                     except:
#                         continue
#                 else:
#                     return jsonify({'error': f'Could not read CSV: {str(e)}'}), 400
        
#         elif filename.endswith(('.xlsx', '.xls')):
#             df = pd.read_excel(file_path)
        
#         summary = dfSummary(df)
        
#         # Ensure all visualization properties are present, even if empty
#         return jsonify({
#             'filename': filename,
#             'shape': df.shape,
#             'summary': summary,
#             'numerical_distribution': None,
#             'correlation_plot': None,
#             'categorical_distributions': {},
#             'descriptive_stats': {}
#         })
    
#     return jsonify({'error': 'File type not allowed'}), 400

# @app.route('/train_classification', methods=['POST'])
# def train_classification():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
#     target_column = request.form.get('target_column')
#     split_ratio = float(request.form.get('split_ratio', 70))
    
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         # Read file
#         if filename.endswith('.csv'):
#             try:
#                 df = pd.read_csv(file_path, engine='python')
#             except Exception as e:
#                 # Try different encodings
#                 encodings = ['utf-8', 'latin-1', 'iso-8859-1']
#                 for encoding in encodings:
#                     try:
#                         df = pd.read_csv(file_path, encoding=encoding)
#                         break
#                     except:
#                         continue
#                 else:
#                     return jsonify({'error': f'Could not read CSV: {str(e)}'}), 400
        
#         elif filename.endswith(('.xlsx', '.xls')):
#             df = pd.read_excel(file_path)
        
#         # Preprocessing
#         new_df, encoder, dropped_columns = preprocess_data(df, target_column)
        
#         # Split data
#         y = new_df[target_column]
#         X = new_df.drop(target_column, axis=1)
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, 
#             test_size=1-split_ratio/100, 
#             random_state=77, 
#             stratify=y
#         )
        
#         # Define models
#         models = {
#             'Logistic Regression': LogisticRegression(random_state=77, max_iter=10000),
#             'Linear SVC': LinearSVC(random_state=77, dual=False),
#             'K-Nearest Neighbor': KNeighborsClassifier(),
#             'Multinomial Naive Bayes': MultinomialNB(),
#             'Decision Tree': DecisionTreeClassifier(random_state=77),
#             'Random Forest': RandomForestClassifier(random_state=77),
#             'Gradient Boosting': GradientBoostingClassifier(random_state=77),
#             'XGBoost': xgb.XGBClassifier(random_state=77)
#         }
        
#         # Train and evaluate models
#         results = []
#         for name, model in models.items():
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
            
#             results.append({
#                 'Model': name,
#                 'Recall': recall_score(y_test, y_pred, average='macro'),
#                 'Accuracy': accuracy_score(y_test, y_pred),
#                 'Precision': precision_score(y_test, y_pred, average='macro'),
#                 'F1 Score': f1_score(y_test, y_pred, average='macro')
#             })
        
#         # Sort results by F1 Score
#         results_df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False)
        
#         # Save best model and encoder
#         best_model_name = results_df.iloc[0]['Model']
#         best_model = {
#             'Logistic Regression': LogisticRegression(random_state=77, max_iter=10000),
#             'Linear SVC': LinearSVC(random_state=77, dual=False),
#             'K-Nearest Neighbor': KNeighborsClassifier(),
#             'Multinomial Naive Bayes': MultinomialNB(),
#             'Decision Tree': DecisionTreeClassifier(random_state=77),
#             'Random Forest': RandomForestClassifier(random_state=77),
#             'Gradient Boosting': GradientBoostingClassifier(random_state=77),
#             'XGBoost': xgb.XGBClassifier(random_state=77)
#         }[best_model_name]
        
#         best_model.fit(X_train, y_train)
        
#         # Save model, encoder, and drop columns
#         os.makedirs('model_storage', exist_ok=True)
#         joblib.dump(best_model, 'model_storage/best_model.joblib')
#         joblib.dump(encoder, 'model_storage/encoder.pkl')
        
#         with open('model_storage/dropped_columns.txt', 'w') as f:
#             f.write('\n'.join(dropped_columns))
        
#         return jsonify({
#             'results': results,
#             'best_model': best_model_name,
#             'dropped_columns': dropped_columns,
#             'train_shape': X_train.shape,
#             'test_shape': X_test.shape
#         })
    
#     return jsonify({'error': 'File type not allowed'}), 400

# @app.route('/predict_classification', methods=['POST'])
# def predict_classification():
#     if 'file' not in request.files or 'model' not in request.files or 'encoder' not in request.files:
#         return jsonify({'error': 'Missing files'}), 400
    
#     data_file = request.files['file']
#     model_file = request.files['model']
#     encoder_file = request.files['encoder']
    
#     # Save files
#     data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predict_data.csv')
#     model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'model.joblib')
#     encoder_path = os.path.join(app.config['UPLOAD_FOLDER'], 'encoder.pkl')
#     dropped_columns_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dropped_columns.txt')
    
#     data_file.save(data_path)
#     model_file.save(model_path)
#     encoder_file.save(encoder_path)
    
#     # Read data
#     if data_path.endswith('.csv'):
#         df = pd.read_csv(data_path, engine='python')
#     elif data_path.endswith(('.xlsx', '.xls')):
#         df = pd.read_excel(data_path)
    
#     # Read dropped columns
#     with open(dropped_columns_path, 'r') as f:
#         dropped_columns = f.read().splitlines()
    
#     # Drop columns
#     final_data = df.drop(columns=dropped_columns)
    
#     # Load model and encoder
#     model = joblib.load(model_path)
#     encoder = joblib.load(encoder_path)
    
#     # Preprocess data
#     obj_columns = final_data.select_dtypes(exclude=np.number).columns
#     for stripSpaces in obj_columns:
#         final_data[stripSpaces] = final_data[stripSpaces].str.replace(' ', '')
    
#     final_data[list(obj_columns)] = encoder.transform(final_data[list(obj_columns)])
#     final_data.fillna(-1, inplace=True)
    
#     # Predict
#     prediction = model.predict(final_data)
    
#     # Create output dataframe
#     output = pd.DataFrame({'prediction': prediction})
#     final_output = pd.concat([df, output], axis=1)
    
#     # Save prediction to CSV
#     output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction.csv')
#     final_output.to_csv(output_path, index=False)
    
#     return send_file(output_path, as_attachment=True)

# if __name__ == '__main__':
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#     app.run(debug=True, port=5000)

