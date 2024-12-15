# Backend Logics - Flask for API Routes

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

