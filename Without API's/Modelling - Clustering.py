import streamlit as st
import pandas as pd
import numpy as np
import csv
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from zipfile import ZipFile
from io import BytesIO
import joblib

def read_data(data):
    """
    Read different file types with proper delimiter detection
    """
    filename = data.name
    x = filename.rsplit('.', 1)

    # Read CSV
    if x[1] == 'csv':
        stringio = StringIO(data.getvalue().decode("utf-8"))
        string_data = stringio.read()
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(string_data)
        df = pd.read_csv(data, sep=dialect.delimiter)
    # Read Excel
    else:
        df = pd.read_excel(data)
    
    return df

def perform_clustering(df, feature_columns, max_clusters, scaling):
    """
    Perform clustering and evaluation
    """
    # Select and preprocess features
    X = df[feature_columns]
    X = X.fillna(X.mean())

    # Scaling
    if scaling:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = None
        X_scaled = X.values

    # Clustering Algorithms
    clustering_methods = {
        'K-Means': KMeans(n_clusters=max_clusters, random_state=42),
        'Hierarchical Clustering': AgglomerativeClustering(n_clusters=max_clusters),
        'Gaussian Mixture': GaussianMixture(n_components=max_clusters, random_state=42),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
    }

    results = []

    for name, model in clustering_methods.items():
        try:
            # Special handling for DBSCAN as it doesn't require preset number of clusters
            if name == 'DBSCAN':
                labels = model.fit_predict(X_scaled)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            else:
                labels = model.fit_predict(X_scaled)
                n_clusters = len(np.unique(labels))

            # Skip evaluation if only one cluster
            if n_clusters > 1:
                silhouette = silhouette_score(X_scaled, labels)
                calinski = calinski_harabasz_score(X_scaled, labels)
                davies = davies_bouldin_score(X_scaled, labels)

                results.append({
                    'Method': name,
                    'Number of Clusters': n_clusters,
                    'Silhouette Score': silhouette,
                    'Calinski-Harabasz Index': calinski,
                    'Davies-Bouldin Index': davies
                })
        except Exception as e:
            st.warning(f"Could not evaluate {name}: {e}")

    return results, X_scaled, scaler

def main():
    st.title('Clustering Model Selector')

    # File Uploader
    data = st.file_uploader('Upload your data here', type=['csv','xlsx', 'xls'])

    if data is not None:
        # Read data
        df = read_data(data)

        # Data Preprocessing
        st.write('Data Preprocessing')
        
        # Select features for clustering
        feature_columns = st.multiselect(
            'Select features for clustering', 
            options=df.select_dtypes(include=[np.number]).columns.tolist()
        )
        
        # Max number of clusters
        max_clusters = st.slider('Maximum number of clusters', 2, 10, 5)

        # Preprocessing options
        scaling = st.checkbox('Standardize features', value=True)

        # Perform clustering when features are selected
        if feature_columns:
            # Perform clustering
            results, X_scaled, scaler = perform_clustering(df, feature_columns, max_clusters, scaling)

            if results:
                # Display results
                results_df = pd.DataFrame(results)
                st.write('Clustering Results')
                st.dataframe(results_df.sort_values('Silhouette Score', ascending=False))

                # Model selection
                selected_method = st.selectbox(
                    'Select clustering method to download', 
                    options=[r['Method'] for r in results]
                )

                # Separate download button outside of form
                if st.button('Download Selected Model'):
                    # Prepare model based on selected method
                    if selected_method == 'K-Means':
                        model = KMeans(n_clusters=max_clusters, random_state=42)
                    elif selected_method == 'Hierarchical Clustering':
                        model = AgglomerativeClustering(n_clusters=max_clusters)
                    elif selected_method == 'Gaussian Mixture':
                        model = GaussianMixture(n_components=max_clusters, random_state=42)
                    elif selected_method == 'DBSCAN':
                        model = DBSCAN(eps=0.5, min_samples=5)

                    # Fit model
                    model.fit(X_scaled)

                    # Prepare files for download
                    temp = BytesIO()
                    with ZipFile(temp, "x") as model_zip:
                        # Save scaler
                        joblib.dump(scaler, 'scaler.pkl')
                        model_zip.write('scaler.pkl')

                        # Save model
                        joblib.dump(model, 'clustering_model.pkl')
                        model_zip.write('clustering_model.pkl')

                        # Save preprocessed data
                        preprocessed_df = pd.DataFrame(X_scaled, columns=feature_columns)
                        preprocessed_df.to_csv('preprocessed_data.csv', index=False)
                        model_zip.write('preprocessed_data.csv')

                    # Download button
                    st.download_button(
                        label=f'Download {selected_method} Clustering Model',
                        data=temp.getvalue(),
                        file_name=f'{selected_method.replace(" ", "_")}_clustering.zip',
                        mime='application/zip'
                    )
            else:
                st.error('No valid clustering results could be generated.')

if __name__ == "__main__":
    main()