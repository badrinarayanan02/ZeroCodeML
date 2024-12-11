# #importing library
# import streamlit as st
# import pandas as pd
# import numpy as np
# import csv
# from io import StringIO
# import matplotlib.pyplot as plt 
# import seaborn as sns

# st.cache_data

# data = st.file_uploader('Upload your data here', type=['csv','xlsx', 'xls'])
# if data is not None:
#     #check the type of file uploaded and get the extension of the file
#     filename=data.name
#     x = filename.rsplit('.', 1)

#     #if uploaded file is CSV
#     if x[1] == 'csv':
#         # because there are many posibilites for csv delimiter, using string io then combine it with csv sniffer to get the delimiter format
#         stringio = StringIO(data.getvalue().decode("utf-8"))
#         string_data = stringio.read()
#         sniffer = csv.Sniffer()
#         dialect = sniffer.sniff(string_data)
#         delimiter = dialect.delimiter
#         df = pd.read_csv(data, sep=dialect.delimiter)
#     #if uploaded file is excel file
#     else:
#         df = pd.read_excel(data) #catatan perlu openpyxl sama xlrd (buat xls)

#     #basic info of dataset
#     st.write('uploaded',filename, 'with the shape of', df.shape)
#     st.write('dataset sample')
#     st.write(df.sample(5))

#     #defining summary
#     def dfSummary(data):
#         summary = pd.DataFrame(data.dtypes,columns=['dtypes'])
#         summary = summary.reset_index()
#         summary['Column'] = summary['index']
#         summary = summary[['Column','dtypes']]
#         summary['non-null'] = data.notnull().sum().values
#         summary['Missing'] = data.isnull().sum().values 
#         summary['Missing (%)'] = data.isnull().sum().values * 100 / len(data) 
#         summary['Uniques'] = data.nunique().values  
#         return summary
#     #create summary dataset
#     dfsum = dfSummary(df)
#     # Show Summary
#     st.write("Summary of the dataset")
#     st.dataframe(dfsum)
    
#     st.write('Descriptive statistics')
#     st.dataframe(df.describe())

#     #create histogram for all numerical data
#     if st.checkbox('Numerical data distribution'):
#         fig = plt.figure(figsize = (15,15))
#         ax = fig.gca()
#         df.hist(ax=ax)
#         st.pyplot(fig)

#     #create plot for all categorical data, with unique value thresholds to mean
#     if st.checkbox('Categorical data distribution'):
#         min_value = dfsum['Uniques'].min().astype(int).astype(object) #remove .astype(int).astype(object) if an error message appears
#         max_value = dfsum['Uniques'].max().astype(int).astype(object) #remove .astype(int).astype(object) if an error message appears
#         current = dfsum['Uniques'].mean().round().astype(int).astype(object)
#         uvt = st.slider('Unique Value Threshold', min_value, max_value, current)
#         cat_data = dfsum[(dfsum['dtypes'] == 'object') & (dfsum['Uniques'] <= uvt)] 
#         cat_df = df[cat_data.Column.values]
        
#         fig = plt.figure(figsize = (15,15))
#         for index in range(len(cat_data['Column'])):
#             plt.subplot((len(cat_data['Column'])),4,index+1)
#             sns.countplot(x=cat_df.iloc[:,index], data=cat_df.dropna())
#             plt.xticks(rotation=90)
#         st.pyplot(fig)
        
#     if st.checkbox('Custom plot'):
#         num_data = dfsum[dfsum['dtypes'] != 'object'] 
#         num_df = df[num_data.Column.values]
#         columns_list = num_df.columns.to_list()
#         cat2_data = dfsum[dfsum['dtypes'] == 'object'] 
#         cat2_df = df[cat2_data.Column.values]
#         columns_cat_list = cat2_df.columns.to_list()
#         chart_options = st.selectbox(
#         'Select chart type',
#         ('Boxplot', 'Scatterplot'))
#         if chart_options == 'Boxplot':
#             col1, col2 = st.columns(2)
#             with col2:
#                 x_axis = st.selectbox(
#                                 'Select column for X axis',
#                                 (columns_cat_list))
#             with col1:
#                 y_axis = st.selectbox(
#                                 'Select column for Y axis',
#                                 (columns_list))
                
#             if st.button('Generate Chart'):
#                 fig, ax = plt.subplots()
#                 sns.boxplot(x=df[x_axis], y=df[y_axis])
#                 st.pyplot(fig)
#             else:
#                 st.write('select chart type and column for the axis')
        
#         if chart_options == 'Scatterplot':
#             col1, col2 = st.columns(2)
#             with col2:
#                 x_axis = st.selectbox(
#                                 'Select column for X axis',
#                                 (columns_list))
#             with col1:
#                 y_axis = st.selectbox(
#                                 'Select column for Y axis',
#                                 (columns_list))
                
#             if st.button('Generate Chart'):
#                 charts = plt.figure() 
#                 plt.scatter(df[x_axis], df[y_axis])
#                 plt.xlabel(x_axis)
#                 plt.ylabel(y_axis)
#                 st.pyplot(charts)
#             else:
#                 st.write('select chart type and column for the axis')

#     #correlation plot
#     if st.checkbox('Correlation plot'):
#         fig, ax = plt.subplots()
#         sns.heatmap(df.corr(),  ax=ax, vmin=-1, vmax=1, annot=True)
#         st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import csv

# Streamlit App Configuration
st.set_page_config(
    page_title="NOCODESK - EDA",
    page_icon=":bar_chart:",
    layout="wide"
)

# File Upload Section
st.title("Exploratory Data Analysis (EDA)")
data = st.file_uploader("Upload your data here (CSV, XLSX)", type=["csv", "xlsx", "xls"])

def load_data(data):
    """
    Load the uploaded dataset into a DataFrame.
    Detect and handle CSV or Excel formats with varying delimiters.
    """
    try:
        filename = data.name
        extension = filename.split('.')[-1]

        if extension == "csv":
            stringio = StringIO(data.getvalue().decode("utf-8"))
            string_data = stringio.read()
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(string_data)
            df = pd.read_csv(data, sep=dialect.delimiter)
        else:
            df = pd.read_excel(data)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

if data is not None:
    # Load and display dataset
    df = load_data(data)
    if df is not None:
        st.success(f"Uploaded: {data.name} | Shape: {df.shape}")
        st.write("### Dataset Sample")
        st.dataframe(df.head())

        # Summary Function
        def df_summary(data):
            summary = pd.DataFrame({
                "Column": data.columns,
                "Dtype": data.dtypes,
                "Non-Null Count": data.notnull().sum(),
                "Missing Values": data.isnull().sum(),
                "Missing (%)": (data.isnull().sum() / len(data)) * 100,
                "Unique Values": data.nunique()
            })
            return summary

        st.write("### Summary of the Dataset")
        st.dataframe(df_summary(df))

        # Descriptive Statistics
        st.write("### Descriptive Statistics")
        st.dataframe(df.describe())

        # Numerical Data Distribution
        if st.checkbox("Show Numerical Data Distribution"):
            fig, ax = plt.subplots(figsize=(12, 8))
            df.select_dtypes(include=["float64", "int64"]).hist(ax=ax)
            st.pyplot(fig)

        # Categorical Data Distribution
        if st.checkbox("Show Categorical Data Distribution"):
            unique_value_threshold = st.slider("Select Unique Value Threshold", 2, 20, 10)
            categorical_cols = [col for col in df.select_dtypes(include=["object"]).columns if df[col].nunique() <= unique_value_threshold]

            if categorical_cols:
                fig, axes = plt.subplots(len(categorical_cols), 1, figsize=(10, len(categorical_cols) * 4))
                for i, col in enumerate(categorical_cols):
                    sns.countplot(x=df[col], ax=axes[i] if len(categorical_cols) > 1 else axes)
                    axes[i].set_title(f"Distribution of {col}")
                    axes[i].tick_params(axis='x', rotation=90)
                st.pyplot(fig)
            else:
                st.info("No categorical columns meet the unique value threshold.")

        # Correlation Plot
        if st.checkbox("Show Correlation Plot"):
            numeric_df = df.select_dtypes(include=["float64", "int64"])
            if not numeric_df.empty:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.info("No numeric columns available for correlation analysis.")

        # Custom Plot Section
        if st.checkbox("Generate Custom Plot"):
            plot_type = st.selectbox("Select Plot Type", ["Boxplot", "Scatterplot"])
            columns = df.columns.tolist()
            x_axis = st.selectbox("X-Axis", columns)
            y_axis = st.selectbox("Y-Axis", columns)

            if st.button("Generate Plot"):
                if plot_type == "Boxplot":
                    fig, ax = plt.subplots()
                    sns.boxplot(x=df[x_axis], y=df[y_axis], ax=ax)
                    st.pyplot(fig)
                elif plot_type == "Scatterplot":
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
                    st.pyplot(fig)
                else:
                    st.error("Invalid plot type selected.")

else:
    st.info("Please upload a dataset to start the EDA process.")
