import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Import the functions from the original script
from ex1_functions import load_data, group_and_aggregate_data, remove_sparse_columns, dimensionality_reduction


# st.header('Intro To ML Ex1:PCA Visualization')
st.subheader('PCA Components Visualization')

# # Sidebar for file upload and configuration
st.sidebar.header('Data Processing Options')

# File Upload
uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])

if uploaded_file is not None:
    # Load the data
    try:
        df = load_data(uploaded_file)
        st.sidebar.success('Data Successfully Loaded!')
        
        # Group By Configuration
        # group_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        group_columns = df.columns.tolist()
        group_by_column = st.sidebar.selectbox('Select Column to Group By', group_columns)
        
        # Aggregation Function Selection
        agg_functions = {
            'Mean': 'mean', 
            'Sum': 'sum', 
            'Count': 'count', 
            'Median': 'median', 
            'Max': 'max', 
            'Min': 'min'
        }
        selected_agg_func = st.sidebar.selectbox('Select Aggregation Function', list(agg_functions.keys()))
        
        # drop irrelevant columns
        potential_drop = df.columns.tolist()
        if str(group_by_column) in potential_drop:
            potential_drop.remove(group_by_column)
        drop_columns = st.sidebar.multiselect('Exclude columns:', potential_drop)
        
        # set K for PCA (i can visualize more then 3 dimentios so.. )
        # num_components = st.sidebar.slider('Number of PCA Components', 1, len(df)-len(drop_columns)-1, 2)
        num_components = st.sidebar.slider('Number of PCA Components', 1, 3, 2)

        # choose PCA rows or columns
        status = st.sidebar.radio("Select PCA direction: ", ('columns('+group_by_column+' as dependent variable)', 'rows(table header as dependent variable)'))

        # Processing Button
        if st.sidebar.button('Process Data'):
            String_Cols =   df.select_dtypes(include=['object', 'category']).columns.tolist()
            len_String_Cols = len(String_Cols)
            if group_by_column in String_Cols:
                len_String_Cols = len(String_Cols)-1
            if len_String_Cols==0:
 
                # Group and Aggregate
                grouped_df = group_and_aggregate_data(df, group_by_column, agg_functions[selected_agg_func])
                #exclude if needed:
                if drop_columns is not None:
                    grouped_df= grouped_df.drop(labels=drop_columns, axis=1)

                # # for checks:
                # st.subheader('Grouped and Aggregated Data')
                # st.dataframe(grouped_df)
                
                # Remove Sparse Columns (with a threshold)
                # threshold = st.sidebar.number_input('Sparsity Threshold', min_value=0, value=10)
                # filtered_df = remove_sparse_columns(grouped_df, threshold)
                # st.subheader('Filtered Data')
                # st.dataframe(filtered_df)
                
                # Dimensionality Reduction
                #test for any string columns

                depVal = str(group_by_column)

            
                if (status == 'rows(table header as dependent variable)' ):
                    #transpose
                    grouped_df = grouped_df.set_index(group_by_column).T.reset_index()
                    #fix col names
                    grouped_df.rename(columns={'index': 'header_values'}, inplace=True)
                    #set dependent var value
                    depVal='header_values'


                reduced_df = dimensionality_reduction(grouped_df, num_components, [depVal])
                
                # # for checks:
                # st.subheader('Dimensionality Reduced Data')
                # st.dataframe(reduced_df)
                
                # Visualization of PCA Components
                
                pc_columns = [col for col in reduced_df.columns if col.startswith('PC')]

                 # Ensure no complex numbers
                for col in pc_columns:
                    reduced_df[col] = reduced_df[col].apply(lambda x: x.real if isinstance(x, complex) else x)
                
                if num_components==1:
                    fig = px.bar(
                    reduced_df,
                    x=depVal,
                    y='PC1',
                    color=depVal,
                    title="PCA HISTOGRAM - 1D",
                    labels={"PC1": "Principal Component 1", depVal: "Group"},
                    hover_data={depVal: True, 'PC1': True},
                    template="plotly_dark",
                )
                if num_components==2:
                    fig = px.scatter(reduced_df, 
                    x='PC1', 
                    y='PC2', 
                    color=depVal, 
                    title="PCA SCATTER - 2D", 
                    labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
                    hover_data={depVal: True, 'PC1': True, 'PC2': True},  
                    template="plotly_dark")
                if num_components==3:
                    fig = px.scatter_3d(
                    reduced_df,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    color=depVal,
                    title="PCA SCATTER - 3D",
                    labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2", "PC3": "Principal Component 3"},
                    hover_data={depVal: True, 'PC1': True, 'PC2': True, 'PC3': True},
                    template="plotly_dark",
                )
                            
                st.plotly_chart(fig, use_container_width=True)

            else:
                    st.warning('Data for PCA contains string/object columns - plese exclude them.')

    except Exception as e:
        st.error('error while handeling data:'+str(e))
        print(e)
