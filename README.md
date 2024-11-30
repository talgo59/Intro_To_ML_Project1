## Intro_To_ML_Project1 - Project 1
---
# Knesset Elections Data Analysis and Dimensionality Reduction

## Project Overview
This project offers a robust data analysis pipeline for exploring Knesset election data using dimensionality reduction techniques, specifically Principal Component Analysis (PCA). It features a comprehensive interface to analyze voting patterns across cities and parties, providing interactive and visual insights.

### A live demo of the interactive interface is available at: [Live Demo](https://ms---with-app-mttjukj93c7mr7uff2q7vn.streamlit.app/).

## Prerequisites
- Python 3.9+
- Required libraries:
  - numpy
  - pandas
  - matplotlib
  - plotly
  - streamlit

## Installation
1.OPT-1:
* Download and unzip EX1.zip
* pip install requierments.txt
2. OPT-2:
  * Clone the repository:
   ```bash
   git clone https://github.com/talgo59/Intro_To_ML_Project1.git
   cd Intro_To_ML_Project1
   ```
* Install required dependencies:
   ```bash
   pip install requierments.txt
   ```

## Project structure and functions overview
- `ex1_functions.py`: Core Python script with data processing functions -
  * uploaded_file - takes xlsx file as an input and returns pandas data frame
  * group_and_aggregate_data - takes data frame as an imput, a column to group by and aggrigation function. returns aggregated data frame by chosen column.
  * remove_sparse_columns - takes data frame and threshold(int) as an input. returns a data frame containing only columns where sum reached threshold.
  * dimensionality_reduction - takes dataframe, number of desired dimentions to reduce to, and meta columns(sependent variables). preforms PCA and returns reduced data frame to the desired number of dimentions.
- `demonstrations.ipynb`: Examples demonstrating the functions in 'ex1_functions.py'
- `VISUALIZATION.ipynb`: Additional notebook for PCA visualizations
- `interface.py`: Streamlit app script offering configurable options for PCA visualization. also available at: [Live Demo](https://ms---with-app-mttjukj93c7mr7uff2q7vn.streamlit.app/).


## Usage Examples
### Running the Streamlit App
```bash
streamlit run interface.py
```
- Upload your dataset.
- Configure grouping, aggregation, and PCA settings via the sidebar.
- Visualize PCA results in 1D, 2D, or 3D.

### Notebook Demonstrations
The Jupyter Notebooks (`notebook.ipynb` and `VISUALIZATION.ipynb`) illustrate:
- Data processing workflows.
- Visualizing PCA components with matplotlib and plotly.

### Example Workflow in Python
```python
from ex1_functions import load_data, group_and_aggregate_data, remove_sparse_columns, dimensionality_reduction

# Load data
df = load_data('election_data.xlsx')

# Aggregate data by city
aggregated_df = group_and_aggregate_data(df, 'city', 'sum')

# Remove sparse columns
filtered_df = remove_sparse_columns(aggregated_df, threshold=1000)

# Reduce dimensionality
reduced_df = dimensionality_reduction(filtered_df, num_components=2, meta_columns=['city'])
```

## Notes
- PCA implementation is custom, avoiding high-level machine learning libraries.
- The app ensures a modular and user-friendly interface.
- Interactive elements allow users to customize analyses directly.

## Author
Yoav Eliav, Tal Gorodetzky, Aviv Tzezana, Rachel Harow  , Maayan Peker
