# Data Analysis Assignment
# Name: [Your Name]
# Date: May 6, 2025

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set the style for our plots
plt.style.use('seaborn-v0_8-darkgrid')

# Task 1: Load and Explore the Dataset
print("Task 1: Loading and Exploring the Dataset")
print("-" * 50)

# We'll use the Iris dataset for this assignment
try:
    # Load the iris dataset
    iris = load_iris()
    
    # Create a DataFrame from the dataset
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    
    # Add the target (species) column
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    # Display the first few rows
    print("First 5 rows of the dataset:")
    print(df.head())
    
    # Explore the structure of the dataset
    print("\nDataset Information:")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    print("\nData Types:")
    print(df.dtypes)
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Since the Iris dataset doesn't have missing values, we'll simulate some to show cleaning
    print("\nSimulating some missing values for demonstration...")
    df_with_missing = df.copy()
    
    # Randomly insert some NaN values (about 5% of the data)
    np.random.seed(42)  # For reproducibility
    mask = np.random.random(df.shape) < 0.05
    df_with_missing[mask] = np.nan
    
    print("Missing values after simulation:")
    print(df_with_missing.isnull().sum())
    
    # Cleaning the dataset by filling missing values with the mean of each column
    print("\nCleaning the dataset...")
    # For numerical columns, fill with mean
    for col in df_with_missing.select_dtypes(include=['float64']).columns:
        df_with_missing[col].fillna(df_with_missing[col].mean(), inplace=True)
    
    # For categorical columns, fill with mode
    for col in df_with_missing.select_dtypes(include=['category']).columns:
        df_with_missing[col].fillna(df_with_missing[col].mode()[0], inplace=True)
    
    print("Missing values after cleaning:")
    print(df_with_missing.isnull().sum())
    
    # We'll continue with the original clean dataset for analysis
    df = df.copy()
    
except Exception as e:
    print(f"Error loading or exploring the dataset: {e}")

# Task 2: Basic Data Analysis
print("\nTask 2: Basic Data Analysis")
print("-" * 50)

try:
    # Compute basic statistics for numerical columns
    print("Basic Statistics:")
    print(df.describe())
    
    # Grouping by species and computing means
    print("\nMean of each feature by species:")
    species_means = df.groupby('species').mean()
    print(species_means)
    
    # Finding interesting patterns
    print("\nInteresting Patterns:")
    # Calculate the range of each feature for each species
    ranges = df.groupby('species').agg(lambda x: x.max() - x.min())
    print("Range of features by species:")
    print(ranges)
    
    # Correlation matrix
    print("\nCorrelation matrix:")
    correlation_matrix = df.select_dtypes(include=['float64']).corr()
    print(correlation_matrix)
    
    # Key findings
    print("\nKey Findings:")
    print("1. Setosa species has the smallest petal length and width.")
    print("2. Virginica species has the largest petal length and width.")
    print("3. Petal length and petal width are highly correlated.")
    print("4. Sepal length and width have a relatively low correlation.")
    
except Exception as e:
    print(f"Error in data analysis: {e}")

# Task 3: Data Visualization
print("\nTask 3: Data Visualization")
print("-" * 50)

try:
    # Set up a figure with subplots for our visualizations
    plt.figure(figsize=(16, 14))
    
    # 1. Line chart showing trends (simulating time-series data)
    plt.subplot(2, 2, 1)
    # Create a simple time series by sorting the data
    df_sorted = df.sort_values('sepal length (cm)')
    plt.plot(df_sorted['sepal length (cm)'], label='Sepal Length')
    plt.plot(df_sorted['petal length (cm)'], label='Petal Length')
    plt.title('Sepal and Petal Length Trends (Sorted by Sepal Length)')
    plt.xlabel('Sample Index (sorted)')
    plt.ylabel('Length (cm)')
    plt.legend()
    plt.grid(True)
    
    # 2. Bar chart showing comparison of numerical values across categories
    plt.subplot(2, 2, 2)
    species_means.plot(kind='bar', ax=plt.gca())
    plt.title('Average Measurements by Species')
    plt.xlabel('Species')
    plt.ylabel('Measurement (cm)')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(True, axis='y')
    
    # 3. Histogram of a numerical column
    plt.subplot(2, 2, 3)
    for species in iris.target_names:
        plt.hist(df[df['species'] == species]['petal length (cm)'], 
                 alpha=0.7, 
                 label=species)
    plt.title('Distribution of Petal Length by Species')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # 4. Scatter plot to visualize relationship between two numerical columns
    plt.subplot(2, 2, 4)
    for i, species in enumerate(iris.target_names):
        plt.scatter(df[df['species'] == species]['sepal length (cm)'], 
                    df[df['species'] == species]['petal length (cm)'], 
                    label=species,
                    alpha=0.7)
    plt.title('Sepal Length vs Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('iris_data_analysis.png')
    
    # Additional visualization: Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    
    # Additional visualization: Pair plot for all features
    sns.pairplot(df, hue='species', height=2.5)
    plt.suptitle('Pair Plot of Iris Features', y=1.02)
    plt.savefig('iris_pairplot.png')
    
    print("Visualizations created successfully!")
    
except Exception as e:
    print(f"Error in data visualization: {e}")

# Final Observations and Conclusions
print("\nFinal Observations and Conclusions")
print("-" * 50)

print("""
Key Observations:
1. The Iris dataset contains 150 samples with 4 features (sepal length, sepal width, petal length, petal width) and 3 species (setosa, versicolor, virginica).
2. There is a clear separation between the setosa species and the other two species, particularly in petal characteristics.
3. Versicolor and virginica species have some overlap but are still largely distinguishable.
4. Petal length and petal width are highly correlated (correlation coefficient of 0.96), making them strong indicators of species classification.
5. Sepal width has a negative correlation with other features, suggesting that as petal size increases, sepal width tends to decrease.

Conclusions:
1. The Iris dataset provides a good example for classification tasks due to the clear separation between species.
2. The petal measurements are more distinctive for species classification than sepal measurements.
3. Data visualization plays a crucial role in understanding the relationships between features and identifying patterns in the data.
4. Preliminary analysis suggests that machine learning models would likely perform well in classifying the species based on these features.
""")

print("\nAssignment Completed Successfully!")