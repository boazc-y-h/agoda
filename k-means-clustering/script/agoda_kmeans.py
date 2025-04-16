# -*- coding: utf-8 -*-
"""Pyspark

This script runs a KMeans clustering algorithm on the Agoda dataset.
"""

# Set environment variables
import os
# import findspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
import pandas as pd 
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.evaluation import ClusteringEvaluator 
import seaborn as sns
import math
from mpl_toolkits.mplot3d import Axes3D


# Start SparkSession

# findspark.init()
spark = SparkSession.builder.appName("AgodaProject").getOrCreate()
print("Spark Session Created Successfully!")

# Confirm Spark version
print(spark.version)  

# Data Loading

# Define the column names
columns = [
    "code",
    "gender",
    "age",
    "company",
    "avg_flight_price",
    "total_mileage",
    "total_flight_price",
    "total_flights",
    "avg_flight_distance",
    "avg_flight_time",
    "total_days_hotel",
    "total_hotel_price",
    "avg_hotel_price_daily",
    "total_hotels",
    "total_price"
]

# Loading the data with tab separator
input_path = "file:///home/boazcyh/agodaproject/agoda.csv"
dataset = spark.read.csv(input_path, header=False, inferSchema=True, sep=',')

# Assign column names
dataset = dataset.toDF(*columns)

# Show the first 20 rows of the dataset
dataset.show(20)

# Data Preprocessing

# Count occurrences of each unique value in gender
dataset.groupBy("gender").count().show()

# Count occurrences of each unique value in company
dataset.groupBy("company").count().show()

# Index the gender column
indexer = StringIndexer(inputCol="gender", outputCol="genderIndex")
dataset = indexer.fit(dataset).transform(dataset)

# One-hot encode the indexed column
encoder = OneHotEncoder(inputCol="genderIndex", outputCol="genderVec")
dataset = encoder.fit(dataset).transform(dataset)

# Drop the original 'gender' column and index column
dataset = dataset.drop("gender", "genderIndex")

# Show the result
dataset.select("genderVec").show()

# Drop company, code, and gender columns
dataset = dataset.drop("company", "code", "genderVec")

# Convert PySpark DataFrame to Pandas DataFrame
dataset_pd = dataset.toPandas()

# Save the pandas DataFrame to a CSV file
dataset_pd.to_csv("dataset_preprocessed.csv", index=False)

#Print schemas
dataset.printSchema()

# Data Transformation - Feature Engineering

vec_assembler = VectorAssembler(inputCols = dataset.columns,
								outputCol='features')

final_data = vec_assembler.transform(dataset)

final_data.select('features').show(5)

final_data.printSchema()

scaler = StandardScaler(inputCol="features",
						outputCol="scaledFeatures",
						withStd=True,
						withMean=False)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(final_data)

# Normalize each feature to have unit standard deviation.
final_data = scalerModel.transform(final_data)

# Save transformed as Parquet (recommended for complex data)
final_data.write.mode("overwrite").parquet("final_data_transformed.parquet")

final_data.select('scaledFeatures').show(5)

# Model initialization and Selection of cluster number

# Elbow Method for Optimal K

# Calculate cost and plot
cost = np.zeros(10)

for k in range(2,10):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol('features')
    model = kmeans.fit(final_data)
    cost[k] = model.summary.trainingCost

# Plot the cost
df_cost = pd.DataFrame(cost[2:])
df_cost.columns = ["cost"]
new_col = [2,3,4,5,6,7,8, 9]
df_cost.insert(0, 'cluster', new_col)

# Saving the values of WCSS
df_cost.to_csv("elbow_method_cost.csv", index=False)

plt.plot(range(2, 10), df_cost)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method')

# Saving the elbow method plot
plt.savefig('elbow_method.png')

# Clear the current figure
plt.clf()

# Silhouette Score

silhouette_scores = []  # List to store silhouette scores and k values

evaluator = ClusteringEvaluator(predictionCol='prediction',
                                featuresCol='scaledFeatures',
                                metricName='silhouette',
                                distanceMeasure='squaredEuclidean')

# Loop to calculate silhouette scores for different k values
for i in range(2, 10):
    kmeans = KMeans(featuresCol='scaledFeatures', k=i, seed=123)
    model = kmeans.fit(final_data)
    predictions = model.transform(final_data)
    score = evaluator.evaluate(predictions)

    silhouette_scores.append({'k': i, 'silhouette_score': score})  # Store k and score in a dict

    print('Silhouette Score for k =', i, 'is', score)

# Create a pandas DataFrame from the silhouette_scores list
silhouette_df = pd.DataFrame(silhouette_scores)

# Save the pandas DataFrame to a CSV file
silhouette_df.to_csv("silhouette_scores.csv", index=False)

silhouette_df.head()

# Visualizing the silhouette scores in a plot
plt.plot(silhouette_df['k'], silhouette_df['silhouette_score'])
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')

# Saving the silhouette score plot
plt.savefig('silhouette_score.png')

# Clear the current figure
plt.clf()

# K-means Clustering Algorithm for k=4

# Trains a k-means model.
kmeans = KMeans(featuresCol='scaledFeatures',k=4, seed=123)
model = kmeans.fit(final_data)
predictions = model.transform(final_data)

# Printing cluster centers
centers = model.clusterCenters()

# Convert to DataFrame
feature_columns = dataset.columns
centers_df = pd.DataFrame(centers, columns=feature_columns)

# Create a column for cluster numbers
centers_df.insert(0, "cluster", range(len(centers)))

# Display the DataFrame
print("Cluster Centers: ")
print(centers_df)

centers_df.to_csv("centers_df.csv", index=False)

# Plotting the location of cluster centers
no_of_plots = len(centers_df.columns)  # Ensure the correct number of plots
no_of_cols = 3
no_of_rows = math.ceil(no_of_plots / no_of_cols)

fig, axes = plt.subplots(no_of_rows, no_of_cols, figsize=(18, 20))
fig.suptitle('Cluster Centers', y=1.0)

# Flatten axes to handle both 1D and 2D cases
if no_of_rows * no_of_cols > 1:
    axes = axes.flatten()

plot_position = 0
for var in dataset.columns:  # Ensure you're iterating over the right columns
    if plot_position < len(axes):
        sns.barplot(
            x='cluster', y=var, data=centers_df,
            ax=axes[plot_position], hue='cluster', palette='viridis',
            legend=False if plot_position > 0 else True  # Suppress legend for all but the first plot
        )
        # Add legend only for the first plot
        if plot_position == 0:
            handles, labels = axes[plot_position].get_legend_handles_labels()
            axes[plot_position].legend_.remove()  # Remove legend from the first plot
    plot_position += 1

# Turn off unused subplots
for i in range(plot_position, len(axes)):
    fig.delaxes(axes[i])

# Add a single legend at the figure level
fig.legend(handles, labels, loc='upper left', ncol=1, title="Cluster")

plt.tight_layout()
plt.savefig("cluster_centers.png")

# Clear the current figure
plt.clf()

# Display the number of data points in each cluster
predictions.groupBy("prediction").count().show()

# Convert the 'scaledFeatures' vector into separate columns for each feature
predictions_pd = predictions.select("scaledFeatures", "prediction").toPandas()

# Split the 'scaledFeatures' column into separate feature columns
predictions_pd[dataset.columns] = pd.DataFrame(
    predictions_pd['scaledFeatures'].tolist(), index=predictions_pd.index)

# Saving predictions with scaled input values
predictions_pd.to_csv("scaled_predictions.csv", index=False)

predictions_act = predictions.select("features", "prediction").toPandas()

predictions_act[dataset.columns] = pd.DataFrame(
    predictions_act['features'].tolist(), index=predictions_act.index)

predictions_act = predictions_act.loc[:, predictions_act.columns != 'features']

# Saving predictions with actual input values
predictions_act.to_csv("actual_predictions.csv", index=False)

avg_df = predictions_act.groupby(['prediction'], as_index=False).mean()

avg_df.to_csv("avg_df.csv", index=False)
print(avg_df)

# Assuming avg_df is the DataFrame containing cluster averages
no_of_plots = len(avg_df.columns)  # Ensure the correct number of plots
no_of_cols = 3
no_of_rows = math.ceil(no_of_plots / no_of_cols)

fig, axes = plt.subplots(no_of_rows, no_of_cols, figsize=(18, 20))
fig.suptitle('Average Values per Cluster', y=1.0)

# Flatten axes to handle both 1D and 2D cases
if no_of_rows * no_of_cols > 1:
    axes = axes.flatten()

plot_position = 0
for var in dataset.columns:  # Ensure you're iterating over the right columns
    if plot_position < len(axes):
        sns.barplot(
            x='prediction', y=var, data=avg_df,
            ax=axes[plot_position], hue='prediction', palette='viridis',
            legend=False if plot_position > 0 else True  # Suppress legend for all but the first plot
        )
        # Add legend only for the first plot
        if plot_position == 0:
            handles, labels = axes[plot_position].get_legend_handles_labels()
            axes[plot_position].legend_.remove()  # Remove legend from the first plot
    plot_position += 1

# Turn off unused subplots
for i in range(plot_position, len(axes)):
    fig.delaxes(axes[i])

# Add a single legend at the figure level
fig.legend(handles, labels, loc='upper left', ncol=1, title="Cluster")

plt.tight_layout()
plt.savefig("avg_values_per_cluster.png")

# Clear the current figure
plt.clf()



# Create the pair plot
pair_plot = sns.pairplot(predictions_pd, hue='prediction', palette='viridis')

# Asssigning title
pair_plot.fig.suptitle('Pair Plots for k=4', y=0.98)

# Add a single legend
pair_plot._legend.set_title("Cluster")
pair_plot.fig.subplots_adjust(top=0.95)  # Adjust the space for better visualization

# Saving plot
plt.savefig("pairplot.png")

# Clear the current figure
plt.clf()


# 3D scatter plot for three features

# Corrected 3D scatter plot for three features
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(predictions_pd['avg_flight_distance'], predictions_pd['total_flights'], predictions_pd['avg_hotel_price_daily'],
           c=predictions_pd['prediction'], cmap='viridis')

# Corrected labels for the axes
ax.set_xlabel('avg_flight_distance')
ax.set_ylabel('total_flights')
ax.set_zlabel('avg_hotel_price_daily')

# Title
plt.title('3D KMeans Clustering (avg_flight_distance, total_flights, avg_hotel_price_daily)')

# Add color bar (legend)
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster')

# Save the plot as an image
plt.savefig("3D_KMeans_Clustering.png")

# Clear the current figure
plt.clf()


# Create a scatter plot for two features
plt.figure(figsize=(8, 6))
plt.scatter(predictions_pd['total_price'], predictions_pd['avg_flight_distance'], c=predictions_pd['prediction'], cmap='viridis', s=50)
plt.title('KMeans Clustering Results (total_price vs avg_flight_distance)')
plt.xlabel('total_price')
plt.ylabel('avg_flight_distance')
plt.colorbar(label='Cluster')

# Save the plot as an image
plt.savefig("2D_KMeans_Clustering.png")

# Clear the current figure
plt.clf()
