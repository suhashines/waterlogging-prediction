import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

# Set the style for the plots
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Define custom colors for each model
colors = {
    'Random Forest': '#3498db',
    'AdaBoost': '#e74c3c',
    'GBDT': '#2ecc71'
}

# Create dummy data for model performance metrics
np.random.seed(42)  # For reproducibility

# Stations
stations = ['Station A', 'Station B', 'Station C']
models = ['Random Forest', 'AdaBoost', 'GBDT']

# Create a DataFrame with model performance metrics
data = {
    'Station': [],
    'Model': [],
    'RMSE': [],
    'R^2': [],
    'Training Time (s)': [],
    'Prediction Time (ms)': []
}

# Fill with realistic data
for station in stations:
    # Base values that vary by station
    if station == 'Station A':
        base_rmse = 0.15
        base_r2 = 0.85
        base_train = 5.0
    elif station == 'Station B':
        base_rmse = 0.12
        base_r2 = 0.88
        base_train = 4.5
    else:  # Station C
        base_rmse = 0.18
        base_r2 = 0.82
        base_train = 5.5
    
    for model in models:
        # Add variation by model
        if model == 'Random Forest':
            rmse = base_rmse * np.random.uniform(0.9, 1.1)
            r2 = base_r2 * np.random.uniform(0.95, 1.02)
            train_time = base_train * np.random.uniform(0.8, 1.2)
            pred_time = 2.5 * np.random.uniform(0.9, 1.1)
        elif model == 'AdaBoost':
            rmse = base_rmse * np.random.uniform(1.1, 1.3)
            r2 = base_r2 * np.random.uniform(0.9, 0.98)
            train_time = base_train * np.random.uniform(0.6, 0.9)
            pred_time = 1.8 * np.random.uniform(0.9, 1.1)
        else:  # GBDT
            rmse = base_rmse * np.random.uniform(0.8, 0.95)
            r2 = base_r2 * np.random.uniform(1.0, 1.05)
            train_time = base_train * np.random.uniform(1.1, 1.4)
            pred_time = 2.0 * np.random.uniform(0.9, 1.1)
        
        # Ensure R² doesn't exceed 1.0
        r2 = min(r2, 0.99)
        
        # Add to data dictionary
        data['Station'].append(station)
        data['Model'].append(model)
        data['RMSE'].append(rmse)
        data['R^2'].append(r2)
        data['Training Time (s)'].append(train_time)
        data['Prediction Time (ms)'].append(pred_time)

# Convert to DataFrame
df = pd.DataFrame(data)

# Calculate overall average performance for each model
# Fix: Only compute mean for numeric columns
model_avg = df.groupby('Model')[['RMSE', 'R^2', 'Training Time (s)', 'Prediction Time (ms)']].mean()

# Print the average performance
print("Average Performance Metrics by Model:")
print(model_avg)

# 1. RMSE Comparison (Bar Chart)
plt.figure(figsize=(14, 10))

# Create a grid for the plots
gs = GridSpec(2, 2, width_ratios=[2, 1])

ax1 = plt.subplot(gs[0, 0])
ax1.set_title('RMSE by Station and Model (lower is better)', fontsize=14, fontweight='bold')

# Create grouped bar chart
sns.barplot(x='Station', y='RMSE', hue='Model', data=df, palette=colors, ax=ax1)
ax1.set_ylabel('Root Mean Squared Error', fontsize=12)
ax1.set_xlabel('Monitoring Station', fontsize=12)
ax1.legend(title='Model')

# Add value labels on bars
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.3f')

# 2. R² Comparison (Bar Chart)
ax2 = plt.subplot(gs[0, 1])
ax2.set_title('R² Score by Model (higher is better)', fontsize=14, fontweight='bold')

# Fixed barplot with hue parameter
sns.barplot(x='R^2', y=model_avg.index, hue=model_avg.index, data=model_avg.reset_index(), 
            palette=colors, ax=ax2, legend=False)
ax2.set_xlabel('Average R² Score', fontsize=12)
ax2.set_ylabel('Model', fontsize=12)

# Add value labels on bars
for container in ax2.containers:
    ax2.bar_label(container, fmt='%.3f')

# 3. Combined Performance Matrix (Heatmap)
ax3 = plt.subplot(gs[1, :])
ax3.set_title('Performance Metrics by Station and Model', fontsize=14, fontweight='bold')

# Pivot the data for the heatmap - normalize metrics to 0-1 range for comparison
pivot_rmse = df.pivot_table(values='RMSE', index='Model', columns='Station')
pivot_r2 = df.pivot_table(values='R^2', index='Model', columns='Station')

# Normalize RMSE (lower is better, so invert)
rmse_min = pivot_rmse.min().min()
rmse_max = pivot_rmse.max().max()
norm_rmse = 1 - (pivot_rmse - rmse_min) / (rmse_max - rmse_min)

# Normalize R² (higher is better)
r2_min = pivot_r2.min().min()
r2_max = pivot_r2.max().max()
norm_r2 = (pivot_r2 - r2_min) / (r2_max - r2_min)

# Calculate combined score (average of normalized metrics)
combined_score = (norm_rmse + norm_r2) / 2

# Create heatmap
sns.heatmap(combined_score, annot=True, cmap='YlGnBu', fmt='.3f', ax=ax3, cbar_kws={'label': 'Combined Performance Score (higher is better)'})
ax3.set_ylabel('Model', fontsize=12)
ax3.set_xlabel('Station', fontsize=12)

plt.tight_layout()
plt.savefig('model_comparison_part1.png', dpi=300, bbox_inches='tight')

# Create a second figure with more visualizations
plt.figure(figsize=(14, 12))

gs2 = GridSpec(2, 2)

# 4. Model Comparison by Station (Line plot)
ax4 = plt.subplot(gs2[0, 0])
ax4.set_title('RMSE by Station for Each Model', fontsize=14, fontweight='bold')

# Create connected line plot
for model in models:
    model_data = df[df['Model'] == model]
    ax4.plot(model_data['Station'], model_data['RMSE'], 'o-', label=model, color=colors[model], linewidth=2, markersize=8)

ax4.set_ylabel('RMSE (lower is better)', fontsize=12)
ax4.set_xlabel('Station', fontsize=12)
ax4.legend(title='Model')

# 5. Training Time vs. R² Scatter Plot
ax5 = plt.subplot(gs2[0, 1])
ax5.set_title('Training Time vs. R² Score', fontsize=14, fontweight='bold')

for model in models:
    model_data = df[df['Model'] == model]
    ax5.scatter(model_data['Training Time (s)'], model_data['R^2'], label=model, color=colors[model], s=100, alpha=0.7)

ax5.set_ylabel('R² Score (higher is better)', fontsize=12)
ax5.set_xlabel('Training Time (seconds)', fontsize=12)
ax5.legend(title='Model')

# 6. 3D Performance Visualization
ax6 = plt.subplot(gs2[1, :], projection='3d')
ax6.set_title('3D Performance Visualization (RMSE, R², Training Time)', fontsize=14, fontweight='bold')

for model in models:
    model_data = df[df['Model'] == model]
    ax6.scatter(model_data['RMSE'], model_data['R^2'], model_data['Training Time (s)'], 
                label=model, color=colors[model], s=100, alpha=0.7)
    
    # Add connecting lines for the same model across stations
    for station in stations:
        station_data = model_data[model_data['Station'] == station]
        ax6.text(station_data['RMSE'].values[0], station_data['R^2'].values[0], 
                 station_data['Training Time (s)'].values[0], station, fontsize=8)

ax6.set_xlabel('RMSE (lower is better)', fontsize=10)
ax6.set_ylabel('R² Score (higher is better)', fontsize=10)
ax6.set_zlabel('Training Time (seconds)', fontsize=10)
ax6.legend(title='Model')

# Add a better angle for 3D visualization
ax6.view_init(elev=30, azim=45)

plt.tight_layout()
plt.savefig('model_comparison_part2.png', dpi=300, bbox_inches='tight')

# Create a third figure for model selection guidance
plt.figure(figsize=(12, 10))

# 7. Performance metrics radar chart
ax7 = plt.subplot(111, polar=True)
ax7.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)

# Define the metrics and angles
metrics = ['RMSE (inverted)', 'R²', 'Training Speed', 'Prediction Speed']
num_metrics = len(metrics)
angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]  # Close the loop

# Prepare the data
# Normalize all metrics to 0-1 range where 1 is always better
model_metrics = {}
for model in models:
    # Fix: Select only numeric columns when computing means
    model_data = df[df['Model'] == model][['RMSE', 'R^2', 'Training Time (s)', 'Prediction Time (ms)']].mean()
    
    # RMSE - invert and normalize
    rmse_norm = 1 - (model_data['RMSE'] - df['RMSE'].min()) / (df['RMSE'].max() - df['RMSE'].min())
    
    # R² - normalize
    r2_norm = (model_data['R^2'] - df['R^2'].min()) / (df['R^2'].max() - df['R^2'].min())
    
    # Training speed - invert and normalize (lower time is better)
    train_norm = 1 - (model_data['Training Time (s)'] - df['Training Time (s)'].min()) / (df['Training Time (s)'].max() - df['Training Time (s)'].min())
    
    # Prediction speed - invert and normalize (lower time is better)
    pred_norm = 1 - (model_data['Prediction Time (ms)'] - df['Prediction Time (ms)'].min()) / (df['Prediction Time (ms)'].max() - df['Prediction Time (ms)'].min())
    
    # Store normalized values
    model_metrics[model] = [rmse_norm, r2_norm, train_norm, pred_norm]
    
    # Close the loop for plotting
    model_metrics[model] += model_metrics[model][:1]

# Plot each model
for model, values in model_metrics.items():
    ax7.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[model])
    ax7.fill(angles, values, alpha=0.1, color=colors[model])

# Set the angle labels
ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(metrics)

# Set the y-axis limits
ax7.set_ylim(0, 1)
ax7.grid(True)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.2, 0.1))

# Add a summary text box
summary_text = """
Model Selection Guidance:

1. Random Forest: Balanced performance with good accuracy and reasonable training time.
   Best for: General use cases where model interpretability is important.

2. AdaBoost: Fastest training but lower accuracy.
   Best for: Scenarios with limited computational resources or time constraints.

3. GBDT: Highest accuracy but longer training time.
   Best for: Applications where prediction accuracy is paramount.
"""

plt.figtext(0.5, 0.02, summary_text, ha="center", fontsize=12, 
            bbox={"facecolor":"white", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.savefig('model_selection_guidance.png', dpi=300, bbox_inches='tight')

# Final conclusion and recommendations
print("\nModel Selection Recommendations:")
print("--------------------------------")
print("1. For high accuracy requirements: GBDT")
print("   - Best overall accuracy (lowest RMSE, highest R²)")
print("   - Consider the longer training time trade-off")
print()
print("2. For balanced performance: Random Forest")
print("   - Good accuracy with reasonable training time")
print("   - More interpretable than other models")
print() 
print("3. For speed-critical applications: AdaBoost")
print("   - Fastest training time")
print("   - Acceptable accuracy for most scenarios")