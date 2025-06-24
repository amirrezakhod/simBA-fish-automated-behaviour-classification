



## Customizing Feature Extraction (Simba)

The Simba feature extraction script computes a default set of kinematic and behavioral features. However, you can selectively include or exclude specific features by editing a configuration list for efficient customization tailored to your specific research needs or the behaviors of interest, without requiring in-depth programming knowledge.

---

### 📌 How to Customize Features

**Step 1:**  
Find the following list in the `lionfish_feature_extraction.py` script:

```python
AVAILABLE_FEATURES = [
    "X_relative_to_Y_movement",
    "movement",
    "X_relative_to_Y_movement_rolling_windows",
    "velocity",
    "acceleration",
    "rotation",
    "N_degree_direction_switches",
    "bouts_in_same_direction",
    "45_degree_direction_switches",
    "hot_end_encode_compass",
    "directional_switches_in_rolling_windows",
    "angular_dispersion",
    "distances_between_body_part",
    "convex_hulls",
    "pose_confidence_probabilities",
    "distribution_tests",
    "rhythmic_patterns",
    "turning_metrics",
    "energy_metrics",
    "complexity_metrics",
    "path_metrics",
    "body_curvature",
    "swimming_bouts",
    "tail_beat_features",
    "inter_body_coordination",
    "jerk_and_angular_acceleration",
    "frequency_domain_features",
    "lateral_symmetry",
    "fractal_dimension",
    "bout_transition_stats",
    "spatial_occupancy"
]
```


### **Step 2:**
Customize the extraction by editing the following parameters:

Include specific features (included_features):
To extract only selected features, use:

```python
included_features = ["velocity", "acceleration", "body_curvature"]
only velocity, acceleration and body_curvature are calculated.
```
To extract all features except certain ones, use:

```python
excluded_features = ["convex_hulls", "spatial_occupancy"]

convex_hulls and spatial_occupancy are excluded from the features calculated.
```

If neither included_features nor excluded_features is specified, the extractor defaults to computing all available features.



**💡 Recommended Workflow**

Identify features relevant to your specific behaviour from the Feature Usage Guide

Specify them using included_features or exclude unwanted features using excluded_features.
