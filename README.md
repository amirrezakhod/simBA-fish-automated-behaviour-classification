

# 🐟 Lionfish Feature Extraction Guide
<img width="960" alt="image" src="https://github.com/user-attachments/assets/e4abd305-e54e-4c5c-8bc6-e44844e09846" />

This feature extraction code is specifically designed to support the classification of behaviors in mid-bodied fishes, such as lionfish, using the SimBA (Simple Behavioral Analysis) platform.
---

## 📖 Available Guides

- ✅ [Feature Inclusion Guide](docs/feature_inclusion_guide.md)  
  Learn when to include or exclude specific features in your pipeline.

- 🎯 [Feature Usage Recommendations](docs/feature_usage_guide.md)  
  Find out which features are best for detecting different behaviors.

- ▶️ [Demo Execution Instructions](docs/run_instructions.md)  
  Run a demo example with DeepLabCut CSV and video.
  


**Simba Feature selection guide**:

Each feature described below helps to identify specific fish behaviors based on movement, posture, orientation, or activity patterns. Each description includes a simple definition, a scenario for optimal usage, and an example fish behavior:

**1. X_relative_to_Y_movement**

What it measures: Difference between horizontal (X) and vertical (Y) movement across all tracked body points per frame.

Use when: The behavior involves directional bias (predominantly side-to-side or up-and-down movements).

Example: Detecting lateral head swings versus vertical nodding during aggression displays.

**2. movement**

What it measures: Total frame-to-frame distance each tracked body part moves; also computes summed movement across all body parts as a global activity measure.

Use when: Distinguishing between active and resting states or broadly quantifying activity levels.

Example: Identifying transitions from stationary resting periods to active swimming bouts.

**3. X_relative_to_Y_movement_rolling_windows**

What it measures: Rolling averages (smoothed values over multiple frames) of the X vs. Y relative movements, stabilizing short-term directional biases.

Use when: Short sustained directional movements or posture adjustments characterize behavior.

Example: Recognizing prolonged dorso-ventral head movements typical of hovering behaviors.

**4. velocity**

What it measures: Movement per second averaged over all tracked body parts, providing rolling-window statistics (mean, min, max velocities).

Use when: Speed consistently differentiates behaviors.

Example: Quantifying high-speed burst movements during escape responses.

**5. acceleration**

What it measures: Rate of change in velocity over various rolling windows, indicating rapid speed changes or bursts.

Use when: Rapid initiation or cessation of movement signals specific behaviors.

Example: Detecting sudden accelerations at the onset of a predatory strike.

**6. jerk_and_angular_acceleration**

What it measures: Changes in acceleration (jerk) and changes in angular velocity (angular acceleration), indicating explosive movements and quick directional adjustments.

Use when: Sudden, explosive behaviors are key identifying features.

Example: Capturing rapid darting movements during escape maneuvers.

**7. rotation**

What it measures: Angle (heading) of the fish’s body relative to a fixed axis (calculated between mouth and tail), including rolling mean for stabilization.

Use when: Body orientation significantly distinguishes behaviors.

Example: Maintaining an upstream orientation during rheotaxis.

**8. angular_dispersion**

What it measures: Variability or stability of the fish's heading over time; higher values indicate more consistent headings, while lower values suggest wandering.

Use when: Differentiating directed swimming from exploratory or random swimming.

Example: Distinguishing migration (straight-line) from foraging (variable paths).

**9. N_degree_direction_switches**

What it measures: Counts directional reversals that exceed defined angular thresholds (90°, 180°), indicating abrupt directional changes.

Use when: Erratic zig-zag swimming differentiates behaviors.

Example: Zig-zag pursuit patterns of predatory fish chasing agile prey.

**10. 45_degree_direction_switches**

What it measures: Counts moderate directional shifts around 45°.

Use when: Subtle directional adjustments matter, such as alignment adjustments.

Example: Fish making slight heading corrections to maintain cohesion within schools.

**11. directional_switches_in_rolling_windows**

What it measures: Smoothed counts of directional changes within short temporal windows, indicating short-term directional instability.

Use when: Brief episodes of twitchy or nervous movements characterize behaviors.

Example: Frequent direction changes during territorial defense movements.

**12. turning_metrics**

What it measures: Quantifies sharp turning events (angle changes >30°) and records their frequency, duration, and intensity.

Use when: Specific turning patterns and their frequency characterize behaviors.

Example: Distinguishing cautious exploration (slow turns) from evasive maneuvers (rapid, sharp turns).

**13. hot_end_encode_compass**

What it measures: Encodes fish heading direction into discrete compass sectors (North, East, South, West, etc.).

Use when: Behavior classification depends explicitly on the fish's heading direction relative to environmental features.

Example: Fish consistently orienting into water currents during rheotactic behaviors.

**14. bouts_in_same_direction**

What it measures: Tracks continuous periods of movement without significant direction changes, quantifying directional persistence.

Use when: Behavioral states involve prolonged directional swimming versus frequent reorientations.

Example: Prolonged swimming in a consistent direction during upstream migration.

**15. distances_between_body_part**

What it measures: Pairwise distances between tracked anatomical landmarks, capturing posture changes.

Use when: Specific postures or expansions (e.g., fins or operculum spread) indicate behaviors.

Example: Operculum expansion signaling aggressive interactions or territorial defense.

**16. convex_hulls**

What it measures: Area enclosed by the fish’s tracked body parts, indicating body posture expansion or contraction.

Use when: Body shape changes (expanded vs. contracted postures) characterize behavior.

Example: Detecting fin-flaring threats or territorial displays.

**17. body_curvature**

What it measures: Angles along the fish’s body midline, quantifying bending or curvature.

Use when: Distinctive body curvature differentiates specific maneuvers or postures.

Example: Tight body bends preceding escape behaviors (C-start).

**18. lateral_symmetry**

What it measures: Evaluates lateral (left-right) symmetry or asymmetry of body posture.

Use when: Behavioral significance lies in asymmetric postures.

Example: One-sided fin displays during aggressive confrontations.

**19. rhythmic_patterns**

What it measures: Periodicity and regularity in movement (autocorrelation and frequency analysis).

Use when: Repeated rhythmic movements are diagnostic.

Example: Regular fin beating during stationary hovering.

**20. frequency_domain_features**

What it measures: Analyzes frequencies of body movements (using FFT), identifying dominant vibration or tremor patterns.

Use when: Movement frequencies distinctly characterize behaviors.

Example: Characteristic rapid vibrations during mating rituals or courtship.

**21. tail_beat_features**

What it measures: Tail-beat amplitude, frequency, and periodicity extracted from caudal fin movements.

Use when: Tail kinematics are integral to behavior identification.

Example: High-frequency tail beats during hovering or slow swimming.

**22. swimming_bouts**

What it measures: Identifies and segments continuous swimming episodes, quantifying duration, frequency, and speed.

Use when: Behavioral episodes (active swimming vs. resting) clearly demarcate the behavior.

Example: Burst-and-coast swimming during predation attempts.

**23. bout_transition_stats**

What it measures: Counts transitions between active and inactive behavioral states.

Use when: Sequential patterns or transitions between states matter behaviorally.

Example: Activity patterns indicative of vigilance behaviors in shelter-seeking.

**24. energy_metrics**

What it measures: Estimates total kinetic energy from velocity data, quantifying energetic intensity.

Use when: Energetic expenditure differentiates behaviors.

Example: High-energy burst swimming versus low-energy routine cruising.

**25. complexity_metrics**

What it measures: Measures randomness or complexity (entropy) in movement trajectories.

Use when: Random versus structured patterns of exploration differentiate behaviors.

Example: Random exploratory paths versus structured homing paths.

**26. fractal_dimension**

What it measures: Complexity of movement paths quantified by fractal dimension analysis.

Use when: Distinguishing complex, repetitive searches from simpler, linear movements.

Example: Complex, self-similar search paths versus simple direct movements.

**27. path_metrics**

What it measures: Evaluates total distance traveled, path straightness, and efficiency.

Use when: Path geometry distinctly separates behaviors.

Example: Linear homing behaviors versus circular or looping patrol behaviors.

**28. spatial_occupancy**

What it measures: Fish’s positional consistency (variance in X and Y positions) within a defined region.

Use when: Spatial localization significantly characterizes behaviors.

Example: Territorial fish remaining close to nest areas.
