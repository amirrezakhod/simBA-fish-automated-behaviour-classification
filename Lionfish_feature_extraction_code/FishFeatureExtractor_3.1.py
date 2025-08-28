
from __future__ import division

import glob
import math
import os
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import jit, prange
from scipy import stats
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from scipy.stats import zscore, entropy
from statsmodels.stats.diagnostic import lilliefors

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.enums import Paths
from simba.utils.printing import SimbaTimer
from simba.utils.read_write import (check_if_filepath_list_is_empty,
                                    get_fn_ext, read_config_file, read_df,
                                    read_project_path_and_file_type,
                                    read_video_info, write_df)

# ---------------------------------------------------------------------------------
# Body part definitions for lionfish skeleton 
HEAD_BP_NAMES = ["bodypart1", "bodypart2", "bodypart13"]
DORSAL_BP_NAMES = ["bodypart3", "bodypart4", "bodypart5"]
PELVIC_BP_NAMES = ["bodypart12"]
TAIL_EXTRA_BP_NAMES = ["bodypart6", "bodypart7", "bodypart8", "bodypart9", "bodypart10", "bodypart11"]
TAIL_BP_NAMES = ["bodypart15", "bodypart16"]
CENTER_BP_NAMES = ["bodypart14"]
MOUTH = ["bodypart2"]

ANGULAR_DISPERSION_S = [10, 5, 2, 1, 0.5, 0.25]

# Full list of features available in the extractor
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
    "spatial_occupancy",
]

# ---------------------------------------------------------------------------------
# GUI categories for organizing features (names MUST match AVAILABLE_FEATURES)
FEATURE_CATEGORIES = {
    "Basic Movement": {
        "features": ["movement", "velocity", "acceleration"],
        "description": "Core movement metrics including distance, velocity, and acceleration",
    },
    "Directional & Rotational": {
        "features": [
            "rotation",
            "N_degree_direction_switches",
            "45_degree_direction_switches",
            "bouts_in_same_direction",
            "hot_end_encode_compass",
            "directional_switches_in_rolling_windows",
            "angular_dispersion",
        ],
        "description": "Direction changes, compass bearings, and angular movements",
    },
    "Spatial Relationships": {
        "features": [
            "X_relative_to_Y_movement",
            "X_relative_to_Y_movement_rolling_windows",
            "distances_between_body_part",
            "convex_hulls",
            "spatial_occupancy",
        ],
        "description": "Spatial positioning and geometric relationships between body parts",
    },
    "Body Dynamics": {
        "features": [
            "body_curvature",
            "tail_beat_features",
            "inter_body_coordination",
            "lateral_symmetry",
        ],
        "description": "Body shape dynamics and coordination between body segments",
    },
    "Swimming Behavior": {
        "features": ["swimming_bouts", "turning_metrics", "path_metrics", "energy_metrics"],
        "description": "Swimming patterns, bout analysis, and energy expenditure",
    },
    "Advanced Kinematics": {
        "features": [
            "jerk_and_angular_acceleration",
            "frequency_domain_features",
            "rhythmic_patterns",
            "complexity_metrics",
        ],
        "description": "Higher-order derivatives and frequency domain analysis",
    },
    "Statistical & Quality": {
        "features": [
            "pose_confidence_probabilities",
            "distribution_tests",
            "fractal_dimension",
            "bout_transition_stats",
        ],
        "description": "Statistical measures and pose estimation quality metrics",
    },
}

# --------------------------------------------------------------------------------------------------------
# Feature extractor with optional GUI selection and per-feature time windows
class FishFeatureExtractor(ConfigReader, FeatureExtractionMixin):
    """
    SimBA user-defined feature extractor with an optional GUI to select kinematic features
    and configure time windows (global and feature-specific).

    Programmatic control:
        FishFeatureExtractor(
            config_path="...",
            included_features=[...],            # optional
            excluded_features=[...],            # optional
            custom_time_windows={               # optional
                "global": [75, 50, 25],         # frames; use negative ints to mean seconds (e.g., [-5, -10])
                "feature_specific": {
                    "velocity": [-5, -10, -15], # seconds (converted using FPS per file)
                    "acceleration": [4, 10, 20] # frames
                }
            },
            use_gui=False
        )
    """

    def __init__(self, config_path: str, included_features: list = None, excluded_features: list = None,
                 custom_time_windows: dict = None, use_gui: bool = True):
        # 1) If caller explicitly provides include/exclude lists, we skip the GUI.
        selected_features = None
        time_windows_config = {}

        if use_gui and (included_features is None and excluded_features is None and custom_time_windows is None):
            selected_features, time_windows_config = self._open_feature_dialog()
            # Fallbacks if user cancels/empty
            if not selected_features:
                selected_features = ["movement", "velocity", "acceleration"]
            if not time_windows_config:
                time_windows_config = {"global": [75, 50, 25, 20, 15, 10, 4, 2], "feature_specific": {}}

        # Initialize base classes AFTER GUI to avoid issues on headless nodes
        ConfigReader.__init__(self, config_path=config_path)
        FeatureExtractionMixin.__init__(self, config_path=config_path)

        # 2) Resolve active features
        if selected_features is not None:
            # GUI path
            for f in selected_features:
                if f not in AVAILABLE_FEATURES:
                    raise ValueError(f"Unknown feature selected: {f}")
            self.included_features = selected_features
            self.excluded_features = []
        else:
            # Programmatic path
            self.included_features = AVAILABLE_FEATURES.copy() if included_features is None else included_features
            self.excluded_features = [] if excluded_features is None else excluded_features

            invalid_included = [f for f in self.included_features if f not in AVAILABLE_FEATURES]
            invalid_excluded = [f for f in self.excluded_features if f not in AVAILABLE_FEATURES]
            if invalid_included:
                raise ValueError(f"Invalid features in included_features: {invalid_included}")
            if invalid_excluded:
                raise ValueError(f"Invalid features in excluded_features: {invalid_excluded}")

        # This part remove excluded from included
        self.active_features = [f for f in self.included_features if f not in self.excluded_features]

        # 3) Handle time windows configuration
        self.time_windows_config = time_windows_config if time_windows_config else (custom_time_windows or {})
        # 4) Default windows (used if no config provided)
        self.roll_windows_values = [75, 50, 25, 20, 15, 10, 4, 2]

        print(f"[SimBA] Active features ({len(self.active_features)}): {self.active_features}")
        if self.excluded_features:
            print(f"[SimBA] Excluded features ({len(self.excluded_features)}): {self.excluded_features}")
        if self.time_windows_config:
            print(f"[SimBA] Time window config: {self.time_windows_config}")

        # 5) Common init
        self.timer = SimbaTimer()
        self.timer.start_timer()

        self.compass_brackets = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
        self.compass_brackets_long = [
            "Direction_N", "Direction_NE", "Direction_E", "Direction_SE",
            "Direction_S", "Direction_SW", "Direction_W", "Direction_NW",
        ]
        self.compass_brackets_digits = ["0", "1", "2", "3", "4", "5", "6", "7", "0"]

        self.config = read_config_file(config_path=config_path)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.input_file_dir = os.path.join(self.project_path, Paths.OUTLIER_CORRECTED.value)
        self.save_dir = os.path.join(self.project_path, Paths.FEATURES_EXTRACTED_DIR.value)
        self.video_info_path = os.path.join(self.project_path, Paths.VIDEO_INFO.value)
        self.video_info_df = pd.read_csv(self.video_info_path)

        bp_names_path = os.path.join(self.project_path, Paths.BP_NAMES.value)
        self.bp_names = list(pd.read_csv(bp_names_path, header=None)[0])

        self.col_headers_shifted = []
        for bp in self.bp_names:
            self.col_headers_shifted.extend((bp + "_x_shifted", bp + "_y_shifted", bp + "_p_shifted"))

        self.x_y_cols = []
        self.x_cols_shifted, self.y_cols_shifted = [], []
        for x_name, y_name in zip(self.x_cols, self.y_cols):
            self.x_y_cols.extend((x_name, y_name))
            self.x_cols_shifted.append(x_name + "_shifted")
            self.y_cols_shifted.append(y_name + "_shifted")

        # 6) Process files
        self.files_found = glob.glob(self.input_file_dir + "/*.{}".format(self.file_type))
        check_if_filepath_list_is_empty(
            filepaths=self.files_found,
            error_msg="SIMBA ERROR: No file in {} directory".format(self.input_file_dir),
        )
        print("Extracting features from {} file(s)...".format(len(self.files_found)))

        for file_path in self.files_found:
            video_timer = SimbaTimer(start=True)
            video_timer.start_timer()

            dir_name, file_name, ext = get_fn_ext(file_path)
            self.save_path = os.path.join(self.save_dir, os.path.basename(file_path))

            video_info, self.px_per_mm, self.fps = read_video_info(vid_info_df=self.video_info_df, video_name=file_name)
            self.video_width, self.video_height = (
                video_info["Resolution_width"].values,
                video_info["Resolution_height"].values,
            )

            # update time windows per file based on FPS + config
            self._update_time_windows_for_file()

            # Setup angular dispersion windows (seconds-based list multiplied by FPS)
            self.angular_dispersion_windows = []
            for i in range(len(ANGULAR_DISPERSION_S)):
                self.angular_dispersion_windows.append(int(self.fps * ANGULAR_DISPERSION_S[i]))

            # Read and prepare data
            self.csv_df = read_df(file_path, self.file_type).fillna(0).apply(pd.to_numeric)
            try:
                self.csv_df.columns = self.bp_headers
            except ValueError:
                msg = (
                    f"ERROR: Data contains the following fields: {self.csv_df.columns}. \\n"
                    f"SimBA wants to use the following field names {self.bp_header_list}"
                )
                print(msg)
                raise ValueError(msg)

            csv_df_shifted = self.csv_df.shift(periods=1)
            csv_df_shifted.columns = self.col_headers_shifted
            self.csv_df_combined = pd.concat([self.csv_df, csv_df_shifted], axis=1, join="inner").fillna(0)

            # --- Conditional Feature Extraction Calls ---
            if "X_relative_to_Y_movement" in self.active_features:
                self.calc_X_relative_to_Y_movement()
            if "movement" in self.active_features:
                self.calc_movement()
            if "X_relative_to_Y_movement_rolling_windows" in self.active_features:
                self.calc_X_relative_to_Y_movement_rolling_windows()
            if "velocity" in self.active_features:
                self.calc_velocity()
            if "acceleration" in self.active_features:
                self.calc_acceleration()
            if "rotation" in self.active_features:
                self.calc_rotation()
            if "N_degree_direction_switches" in self.active_features:
                self.calc_N_degree_direction_switches()
            if "bouts_in_same_direction" in self.active_features:
                self.bouts_in_same_direction()
            if "45_degree_direction_switches" in self.active_features:
                self.calc_45_degree_direction_switches()
            if "hot_end_encode_compass" in self.active_features:
                self.hot_end_encode_compass()
            if "directional_switches_in_rolling_windows" in self.active_features:
                self.calc_directional_switches_in_rolling_windows()
            if "angular_dispersion" in self.active_features:
                self.calc_angular_dispersion()
            if "distances_between_body_part" in self.active_features:
                self.calc_distances_between_body_part()
            if "convex_hulls" in self.active_features:
                self.calc_convex_hulls()
            if "pose_confidence_probabilities" in self.active_features:
                self.pose_confidence_probabilities()
            if "distribution_tests" in self.active_features:
                self.distribution_tests()
            if "rhythmic_patterns" in self.active_features:
                self.calc_rhythmic_patterns()
            if "turning_metrics" in self.active_features:
                self.calc_turning_metrics()
            if "energy_metrics" in self.active_features:
                self.calc_energy_metrics()
            if "complexity_metrics" in self.active_features:
                self.calc_complexity_metrics()
            if "path_metrics" in self.active_features:
                self.calc_path_metrics()
            if "body_curvature" in self.active_features:
                self.calc_body_curvature()
            if "swimming_bouts" in self.active_features:
                self.analyze_swimming_bouts()
            if "tail_beat_features" in self.active_features:
                self.calc_tail_beat_features()
            if "inter_body_coordination" in self.active_features:
                self.calc_inter_body_coordination()
            if "jerk_and_angular_acceleration" in self.active_features:
                self.calc_jerk_and_angular_acceleration()
            if "frequency_domain_features" in self.active_features:
                self.calc_frequency_domain_features()
            if "lateral_symmetry" in self.active_features:
                self.calc_lateral_symmetry()
            if "fractal_dimension" in self.active_features:
                self.calc_fractal_dimension()
            if "bout_transition_stats" in self.active_features:
                self.calc_bout_transition_stats()
            if "spatial_occupancy" in self.active_features:
                self.calc_spatial_occupancy()

            self.save_file()
            video_timer.stop_timer()
            print("Features extracted for video {} (elapsed time {}s)...".format(file_name, video_timer.elapsed_time_str))

        self.timer.stop_timer()
        print(
            "Features extracted for all {} files, data saved in {} (elapsed time {}s)".format(
                len(self.files_found), os.path.join(self.project_path, "csv", "features_extracted"), self.timer.elapsed_time_str
            )
        )

    # ---------------- GUI: returns (features, time_windows_config) ----------------
    def _open_feature_dialog(self):
        """
        Open a tkinter-based GUI to select features by category and configure time windows.
        Returns (selected_features_list, time_windows_config_dict)
        """
        try:
            import tkinter as tk
            from tkinter import ttk, messagebox
        except Exception as e:
            print(f"[SimBA] Tkinter not available ({e!r}). Skipping GUI.")
            return [], {}

        root = tk._default_root
        created_root = False
        if root is None:
            try:
                root = tk.Tk()
                root.withdraw()
                created_root = True
            except Exception as e:
                print(f"[SimBA] Could not create Tk root ({e!r}).")
                return [], {}

        win = tk.Toplevel(root)
        win.title("Select Kinematic Features & Time Windows")
        win.geometry("1000x800")
        win.resizable(True, True)
        try:
            win.transient(root)
            win.grab_set()
        except Exception:
            pass

        main_container = ttk.Frame(win)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        notebook = ttk.Notebook(main_container)
        notebook.pack(fill="both", expand=True)

        # ===== TAB 1: Features =====
        features_frame = ttk.Frame(notebook)
        notebook.add(features_frame, text="Feature Selection")

        canvas = tk.Canvas(features_frame)
        vsb = ttk.Scrollbar(features_frame, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas)

        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)

        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        ttk.Label(frame, text="Select Kinematic Features to Extract", font=("Arial", 16, "bold")).pack(anchor="w", pady=(0, 8))
        ttk.Label(frame, text=("Select features by category. Toggle entire categories or pick individual features.\n"
                               "Tip: Start with 'Basic Movement' if unsure."), foreground="gray").pack(anchor="w", pady=(0, 12))

        category_vars = {}
        feature_vars = {}
        for cat_name, cat in FEATURE_CATEGORIES.items():
            lf = ttk.LabelFrame(frame, text=cat_name)
            lf.pack(fill="x", expand=True, pady=6)

            ttk.Label(lf, text=cat["description"], foreground="blue").pack(anchor="w", padx=6, pady=(4, 6))

            category_vars[cat_name] = tk.BooleanVar(value=(cat_name == "Basic Movement"))
            ttk.Checkbutton(lf, text=f"Select all in {cat_name}", variable=category_vars[cat_name]).pack(anchor="w", padx=6, pady=(0, 6))

            grid = ttk.Frame(lf)
            grid.pack(fill="x", padx=16, pady=(0, 8))

            feats = cat["features"]
            cols = 2 if len(feats) > 4 else 1

            for i, feat in enumerate(feats):
                if feat not in AVAILABLE_FEATURES:
                    continue
                r, c = i // cols, i % cols
                feature_vars[feat] = tk.BooleanVar(value=(cat_name == "Basic Movement"))
                ttk.Checkbutton(grid, text=feat.replace("_", " ").title(), variable=feature_vars[feat]).grid(row=r, column=c, sticky="w", padx=8, pady=2)

            for c in range(cols):
                grid.columnconfigure(c, weight=1)

            def make_handler(cat_name_):
                def on_toggle(*_):
                    state = category_vars[cat_name_].get()
                    for feat in FEATURE_CATEGORIES[cat_name_]["features"]:
                        if feat in feature_vars:
                            feature_vars[feat].set(state)
                return on_toggle
            category_vars[cat_name].trace_add("write", make_handler(cat_name))

        # ===== TAB 2: Time Windows =====
        windows_frame = ttk.Frame(notebook)
        notebook.add(windows_frame, text="Time Windows")

        ttk.Label(windows_frame, text="Configure Time Windows for Feature Extraction", font=("Arial", 16, "bold")).pack(anchor="w", pady=(0, 8))
        ttk.Label(
            windows_frame,
            text=("Choose presets or specify custom windows.\n"
                  "Custom windows can be given in frames, or in seconds (converted using FPS per video)."),
            foreground="gray",
        ).pack(anchor="w", pady=(0, 12))

        window_mode = tk.StringVar(value="preset")
        mode_frame = ttk.LabelFrame(windows_frame, text="Window Configuration Mode")
        mode_frame.pack(fill="x", pady=8)
        ttk.Radiobutton(mode_frame, text="Use Preset Windows", variable=window_mode, value="preset").pack(anchor="w", padx=10, pady=4)
        ttk.Radiobutton(mode_frame, text="Custom Windows", variable=window_mode, value="custom").pack(anchor="w", padx=10, pady=4)

        preset_frame = ttk.LabelFrame(windows_frame, text="Preset Time Windows")
        preset_frame.pack(fill="x", pady=8)

        preset_vars = {}
        presets = {
            "Ultra Short (2-4 frames)": [2, 4],
            "Short (10-25 frames)": [10, 15, 20, 25],
            "Medium (50-75 frames)": [50, 75],
            "All Default": [75, 50, 25, 20, 15, 10, 4, 2],
            "Seconds-based (assume 30fps)": [30, 60, 150, 300],  # 1s, 2s, 5s, 10s @30 FPS
        }
        for preset_name, values in presets.items():
            preset_vars[preset_name] = tk.BooleanVar(value=(preset_name == "All Default"))
            ttk.Checkbutton(preset_frame, text=f"{preset_name}: {values}", variable=preset_vars[preset_name]).pack(anchor="w", padx=10, pady=2)

        custom_frame = ttk.LabelFrame(windows_frame, text="Custom Windows")
        custom_frame.pack(fill="x", pady=8)
        ttk.Label(custom_frame, text="Enter custom window values (comma-separated integers):").pack(anchor="w", padx=10, pady=5)
        custom_windows_var = tk.StringVar(value="30, 60, 150, 300")
        ttk.Entry(custom_frame, textvariable=custom_windows_var, width=60).pack(anchor="w", padx=10, pady=5)

        ttk.Label(custom_frame, text="Units:").pack(anchor="w", padx=10, pady=(10, 0))
        units_var = tk.StringVar(value="frames")
        ttk.Radiobutton(custom_frame, text="Frames", variable=units_var, value="frames").pack(anchor="w", padx=20)
        ttk.Radiobutton(custom_frame, text="Seconds (convert using FPS)", variable=units_var, value="seconds").pack(anchor="w", padx=20)

        specific_frame = ttk.LabelFrame(windows_frame, text="Feature-Specific Windows (Optional)")
        specific_frame.pack(fill="x", pady=8)
        ttk.Label(specific_frame, text="Override windows for specific features (comma-separated):").pack(anchor="w", padx=10, pady=5)

        specific_container = ttk.Frame(specific_frame)
        specific_container.pack(fill="x", padx=10, pady=5)

        specific_vars = {}
        windowed_features = [
            "velocity",
            "acceleration",
            "movement",
            "distances_between_body_part",
            "convex_hulls",
            "energy_metrics",
            "complexity_metrics",
            "path_metrics",
            "body_curvature",
            "directional_switches_in_rolling_windows",
            "angular_dispersion",
        ]
        for i, feat in enumerate(windowed_features):
            row, col = i // 2, i % 2
            rowf = ttk.Frame(specific_container)
            rowf.grid(row=row, column=col, sticky="w", padx=10, pady=2)
            ttk.Label(rowf, text=f"{feat.replace('_', ' ').title()}:").pack(side="left")
            specific_vars[feat] = tk.StringVar(value="")
            ttk.Entry(rowf, textvariable=specific_vars[feat], width=22).pack(side="left", padx=(6, 0))

        for c in range(2):
            specific_container.columnconfigure(c, weight=1)

        # Buttons for the whole dialog
        btns = ttk.Frame(main_container)
        btns.pack(fill="x", pady=(10, 0))

        def select_all_features():
            for v in category_vars.values():
                v.set(True)
            for v in feature_vars.values():
                v.set(True)

        def select_none_features():
            for v in category_vars.values():
                v.set(False)
            for v in feature_vars.values():
                v.set(False)

        def select_basic_features():
            select_none_features()
            for f in FEATURE_CATEGORIES["Basic Movement"]["features"]:
                if f in feature_vars:
                    feature_vars[f].set(True)

        ttk.Button(btns, text="Select All Features", command=select_all_features).pack(side="left", padx=4)
        ttk.Button(btns, text="Select None", command=select_none_features).pack(side="left", padx=4)
        ttk.Button(btns, text="Basic Only", command=select_basic_features).pack(side="left", padx=4)

        result = {"features": [], "time_windows": {}}

        def on_ok():
            # Features
            chosen_features = [f for f, v in feature_vars.items() if v.get()]
            if not chosen_features:
                try:
                    messagebox.showwarning("No Features Selected", "Please select at least one feature.")
                except Exception:
                    pass
                return

            # Global windows
            time_windows = []
            feature_specific_windows = {}

            if window_mode.get() == "preset":
                for preset_name, preset_var in preset_vars.items():
                    if preset_var.get():
                        time_windows.extend(presets[preset_name])
            else:
                try:
                    custom_vals = [int(x.strip()) for x in custom_windows_var.get().split(",") if x.strip()]
                except ValueError:
                    try:
                        messagebox.showerror("Invalid Input", "Custom windows must be comma-separated integers.")
                    except Exception:
                        pass
                    return
                # mark seconds as negative; frames as positive (convert later using fps)
                if units_var.get() == "seconds":
                    custom_vals = [-abs(x) for x in custom_vals]
                time_windows.extend(custom_vals)

            # Feature-specific
            for feat, var in specific_vars.items():
                val = var.get().strip()
                if val:
                    try:
                        spec_vals = [int(x.strip()) for x in val.split(",") if x.strip()]
                    except ValueError:
                        try:
                            messagebox.showerror("Invalid Input", f"Windows for {feat} must be comma-separated integers.")
                        except Exception:
                            pass
                        return
                    if units_var.get() == "seconds":
                        spec_vals = [-abs(x) for x in spec_vals]
                    feature_specific_windows[feat] = spec_vals

            # De-dupe + sort
            time_windows = sorted(list(set(time_windows)), reverse=True) if time_windows else []

            result["features"] = chosen_features
            result["time_windows"] = {
                "global": time_windows,
                "feature_specific": feature_specific_windows,
                "units_mode": units_var.get(),
            }
            win.destroy()

        def on_cancel():
            result["features"] = []
            result["time_windows"] = {}
            win.destroy()

        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="right", padx=4)
        ttk.Button(btns, text="Extract Selected Features", command=on_ok).pack(side="right", padx=4)

        win.update_idletasks()
        x = (win.winfo_screenwidth() // 2) - (win.winfo_width() // 2)
        y = (win.winfo_screenheight() // 2) - (win.winfo_height() // 2)
        win.geometry(f"+{x}+{y}")

        try:
            root.wait_window(win)
        finally:
            pass
        try:
            if created_root and root is not None:
                root.destroy()
        except Exception:
            pass

        return result.get("features", []), result.get("time_windows", {})

    # ---------------- Time-window helpers ----------------
    def _update_time_windows_for_file(self):
        """Update self.roll_windows_values based on time_windows_config and current FPS."""
        if not getattr(self, "time_windows_config", None):
            return
        global_windows = self.time_windows_config.get("global", [])
        if not global_windows:
            return

        converted = []
        for w in global_windows:
            if w < 0:  # seconds
                converted.append(int(abs(w) * self.fps))
            else:      # frames
                converted.append(int(w))
        converted = [w for w in converted if w > 0]
        if converted:
            self.roll_windows_values = sorted(list(set(converted)), reverse=True)
            print(f"[SimBA] Updated global time windows for {self.fps} fps: {self.roll_windows_values}")

    def _get_windows_for_feature(self, feature_name: str):
        """Return feature-specific windows if configured; otherwise the global list."""
        if not getattr(self, "time_windows_config", None):
            return self.roll_windows_values
        feature_windows = self.time_windows_config.get("feature_specific", {}).get(feature_name, [])
        if not feature_windows:
            return self.roll_windows_values
        converted = []
        for w in feature_windows:
            if w < 0:  # seconds
                converted.append(int(abs(w) * self.fps))
            else:      # frames
                converted.append(int(w))
        converted = [w for w in converted if w > 0]
        return sorted(list(set(converted)), reverse=True) if converted else self.roll_windows_values

    # ---------------- Utility & math helpers (subset shown; align with your original) ----------------
    def angle2pt_degrees(self, ax, ay, bx, by):
        angle_degrees = math.degrees(math.atan2(ax - bx, by - ay))
        return angle_degrees + 360 if angle_degrees < 0 else angle_degrees

    def angle2pt_radians(self, degrees):
        return degrees * math.pi / 180

    def angle2pt_sin(self, angle_radians):
        return math.sin(angle_radians)

    def angle2pt_cos(self, angle_radians):
        return math.cos(angle_radians)

    @staticmethod
    @jit(nopython=True)
    def count_values_in_range(data: np.array, ranges: np.array):
        results = np.full((data.shape[0], ranges.shape[0]), 0)
        for i in prange(data.shape[0]):
            for j in prange(ranges.shape[0]):
                lower_bound, upper_bound = ranges[j][0], ranges[j][1]
                results[i][j] = data[i][np.logical_and(data[i] >= lower_bound, data[i] <= upper_bound)].shape[0]
        return results

    @staticmethod
    def convex_hull_calculator_mp(arr: np.array, px_per_mm: float) -> float:
        arr = np.unique(arr, axis=0).astype(int)
        if arr.shape[0] < 3:
            return 0
        for i in range(1, arr.shape[0]):
            if (arr[i] != arr[0]).all():
                try:
                    return ConvexHull(arr, qhull_options="En").area / px_per_mm
                except QhullError:
                    return 0
            else:
                pass
        return 0

    @staticmethod
    @jit(nopython=True)
    def euclidian_distance_calc(bp1xVals, bp1yVals, bp2xVals, bp2yVals):
        return np.sqrt((bp1xVals - bp2xVals) ** 2 + (bp1yVals - bp2yVals) ** 2)

    @staticmethod
    @jit(nopython=True)
    def angular_dispersion(cumsum_cos_np, cumsum_sin_np):
        out_array = np.empty((cumsum_cos_np.shape))
        for index in range(cumsum_cos_np.shape[0]):
            X = cumsum_cos_np[index] / (index + 1)
            Y = cumsum_sin_np[index] / (index + 1)
            out_array[index] = math.sqrt(X**2 + Y**2)
        return out_array

    def windowed_frequentist_distribution_tests(self, data: np.array, feature_name: str, fps: int):
        (ks_results,) = (np.full((data.shape[0]), -1.0),)
        t_test_results = np.full((data.shape[0]), -1.0)
        lillefors_results = np.full((data.shape[0]), -1.0)
        shapiro_results = np.full((data.shape[0]), -1.0)
        peak_cnt_results = np.full((data.shape[0]), -1.0)

        for i in range(fps, data.shape[0] - fps, fps):
            bin_1_idx, bin_2_idx = [i - fps, i], [i, i + fps]
            bin_1_data = data[bin_1_idx[0]:bin_1_idx[1]]
            bin_2_data = data[bin_2_idx[0]:bin_2_idx[1]]
            ks_results[i:i+fps+1] = stats.ks_2samp(data1=bin_1_data, data2=bin_2_data).statistic
            t_test_results[i:i+fps+1] = stats.ttest_ind(bin_1_data, bin_2_data).statistic

        for i in range(0, data.shape[0] - fps, fps):
            lillefors_results[i:i+fps+1] = lilliefors(data[i:i+fps])[0]
            shapiro_results[i:i+fps+1] = stats.shapiro(data[i:i+fps])[0]

        rolling_idx = np.arange(fps)[None, :] + 1 * np.arange(data.shape[0])[:, None]
        for i in range(rolling_idx.shape[0]):
            bin_start_idx, bin_end_idx = rolling_idx[i][0], rolling_idx[i][-1]
            peaks, _ = find_peaks(data[bin_start_idx:bin_end_idx], height=0)
            peak_cnt_results[i] = len(peaks)

        columns = [f"{feature_name}_KS", f"{feature_name}_TTEST", f"{feature_name}_LILLEFORS",
                   f"{feature_name}_SHAPIRO", f"{feature_name}_PEAK_CNT"]
        return pd.DataFrame(np.column_stack((ks_results, t_test_results, lillefors_results, shapiro_results, peak_cnt_results)), columns=columns).round(4)

    @staticmethod
    @jit(nopython=True)
    def consecutive_frames_in_same_compass_direction(direction: np.array):
        results = np.full((direction.shape[0], 1), -1)
        cnt = 0
        results[0] = 0
        last_direction = direction[0]
        for i in prange(1, direction.shape[0]):
            if direction[i] == last_direction:
                cnt += 1
            else:
                cnt = 0
            results[i] = cnt
            last_direction = direction[i]
        return results.flatten()

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def framewise_degree_shift(clockwise_angle: np.array):
        degree_shift = np.full((clockwise_angle.shape[0], 1), np.nan)
        last_angle = clockwise_angle[0]
        for i in prange(1, clockwise_angle.shape[0]):
            degree_shift[i] = math.atan2(math.sin(clockwise_angle[i] - last_angle), math.cos(clockwise_angle[i] - last_angle))
            last_angle = clockwise_angle[i]
        return np.absolute(degree_shift.flatten())

    # ---------------- Feature family methods with per-feature windows ----------------
    def bouts_in_same_direction(self):
        self.csv_df_combined["Consecutive_ms_in_same_compass_direction"] = (
            self.consecutive_frames_in_same_compass_direction(direction=self.csv_df_combined["Compass_digit"].values.astype(int))
            / self.fps
        )
        self.csv_df_combined["Consecutive_ms_in_same_compass_direction_zscore"] = zscore(self.csv_df_combined["Consecutive_ms_in_same_compass_direction"].values)

        windows = self._get_windows_for_feature("directional_switches_in_rolling_windows")
        for window in windows:
            self.csv_df_combined[f"Unique_compass_directions_in_{window}_window"] = (
                self.csv_df_combined["Compass_digit"].astype(int).rolling(window, min_periods=1).apply(lambda x: len(np.unique(x))).astype(int)
            )

        framewise_degree_shift = pd.Series(self.framewise_degree_shift(clockwise_angle=self.csv_df_combined["Clockwise_angle_degrees"].values))
        for window in windows:
            self.csv_df_combined[f"Degree_shift_{window}_mean"] = framewise_degree_shift.rolling(window, min_periods=1).mean()
            self.csv_df_combined[f"Degree_shift_{window}_median"] = framewise_degree_shift.rolling(window, min_periods=1).median()
            self.csv_df_combined[f"Degree_shift_{window}_sum"] = framewise_degree_shift.rolling(window, min_periods=1).sum()
            self.csv_df_combined[f"Degree_shift_{window}_std"] = framewise_degree_shift.rolling(window, min_periods=1).std()

    def calc_angular_dispersion(self):
        dispersion_array = self.angular_dispersion(self.csv_df_combined["Angle_cos_cumsum"].values,
                                                     self.csv_df_combined["Angle_sin_cumsum"].values)
        self.csv_df_combined["Angular_dispersion"] = dispersion_array
        windows = self._get_windows_for_feature("angular_dispersion")
        for win in windows:
            col_name = "Angular_dispersion_window_" + str(win)
            self.csv_df_combined[col_name] = self.csv_df_combined["Angular_dispersion"].rolling(win, min_periods=1).mean()

    def calc_X_relative_to_Y_movement(self):
        temp_df = pd.DataFrame()
        for bp in range(len(self.x_cols)):
            curr_x_col = self.x_cols[bp]
            curr_x_shifted_col = self.x_cols_shifted[bp]
            curr_y_col = self.y_cols[bp]
            curr_y_shifted_col = self.y_cols_shifted[bp]
            temp_df["x"] = self.csv_df_combined[curr_x_col] - self.csv_df_combined[curr_x_shifted_col]
            temp_df["y"] = self.csv_df_combined[curr_y_col] - self.csv_df_combined[curr_y_shifted_col]
            temp_df["Movement_{}_X_relative_2_Y".format(bp)] = temp_df["x"] - temp_df["y"]
            temp_df.drop(["x", "y"], axis=1, inplace=True)
        self.csv_df_combined["Movement_X_axis_relative_to_Y_axis"] = temp_df.sum(axis=1)

    def calc_movement(self):
        movement_cols = []
        for bp in self.bp_names:
            self.csv_df_combined[f"{bp}_movement"] = (
                self.euclidian_distance_calc(self.csv_df_combined[f"{bp}_x"].values,
                                               self.csv_df_combined[f"{bp}_y"].values,
                                               self.csv_df_combined[f"{bp}_x_shifted"].values,
                                               self.csv_df_combined[f"{bp}_y_shifted"].values) / self.px_per_mm
            )
            movement_cols.append(f"{bp}_movement")
        self.csv_df_combined["Summed_movement"] = self.csv_df_combined[movement_cols].sum(axis=1)

        windows = self._get_windows_for_feature("movement")
        for bp in self.bp_names:
            for window in windows:
                self.csv_df_combined[f"{bp}_movement_{window}_mean"] = self.csv_df_combined[f"{bp}_movement"].rolling(window, min_periods=1).mean()
                self.csv_df_combined[f"{bp}_movement_{window}_sum"] = self.csv_df_combined[f"{bp}_movement"].rolling(window, min_periods=1).sum()

    def calc_X_relative_to_Y_movement_rolling_windows(self):
        windows = self._get_windows_for_feature("movement")
        for i in windows:
            self.csv_df_combined[f"Movement_X_axis_relative_to_Y_axis_mean_{i}"] = self.csv_df_combined["Movement_X_axis_relative_to_Y_axis"].rolling(i, min_periods=1).mean()
            self.csv_df_combined[f"Movement_X_axis_relative_to_Y_axis_sum_{i}"] = self.csv_df_combined["Movement_X_axis_relative_to_Y_axis"].rolling(i, min_periods=1).sum()

    def calc_directional_switches_in_rolling_windows(self):
        windows = self._get_windows_for_feature("directional_switches_in_rolling_windows")
        for i in windows:
            self.csv_df_combined[f"Number_of_direction_switches_{i}"] = self.csv_df_combined["Direction_switch"].rolling(i, min_periods=1).sum()
            self.csv_df_combined[f"Directionality_of_switches_switches_{i}"] = self.csv_df_combined["Switch_direction_value"].rolling(i, min_periods=1).sum()

    def calc_velocity(self):
        self.velocity_fields = []
        for bp in self.bp_names:
            self.csv_df_combined[f"{bp}_velocity"] = self.csv_df_combined[bp + "_movement"].rolling(int(self.fps), min_periods=1).sum()
            self.velocity_fields.append(bp + "_velocity")
        self.csv_df_combined["Bp_velocity_mean"] = self.csv_df_combined[self.velocity_fields].mean(axis=1)
        self.csv_df_combined["Bp_velocity_stdev"] = self.csv_df_combined[self.velocity_fields].std(axis=1)

        windows = self._get_windows_for_feature("velocity")
        for i in windows:
            self.csv_df_combined[f"Minimum_avg_bp_velocity_{i}_window"] = self.csv_df_combined["Bp_velocity_mean"].rolling(i, min_periods=1).min()
            self.csv_df_combined[f"Max_avg_bp_velocity_{i}_window"] = self.csv_df_combined["Bp_velocity_mean"].rolling(i, min_periods=1).max()
            self.csv_df_combined[f"Absolute_diff_min_max_avg_bp_velocity_{i}_window"] = abs(self.csv_df_combined[f"Minimum_avg_bp_velocity_{i}_window"] - self.csv_df_combined[f"Max_avg_bp_velocity_{i}_window"])

    def calc_acceleration(self):
        windows = self._get_windows_for_feature("acceleration")
        for i in windows:
            acceleration_fields = []
            for bp in self.bp_names:
                self.csv_df_combined[f"{bp}_velocity_shifted"] = self.csv_df_combined[f"{bp}_velocity"].shift(i).fillna(self.csv_df_combined[f"{bp}_velocity"])
                self.csv_df_combined[f"{bp}_acceleration_{i}_window"] = self.csv_df_combined[f"{bp}_velocity"] - self.csv_df_combined[f"{bp}_velocity_shifted"]
                self.csv_df_combined = self.csv_df_combined.drop([f"{bp}_velocity_shifted"], axis=1)
                acceleration_fields.append(f"{bp}_acceleration_{i}_window")
            self.csv_df_combined[f"Bp_acceleration_mean_{i}_window"] = self.csv_df_combined[acceleration_fields].mean(axis=1)
            self.csv_df_combined[f"Bp_acceleration_stdev_{i}_window"] = self.csv_df_combined[acceleration_fields].std(axis=1)

        for i in windows:
            self.csv_df_combined[f"Min_avg_bp_acceleration_{i}_window"] = self.csv_df_combined[f"Bp_acceleration_mean_{i}_window"].rolling(i, min_periods=1).mean()
            self.csv_df_combined[f"Max_avg_bp_acceleration_{i}_window"] = self.csv_df_combined[f"Bp_acceleration_mean_{i}_window"].rolling(i, min_periods=1).mean()
            self.csv_df_combined[f"Absolute_diff_min_max_avg_bp_acceleration_{i}_window"] = abs(self.csv_df_combined[f"Min_avg_bp_acceleration_{i}_window"] - self.csv_df_combined[f"Max_avg_bp_acceleration_{i}_window"])

    def calc_N_degree_direction_switches(self):
        degree_lk_180 = {"N": ["S"], "NE": ["SW"], "E": ["W"], "SE": ["NW"]}
        degree_lk_90 = {
            "N": ["W", "E"],
            "NE": ["NW", "SE"],
            "NW": ["SW", "NE"],
            "SW": ["NW", "SE"],
            "SE": ["NE", "SW"],
            "S": ["W", "E"],
            "E": ["N", "S"],
            "W": ["N", "S"],
        }
        dg_df = pd.DataFrame(self.csv_df_combined["Compass_direction"])
        windows = self._get_windows_for_feature("directional_switches_in_rolling_windows")
        for window in windows:
            dg_df[f"Compass_direction_{window}"] = dg_df["Compass_direction"].shift(window)
            dg_df[f"Compass_direction_{window}"].fillna(dg_df["Compass_direction"], inplace=True)
            dg_df[f"180_degree_switch_{window}"] = 0
            dg_df[f"90_degree_switch_{window}"] = 0
            for k, v in degree_lk_180.items():
                for value in v:
                    dg_df.loc[(dg_df["Compass_direction"] == k) & (dg_df[f"Compass_direction_{window}"] == value), f"180_degree_switch_{window}"] = 1
                    dg_df.loc[(dg_df[f"Compass_direction_{window}"] == k) & (dg_df["Compass_direction"] == value), f"180_degree_switch_{window}"] = 1
            for k, v in degree_lk_90.items():
                for value in v:
                    dg_df.loc[(dg_df["Compass_direction"] == k) & (dg_df[f"Compass_direction_{window}"] == value), f"90_degree_switch_{window}"] = 1
                    dg_df.loc[(dg_df[f"Compass_direction_{window}"] == k) & (dg_df["Compass_direction"] == value), f"90_degree_switch_{window}"] = 1
            self.csv_df_combined[f"180_degree_switch_{window}"] = dg_df[f"180_degree_switch_{window}"]
            self.csv_df_combined[f"90_degree_switch_{window}"] = dg_df[f"90_degree_switch_{window}"]

    def calc_rotation(self):
        # Uses bodypart2 and bodypart16 for rotation/orientation
        self.csv_df_combined["Clockwise_angle_degrees"] = self.csv_df_combined.apply(
            lambda x: self.angle2pt_degrees(x["bodypart2_x"],
                                             x["bodypart2_y"],
                                             x["bodypart16_x"],
                                             x["bodypart16_y"]),
            axis=1,
        )
        self.csv_df_combined["Angle_radians"] = self.angle2pt_radians(self.csv_df_combined["Clockwise_angle_degrees"])
        self.csv_df_combined["Angle_sin"] = self.csv_df_combined.apply(lambda x: self.angle2pt_sin(x["Angle_radians"]), axis=1)
        self.csv_df_combined["Angle_cos"] = self.csv_df_combined.apply(lambda x: self.angle2pt_cos(x["Angle_radians"]), axis=1)
        self.csv_df_combined["Angle_sin_cumsum"] = self.csv_df_combined["Angle_sin"].cumsum()
        self.csv_df_combined["Angle_cos_cumsum"] = self.csv_df_combined["Angle_cos"].cumsum()
        compass_lookup = list(round(self.csv_df_combined["Clockwise_angle_degrees"] / 45))
        compass_lookup = [int(i) for i in compass_lookup]
        compasFaceList_bracket, compasFaceList_digit = [], []
        for compasDirection in compass_lookup:
            compasFaceList_bracket.append(self.compass_brackets[compasDirection])
            compasFaceList_digit.append(self.compass_brackets_digits[compasDirection])
        self.csv_df_combined["Compass_direction"] = compasFaceList_bracket
        self.csv_df_combined["Compass_digit"] = compasFaceList_digit

        windows = self._get_windows_for_feature("rotation")
        for i in windows:
            self.csv_df_combined[f"Mean_angle_time_window_{i}"] = self.csv_df_combined["Clockwise_angle_degrees"].rolling(i, min_periods=1).mean()

    def hot_end_encode_compass(self):
        compass_hot_end = pd.get_dummies(self.csv_df_combined["Compass_direction"], prefix="Direction")
        compass_hot_end = compass_hot_end.T.reindex(self.compass_brackets_long).T.fillna(0)
        self.csv_df_combined = pd.concat([self.csv_df_combined, compass_hot_end], axis=1)

    def calc_45_degree_direction_switches(self):
        self.grouped_df = pd.DataFrame()
        v = (self.csv_df_combined["Compass_digit"] != self.csv_df_combined["Compass_digit"].shift()).cumsum()
        u = self.csv_df_combined.groupby(v)["Compass_digit"].agg(["all", "count"])
        m = u["all"] & u["count"].ge(1)
        self.grouped_df["groups"] = self.csv_df_combined.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
        currdirectionList, DirectionSwitchIndexList, currdirectionListValue = [], [], []
        for indexes, row in self.grouped_df.iterrows():
            currdirectionList.append(self.csv_df_combined.loc[row["groups"][0]]["Compass_direction"])
            DirectionSwitchIndexList.append(row["groups"][1])
            currdirectionListValue.append(self.csv_df_combined.loc[row["groups"][0]]["Compass_digit"])
        self.grouped_df["Direction_switch"] = currdirectionList
        self.grouped_df["Direction_value"] = currdirectionListValue
        self.csv_df_combined.loc[DirectionSwitchIndexList, "Direction_switch"] = 1
        self.csv_df_combined["Compass_digit_shifted"] = self.csv_df_combined["Compass_digit"].shift(-1)
        self.csv_df_combined = self.csv_df_combined.fillna(0)
        self.csv_df_combined["Switch_direction_value"] = self.csv_df_combined.apply(lambda x: self.calc_switch_direction(x["Compass_digit_shifted"], x["Compass_digit"]), axis=1)

    def calc_switch_direction(self, compass_digit_shifted, compass_digit):
        if (compass_digit_shifted == "0") and (compass_digit == "7"):
            return 1
        else:
            return int(compass_digit_shifted) - int(compass_digit)

    def calc_distances_between_body_part(self):
        two_point_combs = np.array(list(combinations(self.bp_names, 2)))
        distance_fields = []
        for bps in two_point_combs:
            self.csv_df_combined[f"Distance_{bps[0]}_{bps[1]}"] = (
                self.euclidian_distance_calc(self.csv_df_combined[bps[0] + "_x"].values,
                                               self.csv_df_combined[bps[0] + "_y"].values,
                                               self.csv_df_combined[bps[1] + "_x"].values,
                                               self.csv_df_combined[bps[1] + "_y"].values) / self.px_per_mm
            )
            distance_fields.append(f"Distance_{bps[0]}_{bps[1]}")

        windows = self._get_windows_for_feature("distances_between_body_part")
        for distance_field in distance_fields:
            for window in windows:
                self.csv_df_combined[f"{distance_field}_mean_{window}"] = self.csv_df_combined[distance_field].rolling(window, min_periods=1).mean()
                self.csv_df_combined[f"{distance_field}_std_{window}"] = self.csv_df_combined[distance_field].rolling(window, min_periods=1).std()
                try:
                    self.csv_df_combined[f"{distance_field}_skew_{window}"] = self.csv_df_combined[distance_field].rolling(window, min_periods=1).skew()
                    self.csv_df_combined[f"{distance_field}_kurtosis_{window}"] = self.csv_df_combined[distance_field].rolling(window, min_periods=1).kurt()
                except Exception:
                    self.csv_df_combined[f"{distance_field}_skew_{window}"] = -1
                    self.csv_df_combined[f"{distance_field}_kurtosis_{window}"] = -1

    def calc_convex_hulls(self):
        # NOTE: shape calc mirrors original intent but ensure reshape length correctness
        arr = self.csv_df[self.x_y_cols].values
        n_frames = len(self.csv_df)
        n_bps = len(self.bp_names)
        fish_array = np.reshape(arr, (n_frames, n_bps, 2))

        self.csv_df_combined["Convex_hull"] = Parallel(n_jobs=-1, verbose=0, backend="threading")(
            delayed(self.convex_hull_calculator_mp)(x, self.px_per_mm) for x in fish_array
        )
        windows = self._get_windows_for_feature("convex_hulls")
        for window in windows:
            self.csv_df_combined[f"Convex_hull_mean_{window}_window"] = self.csv_df_combined["Convex_hull"].rolling(window, min_periods=1).mean()
            self.csv_df_combined[f"Convex_hull_std_{window}_window"] = self.csv_df_combined["Convex_hull"].rolling(window, min_periods=1).std()
            self.csv_df_combined[f"Convex_hull_min_{window}_window"] = self.csv_df_combined["Convex_hull"].rolling(window, min_periods=1).min()
            self.csv_df_combined[f"Convex_hull_max_{window}_window"] = self.csv_df_combined["Convex_hull"].rolling(window, min_periods=1).max()
            self.csv_df_combined[f"Absolute_diff_min_max_convex_hull_{window}_window"] = abs(
                self.csv_df_combined[f"Convex_hull_min_{window}_window"] - self.csv_df_combined[f"Convex_hull_max_{window}_window"]
            )
            try:
                self.csv_df_combined[f"Convex_hull_skew_{window}"] = self.csv_df_combined["Convex_hull"].rolling(window, min_periods=1).skew()
                self.csv_df_combined[f"Convex_hull_kurtosis_{window}"] = self.csv_df_combined["Convex_hull"].rolling(window, min_periods=1).kurt()
            except Exception:
                self.csv_df_combined[f"Convex_hull_skew_{window}"] = -1
                self.csv_df_combined[f"Convex_hull_kurtosis_{window}"] = -1

    def distribution_tests(self):
        distribution_features = [
            "Bp_velocity_mean",
            "Bp_acceleration_mean_25_window",
            "Clockwise_angle_degrees",
            "Convex_hull",
            "Sum_probabilities",
            "Consecutive_ms_in_same_compass_direction",
        ]
        for feature_name in distribution_features:
            if feature_name in self.csv_df_combined.columns:
                results = self.windowed_frequentist_distribution_tests(
                    data=self.csv_df_combined[feature_name].values,
                    feature_name=feature_name,
                    fps=int(self.fps),
                )
                self.csv_df_combined = pd.concat([self.csv_df_combined, results], axis=1)

    def pose_confidence_probabilities(self):
        self.csv_df_combined["Sum_probabilities"] = self.csv_df_combined[self.p_cols].sum(axis=1)
        self.csv_df_combined["Sum_probabilities_deviation"] = (self.csv_df_combined["Sum_probabilities"].mean() - self.csv_df_combined["Sum_probabilities"])
        p_brackets_results = pd.DataFrame(
            self.count_values_in_range(
                data=self.csv_df_combined.filter(self.p_cols).values,
                ranges=np.array([[0.0, 0.1], [0.0, 0.5], [0.0, 0.75], [0.0, 0.95], [0.0, 0.99]]),
            ),
            columns=["Low_prob_detections_0.1", "Low_prob_detections_0.5", "Low_prob_detections_0.75", "Low_prob_detections_0.95", "Low_prob_detections_0.99"],
        )
        self.csv_df_combined = pd.concat([self.csv_df_combined, p_brackets_results], axis=1).reset_index(drop=True).fillna(0)

    def calc_rhythmic_patterns(self):
        """Analyze rhythmic patterns in movement using autocorrelation and spectral analysis"""
        from scipy import signal
        for bp in self.bp_names:
            movement_data = self.csv_df_combined.get(f"{bp}_movement", None)
            if movement_data is None:
                continue
            movement_data = movement_data.values
            autocorr = np.correlate(movement_data, movement_data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            if autocorr[0] != 0:
                self.csv_df_combined[f"{bp}_movement_autocorr_peak"] = np.max(autocorr[1:]) / autocorr[0]
            if len(movement_data) > self.fps * 2:
                freqs, psd = signal.welch(movement_data, fs=self.fps, nperseg=min(256, len(movement_data)//2))
                if len(psd) > 1:
                    dominant_freq_idx = np.argmax(psd[1:]) + 1
                    self.csv_df_combined[f"{bp}_movement_dominant_freq"] = freqs[dominant_freq_idx]
                    self.csv_df_combined[f"{bp}_movement_dominant_freq_power"] = psd[dominant_freq_idx]

    def calc_turning_metrics(self):
        """Calculate features related to turning behavior"""
        if "Clockwise_angle_degrees" not in self.csv_df_combined.columns:
            return
        angle_diff = np.diff(self.csv_df_combined["Clockwise_angle_degrees"].values, prepend=0)
        angle_diff = np.where(angle_diff > 180, angle_diff - 360, angle_diff)
        angle_diff = np.where(angle_diff < -180, angle_diff + 360, angle_diff)
        self.csv_df_combined["Turning_rate"] = np.abs(angle_diff) * self.fps / 360
        turn_window = max(1, int(self.fps / 4))
        rolling_turn = pd.Series(np.abs(angle_diff)).rolling(turn_window, min_periods=1).sum()
        self.csv_df_combined["Sharp_turn"] = (rolling_turn > 30).astype(int)
        v = (self.csv_df_combined["Sharp_turn"] != self.csv_df_combined["Sharp_turn"].shift()).cumsum()
        turn_bouts = self.csv_df_combined.groupby(v)["Sharp_turn"].agg(["sum", "count"])
        turn_bouts = turn_bouts[turn_bouts["sum"] > 0]
        self.csv_df_combined["Turn_bout_duration"] = 0
        for idx, bout in turn_bouts.iterrows():
            bout_frames = self.csv_df_combined[v == idx].index
            if len(bout_frames) > 0:
                self.csv_df_combined.loc[bout_frames, "Turn_bout_duration"] = len(bout_frames) / self.fps

    def calc_energy_metrics(self):
        """Estimate energy expenditure based on movement and acceleration"""
        for bp in self.bp_names:
            if f"{bp}_velocity" in self.csv_df_combined.columns:
                self.csv_df_combined[f"{bp}_kinetic_energy_proxy"] = 0.5 * self.csv_df_combined[f"{bp}_velocity"]**2
        energy_cols = [f"{bp}_kinetic_energy_proxy" for bp in self.bp_names if f"{bp}_kinetic_energy_proxy" in self.csv_df_combined.columns]
        if energy_cols:
            self.csv_df_combined["Total_kinetic_energy"] = self.csv_df_combined[energy_cols].sum(axis=1)
            self.csv_df_combined["Cumulative_energy"] = self.csv_df_combined["Total_kinetic_energy"].cumsum()
            windows = self._get_windows_for_feature("energy_metrics")
            for window in windows:
                self.csv_df_combined[f"Energy_rate_{window}"] = self.csv_df_combined["Total_kinetic_energy"].rolling(window, min_periods=1).mean()

    def calc_complexity_metrics(self):
        """Calculate complexity/entropy measures of movement patterns"""
        windows = self._get_windows_for_feature("complexity_metrics")
        for bp in self.bp_names:
            key = f"{bp}_movement"
            if key not in self.csv_df_combined.columns:
                continue
            for window in windows:
                if window > 10:
                    movement_window = self.csv_df_combined[key].rolling(window, min_periods=1)
                    def window_entropy(x):
                        bins = max(1, min(10, len(x) // 2))
                        hist, _ = np.histogram(x, bins=bins)
                        if np.sum(hist) > 0:
                            return entropy(hist / np.sum(hist))
                        return 0
                    self.csv_df_combined[f"{bp}_movement_entropy_{window}"] = movement_window.apply(window_entropy)

    def calc_path_metrics(self):
        """Calculate path tortuosity and swimming efficiency metrics using bodypart16"""
        bp = "bodypart16"
        key = f"{bp}_movement"
        if key not in self.csv_df_combined.columns:
            return
        windows = self._get_windows_for_feature("path_metrics")
        for window in windows:
            if window > 2:
                cum_dist = self.csv_df_combined[key].rolling(window, min_periods=2).sum()
                x_start = self.csv_df_combined[f"{bp}_x"].shift(window-1)
                y_start = self.csv_df_combined[f"{bp}_y"].shift(window-1)
                x_end = self.csv_df_combined[f"{bp}_x"]
                y_end = self.csv_df_combined[f"{bp}_y"]
                straight_dist = self.euclidian_distance_calc(x_start.values, y_start.values, x_end.values, y_end.values) / self.px_per_mm
                self.csv_df_combined[f"Path_tortuosity_{window}"] = cum_dist / (straight_dist + 1e-6)
                self.csv_df_combined[f"Swimming_efficiency_{window}"] = 1 / self.csv_df_combined[f"Path_tortuosity_{window}"]

    def calc_segment_angle(self, x1, y1, x2, y2, x3, y3):
        """Calculate the angle between three points (in degrees)"""
        v1x, v1y = x1 - x2, y1 - y2
        v2x, v2y = x3 - x2, y3 - y2
        dot_product = v1x * v2x + v1y * v2y
        mag_v1 = math.sqrt(v1x**2 + v1y**2)
        mag_v2 = math.sqrt(v2x**2 + v2y**2)
        if mag_v1 * mag_v2 == 0:
            return 0
        cos_angle = max(-1, min(1, dot_product / (mag_v1 * mag_v2)))
        angle = math.degrees(math.acos(cos_angle))
        cross_product = v1x * v2y - v1y * v2x
        if cross_product < 0:
            angle = -angle
        return angle

    def calc_body_curvature(self):
        """Calculate body curvature metrics using multiple body points"""
        if len(self.bp_names) >= 3:
            for i in range(len(self.bp_names) - 2):
                bp1, bp2, bp3 = self.bp_names[i], self.bp_names[i+1], self.bp_names[i+2]
                self.csv_df_combined[f"Segment_angle_{bp1}_{bp2}_{bp3}"] = self.csv_df_combined.apply(
                    lambda row: self.calc_segment_angle(row[f"{bp1}_x"], row[f"{bp1}_y"],
                                                        row[f"{bp2}_x"], row[f"{bp2}_y"],
                                                        row[f"{bp3}_x"], row[f"{bp3}_y"]),
                    axis=1
                )
            angle_cols = [col for col in self.csv_df_combined.columns if col.startswith("Segment_angle_")]
            if angle_cols:
                self.csv_df_combined["Total_body_curvature"] = self.csv_df_combined[angle_cols].sum(axis=1)
                self.csv_df_combined["Body_curvature_change"] = self.csv_df_combined["Total_body_curvature"].diff()
                windows = self._get_windows_for_feature("body_curvature")
                for window in windows:
                    self.csv_df_combined[f"Body_curvature_mean_{window}"] = self.csv_df_combined["Total_body_curvature"].rolling(window, min_periods=1).mean()
                    self.csv_df_combined[f"Body_curvature_std_{window}"] = self.csv_df_combined["Total_body_curvature"].rolling(window, min_periods=1).std()

    def analyze_swimming_bouts(self):
        """Identify and analyze swimming bouts (active swimming followed by coasting)"""
        if "Summed_movement" not in self.csv_df_combined.columns or "Bp_velocity_mean" not in self.csv_df_combined.columns:
            return
        movement_threshold = self.csv_df_combined["Summed_movement"].mean() * 1.5
        self.csv_df_combined["Active_swimming"] = (self.csv_df_combined["Summed_movement"] > movement_threshold).astype(int)
        bout_id = (self.csv_df_combined["Active_swimming"] != self.csv_df_combined["Active_swimming"].shift(1)).cumsum()
        self.csv_df_combined["Bout_ID"] = bout_id
        bout_stats = self.csv_df_combined.groupby(["Bout_ID", "Active_swimming"]).agg({
            "Summed_movement": ["mean", "max", "sum"],
            "Bp_velocity_mean": ["mean", "max"]
        })
        bout_stats.columns = ['_'.join(col).strip() for col in bout_stats.columns.values]
        bout_stats = bout_stats.reset_index()
        bout_counts = self.csv_df_combined.groupby(["Bout_ID", "Active_swimming"]).size().reset_index(name="bout_frames")
        bout_stats = pd.merge(bout_stats, bout_counts, on=["Bout_ID", "Active_swimming"])
        bout_stats["bout_duration"] = bout_stats["bout_frames"] / self.fps
        windows = self._get_windows_for_feature("swimming_bouts")
        for prop in ["bout_duration", "Summed_movement_mean", "Bp_velocity_mean_mean"]:
            self.csv_df_combined[f"Current_{prop}"] = 0
            for idx, row in bout_stats.iterrows():
                bout_frames = self.csv_df_combined[self.csv_df_combined["Bout_ID"] == row["Bout_ID"]].index
                if len(bout_frames) > 0:
                    self.csv_df_combined.loc[bout_frames, f"Current_{prop}"] = row[prop]
        for window in windows:
            if window > self.fps:
                bout_starts = (self.csv_df_combined["Active_swimming"] == 1) & (self.csv_df_combined["Active_swimming"].shift(1) == 0)
                self.csv_df_combined[f"Bout_frequency_{window}"] = bout_starts.rolling(window, min_periods=1).sum() * self.fps / window

    def calc_tail_beat_features(self):
        """
        Calculate tail beat frequency, amplitude, secondary frequency, and spectral entropy 
        using the movement signals from tail extra body parts.
        """
        import scipy.signal as signal
        tail_cols = [bp for bp in TAIL_EXTRA_BP_NAMES if bp in self.bp_names]
        if not tail_cols:
            return
        tail_movement_cols = [f"{bp}_movement" for bp in tail_cols if f"{bp}_movement" in self.csv_df_combined.columns]
        if not tail_movement_cols:
            return
        self.csv_df_combined["Tail_movement_avg"] = self.csv_df_combined[tail_movement_cols].mean(axis=1)
        tail_signal = self.csv_df_combined["Tail_movement_avg"].values
        peaks, properties = signal.find_peaks(tail_signal, height=np.mean(tail_signal))
        if len(peaks) > 1:
            tail_beat_freq = (len(peaks) / len(tail_signal)) * self.fps
            self.csv_df_combined["Tail_beat_frequency"] = tail_beat_freq
            troughs, _ = signal.find_peaks(-tail_signal)
            if len(troughs) > 0:
                peak_values = tail_signal[peaks]
                trough_values = tail_signal[troughs]
                tail_beat_amplitude = np.mean(peak_values) - np.mean(trough_values)
                self.csv_df_combined["Tail_beat_amplitude"] = tail_beat_amplitude
            else:
                self.csv_df_combined["Tail_beat_amplitude"] = 0
        else:
            self.csv_df_combined["Tail_beat_frequency"] = 0
            self.csv_df_combined["Tail_beat_amplitude"] = 0

        if len(tail_signal) > self.fps * 2:
            freqs, psd = signal.welch(tail_signal, fs=self.fps, nperseg=min(256, len(tail_signal)//2))
            if len(psd) > 1:
                dominant_idx = np.argmax(psd[1:]) + 1
                dominant_freq = freqs[dominant_idx]
                dominant_power = psd[dominant_idx]
                self.csv_df_combined["Tail_dominant_freq"] = dominant_freq
                self.csv_df_combined["Tail_dominant_freq_power"] = dominant_power
                psd_copy = psd.copy()
                psd_copy[dominant_idx] = 0
                secondary_idx = np.argmax(psd_copy[1:]) + 1
                secondary_freq = freqs[secondary_idx]
                secondary_power = psd[secondary_idx]
                self.csv_df_combined["Tail_secondary_freq"] = secondary_freq
                self.csv_df_combined["Tail_secondary_freq_power"] = secondary_power
                psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
                self.csv_df_combined["Tail_spectral_entropy"] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    def calc_inter_body_coordination(self):
        """
        Compute inter-body-part coordination features (e.g., rolling correlation between head and tail movements).
        """
        head_cols = [bp for bp in HEAD_BP_NAMES if bp in self.bp_names]
        tail_cols = [bp for bp in TAIL_BP_NAMES if bp in self.bp_names]
        if not head_cols or not tail_cols:
            return
        head_movement_cols = [f"{bp}_movement" for bp in head_cols if f"{bp}_movement" in self.csv_df_combined.columns]
        tail_movement_cols = [f"{bp}_movement" for bp in tail_cols if f"{bp}_movement" in self.csv_df_combined.columns]
        if not head_movement_cols or not tail_movement_cols:
            return
        self.csv_df_combined["Head_movement_avg"] = self.csv_df_combined[head_movement_cols].mean(axis=1)
        self.csv_df_combined["Tail_movement_avg_for_coord"] = self.csv_df_combined[tail_movement_cols].mean(axis=1)
        window_size = max(2, int(self.fps / 2))
        self.csv_df_combined["Head_Tail_corr"] = self.csv_df_combined["Head_movement_avg"].rolling(window=window_size, min_periods=1).corr(self.csv_df_combined["Tail_movement_avg_for_coord"])

    def calc_jerk_and_angular_acceleration(self):
        """
        Calculate jerk (third derivative of position) for the center body part and angular acceleration.
        """
        center_bp = CENTER_BP_NAMES[0] if CENTER_BP_NAMES else None
        if center_bp and f"{center_bp}_velocity" in self.csv_df_combined.columns:
            self.csv_df_combined["Center_acceleration"] = self.csv_df_combined[f"{center_bp}_velocity"].diff().fillna(0)
            self.csv_df_combined["Center_jerk"] = self.csv_df_combined["Center_acceleration"].diff().fillna(0)
        if "Turning_rate" in self.csv_df_combined.columns:
            self.csv_df_combined["Angular_acceleration"] = self.csv_df_combined["Turning_rate"].diff().fillna(0)

    def calc_frequency_domain_features(self):
        """
        Compute additional frequency domain features (dominant frequency, secondary frequency, spectral entropy)
        for the center movement signal.
        """
        center_bp = CENTER_BP_NAMES[0] if CENTER_BP_NAMES else None
        if center_bp and f"{center_bp}_movement" in self.csv_df_combined.columns:
            center_signal = self.csv_df_combined[f"{center_bp}_movement"].values
            import scipy.signal as signal
            if len(center_signal) > self.fps * 2:
                freqs, psd = signal.welch(center_signal, fs=self.fps, nperseg=min(256, len(center_signal)//2))
                if len(psd) > 1:
                    dominant_idx = np.argmax(psd[1:]) + 1
                    self.csv_df_combined["Center_dominant_freq"] = freqs[dominant_idx]
                    self.csv_df_combined["Center_dominant_freq_power"] = psd[dominant_idx]
                    psd_copy = psd.copy()
                    psd_copy[dominant_idx] = 0
                    secondary_idx = np.argmax(psd_copy[1:]) + 1
                    self.csv_df_combined["Center_secondary_freq"] = freqs[secondary_idx]
                    self.csv_df_combined["Center_secondary_freq_power"] = psd[secondary_idx]
                    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
                    self.csv_df_combined["Center_spectral_entropy"] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    def calc_lateral_symmetry(self):
        """
        Compute lateral symmetry metrics based on the x-coordinates of all body parts.
        """
        x_columns = [f"{bp}_x" for bp in self.bp_names if f"{bp}_x" in self.csv_df_combined.columns]
        if not x_columns:
            return
        self.csv_df_combined["Lateral_width"] = self.csv_df_combined[x_columns].max(axis=1) - self.csv_df_combined[x_columns].min(axis=1)
        window_size = max(2, int(self.fps / 2))
        self.csv_df_combined["Lateral_width_std"] = self.csv_df_combined["Lateral_width"].rolling(window=window_size, min_periods=1).std()

    def calc_fractal_dimension(self):
        """
        Estimate the fractal dimension of the center trajectory using a box-counting method.
        This global measure is added as a constant feature.
        """
        center_bp = CENTER_BP_NAMES[0] if CENTER_BP_NAMES else None
        if center_bp and f"{center_bp}_x" in self.csv_df_combined.columns and f"{center_bp}_y" in self.csv_df_combined.columns:
            x = self.csv_df_combined[f"{center_bp}_x"].values
            y = self.csv_df_combined[f"{center_bp}_y"].values
            coords = np.vstack((x, y)).T

            def boxcount(Z, k):
                S = np.add.reduceat(
                    np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
                return len(np.where((S > 0) & (S < k * k))[0])

            H, xedges, yedges = np.histogram2d(x, y, bins=64)
            sizes = np.logspace(0, np.log10(max(2, H.shape[0])), num=8, endpoint=True, base=10).astype(int)
            sizes = np.unique(sizes)
            counts = []
            for size in sizes:
                counts.append(boxcount(H > 0, max(1, size)))
            if len(sizes) > 1 and np.all(np.array(counts) > 0):
                coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
                fractal_dim = -coeffs[0]
            else:
                fractal_dim = 0
            self.csv_df_combined["Center_fractal_dimension"] = fractal_dim

    def calc_bout_transition_stats(self):
        """
        Calculate bout transition statistics such as the number of transitions between active and inactive swimming.
        """
        if "Active_swimming" in self.csv_df_combined.columns:
            transitions = self.csv_df_combined["Active_swimming"].diff().fillna(0).abs()
            bout_transition_count = transitions.sum()
            self.csv_df_combined["Bout_transition_count"] = bout_transition_count

    def calc_spatial_occupancy(self):
        """
        Compute spatial occupancy metrics based on the center body part.
        """
        center_bp = CENTER_BP_NAMES[0] if CENTER_BP_NAMES else None
        if center_bp and f"{center_bp}_x" in self.csv_df_combined.columns and f"{center_bp}_y" in self.csv_df_combined.columns:
            x = self.csv_df_combined[f"{center_bp}_x"]
            y = self.csv_df_combined[f"{center_bp}_y"]
            start_x = x.iloc[0]
            start_y = y.iloc[0]
            self.csv_df_combined["Center_displacement"] = np.sqrt((x - start_x)**2 + (y - start_y)**2) / self.px_per_mm
            window_size = max(2, int(self.fps / 2))
            self.csv_df_combined["Center_x_variance"] = x.rolling(window=window_size, min_periods=1).var()
            self.csv_df_combined["Center_y_variance"] = y.rolling(window=window_size, min_periods=1).var()

    # ---------------- Save ----------------
    def save_file(self):
        # Drop helper columns but be forgiving if they are missing
        self.csv_df_combined = self.csv_df_combined.drop(self.col_headers_shifted, axis=1, errors='ignore')
        self.csv_df_combined = self.csv_df_combined.drop(
            [
                "Compass_digit_shifted",
                "Direction_switch",
                "Switch_direction_value",
                "Compass_digit",
                "Compass_direction",
                "Angle_sin_cumsum",
                "Angle_cos_cumsum",
            ],
            axis=1,
            errors='ignore'
        ).fillna(0)
        write_df(self.csv_df_combined.astype(np.float32), self.file_type, self.save_path)
