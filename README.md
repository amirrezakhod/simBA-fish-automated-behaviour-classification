

## Fish Feature Extraction Guide for SimBA
<img width="960" alt="image" src="https://github.com/user-attachments/assets/e4abd305-e54e-4c5c-8bc6-e44844e09846" />

Feature extraction converts raw pose outputs (e.g., from DeepLabCut (DLC))—joint coordinates and detection likelihoods—into interpretable kinematic and postural descriptors such as tail-beat frequency, movement irregularity (e.g., path tortuosity or spectral entropy) over defined time windows. Because recording conditions (lighting, lens, viewpoint, frame rate) and species behaviours vary widely, maintaining separate extractor scripts per experiment is not scalable or reproducible. We propose a single, configurable extractor that exposes a broad library of fish-relevant features; users enable only the metrics that match their species and behavioural question, without modifying code and by simply adding their desired features to the feature inclusion/exclusion list defined in the feature extractor file. A decision guide maps common scenarios to features—for example, pursuit/hunting → tail-beat frequency, burst acceleration, heading-change rate.
This feature extraction code is specifically designed to support the classification of behaviours in mid-bodied fishes, such as lionfish, using the SimBA (Simple Behavioral Analysis) platform.
---

## 📖 Available Guides

- ✅ [Feature Inclusion Guide](Lionfish_feature_extraction_code/feature_inclusion_guide.md)
  learn how to include or exclude specific features in your pipeline.

- 🎯 [Feature Usage Recommendations](https://github.com/amirrezakhod/Fish-simba-features/blob/main/Lionfish_feature_extraction_code/Feature%20Usage%20Guide.md)
 Find out which features are best for detecting your specific behaviours.

- ▶️ [Demo Execution Instructions](Lionfish_feature_extraction_code/Demo_run_instructions.md)
  Run a demo feature extraction with DeepLabCut CSV.
  


![06242-ezgif com-gif-to-mp4-converter](https://github.com/user-attachments/assets/29496d29-b716-4d4d-bf5a-27abfd7d3932)



H=Hovering, T=Attack, S=Swimming, R=Resting

