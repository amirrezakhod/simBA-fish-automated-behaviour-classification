<img width="960" alt="image" src="https://github.com/user-attachments/assets/e4abd305-e54e-4c5c-8bc6-e44844e09846" />
In this demo we assume that the user is familiar with the usual workflow in simba and we are more focused on behaviour classification in fish using more customized workflow tailored to fish. 
In  this demo, we will go over a simple behaviour classification with simba using pose estimation data from lionfish analysez by deeplabcut. 
to run this demo, you only need to have simba as the pose estimation data is provided. 
we will use the two training videos(Hunting_demo/Training_video) to train the model and test its ferfurmance on the test video(Hunting_demo/Testing_videos). 
To better understand how each behaviour looks like refer to the [behaviour guid](Behaviour_guide)
create your project in simba and use the following landmarks to crate your custome pose config 
![Landmark positions](https://raw.githubusercontent.com/amirrezakhod/simBA-fish-automated-behaviour-classification/main/feature-catalog/Landmark_positions.jpg)
Smooth the csv files usin gaussian smoothing and set it at 10 sec and outlier correction(movement criteria of 2 and location critaria of1)  and this is how the pose estimation should look like after the smoothing and applying the smoothing and and outlier corrections: 
(Hunting_demo/Images_Gifs/16Aug24_Tank4_Dawn_pt2_1080p_corrected.gif)
we can define and roi and use it in our behaviour classififcation if a certain behaviour is happening in an specific area, here we know that hunting is mostly happening in the bottom of the tank and near the devider. so we define 2 ROIs near and bottom near:
(Hunting_demo/Images_Gifs/ROI_Hunt.png)
we can now run the feature extraction using our custom fish feature extractor(Lionfish_feature_extraction_code/FishFeatureExtractor_3.1.py) 
once you run the code you can see the feature selection window(Hunting_demo/Images_Gifs/Hunt_selected_features.png) it is a good habit to start by basic movements and that is what we are doing. we are jyst calculating the basic movements. you can find what exactly being calculated by refering to the feature catalog (feature-catalog)
after calculating the features and appending the roi data you can label the behaviours and train the model. use behaviour guide( Behaviour_guide) to better understand how huntiong looks like. model setting ( Hunting_demo/Images_Gifs/Hunting_0_meta.csv)

We can now run the model made on the unseen data( Hunting_demo/Testing_videos) 
interactive probability plot in simba is a grate tool to visialize and better understand the perfurmance of your classifier. 
here we can see that our model is detecting the hunting behaviour using only basic movements calculated from the pose data: 
(Hunting_demo/Images_Gifs/Hunt_prob1.png)
(Hunting_demo/Images_Gifs/Hunt_prob2.png)

# Lionfish Behaviour Classification (SimBA Demo)

This demo assumes you’re already comfortable with SimBA’s standard flow. We focus on a **fish-tailored** workflow for classifying **Hunting** in lionfish using provided DeepLabCut pose CSVs (no pose re-estimation needed).

---

## What you’ll use
- Training videos: [`Training_video/`](Training_video/)
- Test video: [`Testing_videos/`](Testing_videos/)
- Behaviour guide: [`../Behaviour_guide/`](../Behaviour_guide/)
- Feature catalog (feature definitions): [`../feature-catalog/`](../feature-catalog/)
- Custom fish feature extractor: [`../Lionfish_feature_extraction_code/FishFeatureExtractor_3.1.py`](../Lionfish_feature_extraction_code/FishFeatureExtractor_3.1.py)
- Example meta file: [`Images_Gifs/Hunting_0_meta.csv`](Images_Gifs/Hunting_0_meta.csv)

---

## 1) Create a SimBA project
Create a new SimBA project, add the **two training videos** from `Training_video/` (with their DLC CSVs), and configure your **body-parts** to match the landmarks below.

**Landmark layout**  
![Landmark positions](https://raw.githubusercontent.com/amirrezakhod/simBA-fish-automated-behaviour-classification/main/feature-catalog/Landmark_positions.jpg)

---

## 2) Smooth & correct the pose
Apply to all imported pose CSVs:
- **Smoothing:** Gaussian, **10 s** window  
- **Outlier correction:** Movement criterion **2**, Location criterion **1**

**After smoothing + outlier correction (example):**  
![Corrected pose example](https://raw.githubusercontent.com/amirrezakhod/simBA-fish-automated-behaviour-classification/main/Hunting_demo/Images_Gifs/16Aug24_Tank4_Dawn_pt2_1080p_corrected.gif)

---

## 3) Define ROIs for Hunting
Hunting occurs mostly **near the tank bottom** and **near the divider**. Create two ROIs (name suggestions):
- `bottom_near`
- `near_divider`

**ROI layout (example):**  
![ROIs for hunting](https://raw.githubusercontent.com/amirrezakhod/simBA-fish-automated-behaviour-classification/main/Hunting_demo/Images_Gifs/ROI_Hunt.png)

---

## 4) Extract fish-specific features
Run the custom extractor:  
[`FishFeatureExtractor_3.1.py`](../Lionfish_feature_extraction_code/FishFeatureExtractor_3.1.py)

Start simple with **Basic Movements** only (good baseline, fast).  
Refer to the **feature catalog** for what each feature means: [`../feature-catalog/`](../feature-catalog/)

**Feature selection dialog (example):**  
![Select basic movement features](https://raw.githubusercontent.com/amirrezakhod/simBA-fish-automated-behaviour-classification/main/Hunting_demo/Images_Gifs/Hunt_selected_features.png)

**Defined behaviours sheet (reference):**  
![Defined behaviours](https://raw.githubusercontent.com/amirrezakhod/simBA-fish-automated-behaviour-classification/main/Hunting_demo/Images_Gifs/Defined_behaviours.png)

---

## 5) Label behaviours & train the model
- Use the **Behaviour Guide** to label **Hunting** on the training videos: [`../Behaviour_guide/`](../Behaviour_guide/)  
- Train your classifier (e.g., Random Forest) using the extracted features **plus** ROI channels.

---

## 6) Evaluate on an unseen video
- Add the **test** video from [`Testing_videos/`](Testing_videos/)  
- Run the trained model on it  
- Inspect results with **Interactive Probability Plot** in SimBA

**Example probability plots (Hunting):**  
![Hunting probability 1](https://raw.githubusercontent.com/amirrezakhod/simBA-fish-automated-behaviour-classification/main/Hunting_demo/Images_Gifs/Hunt_prob1.png)  
![Hunting probability 2](https://raw.githubusercontent.com/amirrezakhod/simBA-fish-automated-behaviour-classification/main/Hunting_demo/Images_Gifs/Hunt_prob2.png)

> In this demo, **Hunting** is detected using **Basic Movement** features derived from pose, showing a simple fish-aware pipeline can perform well.

---

## Quick links (resources)
- Training videos: [`Training_video/`](Training_video/)  
- Test video: [`Testing_videos/`](Testing_videos/)  
- Behaviour guide: [`../Behaviour_guide/`](../Behaviour_guide/)  
- Feature catalog: [`../feature-catalog/`](../feature-catalog/)  
- Fish feature extractor: [`../Lionfish_feature_extraction_code/FishFeatureExtractor_3.1.py`](../Lionfish_feature_extraction_code/FishFeatureExtractor_3.1.py)

