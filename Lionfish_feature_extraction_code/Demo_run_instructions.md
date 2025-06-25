## 🐟Lionfish Feature-Extraction Demo (SimBA)

This short guide walks you through running our custom feature extraction script on an 11-second lionfish video already tracked with DeepLabCut.
---

⚙️  Instructions

1. Install [SimBA](https://github.com/sgoldenlab/simba/tree/master)

Follow the official SimBA installation guide.

2. Download [Required Files](https://github.com/amirrezakhod/Fish-simba-features/tree/main/Lionfish_feature_extraction_code) which contains: Feature_extraction_file and Simba_config/project_folder


3. Open the Project in SimBA

Set your working directory to the folder containing project_folder/. Launch SimBA and load the configuration file found in:
Lionfish_feature_extraction_code/Simba_config/project_folder/project_config.ini

4. Select the custom extractor
In SimBA’s Extract Features menu choose Apply user-defined feature extraction, then point SimBA to:
Lionfish_feature_extraction_code/Feature_extraction_file/lionfish_feature_extraction.py

5. Run extraction
Start the extraction. The resulting CSV will be written to:
Lionfish_feature_extraction_code/Simba_config/project_folder/csv/features_extracted/
