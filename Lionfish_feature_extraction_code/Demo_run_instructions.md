## Lionfish Feature-Extraction Demo (SimBA)

This short README walks you through running our custom feature-extraction script on an 11-second lionfish video that has already been tracked with DeepLabCut.

1. Install SimBA
Follow the official installation guide at https://github.com/sgoldenlab/simba.

2. Open the project
Change your working directory to the folder that contains project_folder/, then launch SimBA and load the configuration file:
Lionfish_feature_extraction_code/Simba_config/project_folder/project_config.ini

3. Select the custom extractor
In SimBA’s Extract Features menu choose Apply user-defined feature extraction, then point SimBA to:
Lionfish_feature_extraction_code/Feature_extraction_file/lionfish_feature_extraction.py

4. Run extraction
Start the extraction. The resulting CSV will be written to:
Lionfish_feature_extraction_code/Simba_config/project_folder/csv/features_extracted/
