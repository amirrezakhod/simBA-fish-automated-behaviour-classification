<img width="960" alt="image" src="https://github.com/user-attachments/assets/e4abd305-e54e-4c5c-8bc6-e44844e09846" />
In this demo we assume that the user is familiar with the usual workflow in simba and we are more focused on behaviour classification in fish using more customized workflow tailored to fish. 
In  this demo, we will go over a simple behaviour classification with simba using pose estimation data from lionfish analysez by deeplabcut. 
to run this demo, you only need to have simba as the pose estimation data is provided. 
we will use the two training videos(Hunting_demo/Training_video) to train the model and test its ferfurmance on the test video(Hunting_demo/Testing_videos). 
To better understand how each behaviour looks like refer to the [behaviour guid](Behaviour_guide)
create your project in simba and use the following landmarks to crate your custome pose config 
![Landmark positions](https://raw.githubusercontent.com/amirrezakhod/simBA-fish-automated-behaviour-classification/main/feature-catalog/Landmark_positions.jpg)
