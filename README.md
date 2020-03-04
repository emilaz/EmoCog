# OpenMindv2

This project, titled EmoCog, aims at decoding Happiness from human brain activity (ECoG data). It uses the emotion predictions from the OpenFace network developed by Gautham as labels and the corresponding ECoG data as features.

To get a better understanding of this project and for a high-level explanation of the intermediate steps taken, I recommend looking at the presentation at 
https://drive.google.com/drive/folders/1Y6sPEbpIcLzkYUigXTDULHD-OqJXotry
If you don't have access to the file, contact me at emil.azadian@gmail.com


The codebase can be considered as consisting of four main parts, which is reflected by the repo structure.
 1. Feature Related. Has:
	- Class to load the raw feature data into memory (FeatureDataHolder)
	- Class to preprocess the raw features (simple_edf_preprocessing)
	- Class to generate features (FeatureGenerator)
	2. Label Related. Has:
	- Class to load the labels into memory (LabelDataHolder)
	- Class to process the labels into usable format (LabelGenerator)
	3. DataProvider class. Since generating the features and labels take quite some time and require synchronizing, this class brings both sides together and is where the data for train and test should be generated and then saved to file.
	4. Classifications. This folder contains different ML approaches tried so far to solve the problem. It loads the data generated in the DataProvider class.
	5. Moreover, there are some helper classes found in Util.py which aids with feature and label processing as well as saving and loading data and results to disk. There is also a file with  visualization tools (Vis.py)
The repo includes comments for every method used.
	
To run the project code, simply run it in the conda environment provided with this repo. 

How to run the code:

If you want to generate new data: Go to DataProvider.ipynb. Set the hyperparameters the way you want  them. Then, generate the data (just execute the cells that are filled already).

Using already generated data: Go to one of the classification files. Because I always wanted to compare different methods for the same hyperparameters, I set the hyperparas in the DataUtil class and load them from here using the load_config method. The data is then loaded into memory and can be used with the ML method in question.



