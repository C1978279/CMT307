IMDbAnalyse.py

System Requirements
This python script requires a python environment running at least python 3.6. Numpy, SKLearn and NLTK are dependent packages, and internet access is required to download some components of NLTK. For example, execution in an ipython environment would require the command "%run IMDbAnalyse.py". The IMDbAnalyse.py file should be located in the same directory as the accompanying datasets directory.

Overview
The python script uses the accompanying IMDb dataset to perform sentiment analysis using an SVM binary Classifier. Occasional command line print statements are made by the code to inform the user of progress and to return results. The final results of the script are returned in a format for easy inclusion in a LaTeX table.