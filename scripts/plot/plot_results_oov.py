#Script written by Aleksandr Schamberger (GitHub: https://github.com/a-leks-icon) as part of the introductory course by Roland Meyer 'Einführung in die Computerlinguistik (mit Anwendung auf Slawische Sprachen)' at the Humboldt Universität zu Berlin in the winter semester 2023/24.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Name of the .csv-file containing the results of the tagger.
csv_file = "../../results/accuracy_and_time.csv"
#Turn the .csv-file's data into a pandas data frame.
df = pd.read_csv(csv_file,header=0,sep=";")
#Arange the data into lists.
datasets = ["Urum Words", "Urum Morphs", "Brown Small", "Brown Medium", "Brown Words"]
oovs = df["oov"]
#Changing the font size of the plots.
font_size = 14
plt.rc("font",size=font_size)
#Create a figure and axes (plot) object.
fig,ax = plt.subplots()
#Create the bars.
ax.bar(datasets,oovs,color="b",align="center")
#Set the title of the figure and of the axes (x and y).
ax.set_title("Average number of word tokens not present in the training data\nnormalized by the number of tokens in the test data in %")
ax.set_xlabel("Datasets")
ax.set_ylabel("Average number of out of vocabulary words in %")
#Set the limits for values on the y-axis.
ax.set_ylim(top=100,bottom=0)
#Add the value (percentage) of every bar on top of it.
for i in range(len(datasets)):
    ax.text(i,oovs[i]+1,str(df["oov"][i]),ha="center",va="bottom")
#Print the plot.
plt.tight_layout()
plt.show()
