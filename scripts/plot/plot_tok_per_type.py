#Script written by Aleksandr Schamberger (GitHub: https://github.com/a-leks-icon) as part of the introductory course by Roland Meyer 'Einführung in die Computerlinguistik (mit Anwendung auf Slawische Sprachen)' at the Humboldt Universität zu Berlin in the winter semester 2023/24.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Name of the .csv-file containing the results of the tagger.
csv_file = "../../results/datasets_statistics.csv"
#Turn the .csv-file's data into a pandas data frame.
df = pd.read_csv(csv_file,header=0,sep=";")
#Arange the data into lists.
datasets = ["Urum Words", "Urum Morphs", "Brown Small", "Brown Medium", "Brown Words"]
tok_freq = df["tokens"]
type_freq = df["types"]
#Compute the number of tokens per types per dataset.
tok_per_type = [round(tok_freq[i]/type_freq[i],2) for i in range(len(tok_freq))]
#Changing the font size of the plots.
font_size = 14
plt.rc("font",size=font_size)
#Create a figure and axes (plot) object.
fig,ax = plt.subplots()
#Create the bars.
ax.bar(datasets,tok_per_type,color="b",align="center")
#Set the title of the figure and of the axes (x and y).
ax.set_title("Number of tokens per types per dataset")
ax.set_xlabel("Datasets")
ax.set_ylabel("Number of tokens per types")
#Set the limits for values on the y-axis.
ax.set_ylim(top=15,bottom=0)
#Add the value of every bar on top of it.
for i in range(len(datasets)):
    ax.text(i,tok_per_type[i]+0.25,str(tok_per_type[i]),ha="center",va="bottom")
#Print the plot.
plt.tight_layout()
plt.show()
