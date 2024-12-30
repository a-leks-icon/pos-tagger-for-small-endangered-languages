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
sents = list(df["sent_acc"])
tokens = list(df["tok_acc"])
sents_sd = list(df["sent_sd"])
tokens_sd = list(df["tok_sd"])
#Changing the font size of the plots.
font_size = 14
plt.rc("font",size=font_size)
#Create a figure and axes (plot) object.
fig,ax = plt.subplots()
#Create arrays representing the positions of the grouped bars.
#Define the width of every single bar in a grouped bar and
#it's factor (parameter furthermore influencing the bar width)
bar_width = 0.2
x_factor = 0.5
first_bar_xticks = np.array([i * x_factor for i in range(len(datasets))])
second_bar_xticks = np.array([num + bar_width for num in first_bar_xticks])
#Create the grouped bars by adding a single bar from a category to a
#specific position for all datasets and their values.
ax.bar(first_bar_xticks,tokens,width=bar_width,color="b",align="center",label="Tokens")
ax.bar(second_bar_xticks,sents,width=bar_width,color="r",align="center",label="Sentences")
#Center the ticks on the x-axis.
ax.set_xticks([(i * x_factor) + (bar_width/2) for i in range(len(datasets))])
#Label the ticks on the x-axis.
ax.set_xticklabels(datasets)
#Set the title of the figure and of the axes (x and y).
ax.set_title("Prediction accuracies in % for tokens and sentences per dataset\nwith standard deviation in white font")
ax.set_xlabel("Datasets")
ax.set_ylabel("Prediction accuracies in %")
#Set the limits for values on the y-axis.
ax.set_ylim(top=100,bottom=0)
#Add a legend.
ax.legend(loc="upper right")
#Add the value (percentage) of every bar on top of it.
for y,val in enumerate(sents):
    ax.text((y*x_factor)+bar_width,val+1,str(val)+"%",ha="center",va="bottom")
    ax.text((y*x_factor)+bar_width,val/2,str(sents_sd[y]),ha="center",va="bottom",color="white")
for y,val in enumerate(tokens):
    ax.text(y*x_factor,val+1,str(val)+"%",ha="center",va="bottom")
    ax.text(y*x_factor,val/2,str(tokens_sd[y]),ha="center",va="bottom",color="white")
#Print the plot.
plt.tight_layout()
plt.show()
