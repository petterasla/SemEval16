import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.style.use('ggplot')

#******* Processing data *********
original_data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t', index_col=0)
atheims = original_data[original_data.Target == "Atheism"]
t =[len(atheims[atheims.Stance == "AGAINST"].Stance), len(atheims[atheims.Stance == "FAVOR"].Stance), len(atheims[atheims.Stance == "NONE"].Stance)]
targets = list(original_data.Target.unique())

against, favor, none = [], [], []
for target in targets:
    data = original_data[original_data.Target == target]
    against.append(len(data[data.Stance == "AGAINST"]))
    favor.append(len(data[data.Stance == "FAVOR"]))
    none.append(len(data[data.Stance == "NONE"]))

# Creating a figure with two plots on one row
fig, (ax1, ax2) = plt.subplots(1,2, sharex=False, sharey=False)

# Setting some basic parameters
bar_width = 0.3
opacity = 0.8
index = np.arange(len(targets))*2
i=0
colors = ['crimson', 'burlywood', 'chartreuse']

# Subplot 1:
# Settings bars for against, favor and none with parameters
ax1.bar(index + (bar_width*2), none, align="center", width=bar_width, color=colors[2], label="None", alpha=opacity)
ax1.bar(index + bar_width, favor, align="center", width=bar_width, color=colors[1], label="Favor", alpha=opacity)
ax1.bar(index, against, align="center", width=bar_width, color=colors[0], label="Against", alpha=opacity)

# X axis:
ax1.set_xlabel("Targets")
ax1.set_xticks(index + bar_width)
ax1.set_xticklabels(targets, rotation=45, ha="right")

# Y axis
ax1.set_ylabel("Number of instances")
ax1.set_yticks(np.arange(0,500, 50))
ax1.set_ylim((0,450))

# Other shizzle
ax1.legend()        # Display what each color refer to in top right corner
ax1.set_title("Bar representation of individual stances")

# Subplot 2: Settings bars for against, favor and none with parameters
bar_width = 1.0

ax2.bar(index, none, align="center", width=bar_width, color=colors[2], label="None", alpha=opacity)
ax2.bar(index, favor, bottom=none, align="center", width=bar_width, color=colors[1], label="Favor", alpha=opacity)
ax2.bar(index, against, bottom=np.add(favor,none), align="center", width=bar_width, color=colors[0], label="Against", alpha=opacity)

# X axis:
ax2.set_xlabel("Targets")
ax2.set_xticks(index)
ax2.set_xticklabels(targets, rotation=45, ha="right")

# Y axis
ax2.set_ylabel("Number of instances")
ax2.set_yticks(np.arange(0,900, 100))
ax2.set_ylim((0,800))

# Other shizzle
ax2.legend()        # Display what each color is referencing
ax2.set_title("Stacked bar representation of the stances")

# Print that shit
plt.subplots_adjust(hspace=50)
plt.show()
#fig, ax = plt.subplots()
#fig.set_size_inches((15,10))
#means.columns = clf_names
#errors.columns = clf_names
#ax.set_ylim([0,1.0])
#ax.set_ylabel('Macro F')
#means.plot(yerr=errors, ax=ax, kind='bar')