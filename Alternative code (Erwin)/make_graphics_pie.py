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

sizes = []
for target in targets:
    sizes.append(len(original_data[original_data.Target == target]))

print sizes
# Creating a figure and axes
sizes = np.divide(sizes, float(len(original_data)))
print sizes
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'red']
explode = (0, 0.1, 0, 0, 0)

patches, texts, autotexts= plt.pie(sizes, explode=explode, labels=targets, colors=colors, shadow=True, startangle=90, autopct='%1.1f%%')

for t in texts:
    t.set_fontsize(16)
for t in autotexts:
    t.set_fontsize(20)
plt.axis('equal')
#fig = plt.figure()
#ax = fig.gca()

plt.show()
