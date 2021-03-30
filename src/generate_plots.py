import os

import matplotlib.pyplot as plt
import numpy as np

# data to plot
n_groups = 4
retrieval_rate_bert = (0.4992, 0.7839, 0.8262, 0.8643)
retrieval_rate_roberta = (0.4401, 0.7413, 0.7903, 0.8423)
retrieval_rate_declutr = (0.4606, 0.771, 0.8172, 0.8587)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8

rects1 = plt.bar(index, retrieval_rate_bert, bar_width,
                 alpha=opacity,
                 color='b',
                 label='BERT base uncased')

rects2 = plt.bar(index + bar_width, retrieval_rate_roberta, bar_width,
                 alpha=opacity,
                 color='g',
                 label='RoBERTa base')

rects3 = plt.bar(index + 2 * bar_width, retrieval_rate_declutr, bar_width,
                 alpha=opacity,
                 color='r',
                 label='DeCLUTR')

plt.xlabel('k = Number of Candidates')
plt.ylabel('Retrieval Rate')
plt.title('Retrieval Rates by Base Encoder')
plt.xticks(index + bar_width, ('k = 1', 'k = 16', 'k = 32', 'k = 64'))
plt.legend()
plt.ylim(0, 1)
plt.tight_layout()
plot_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../static/retrieval_rates_bar_chart.png')
if os.path.isfile(plot_file_path):
    os.remove(plot_file_path)
plt.savefig(plot_file_path)
plt.draw()
plt.waitforbuttonpress(0)
plt.close()
