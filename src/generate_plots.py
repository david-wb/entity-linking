import matplotlib.pyplot as plt
import numpy as np

# data to plot
n_groups = 4
retrieval_rate_bert = (1, 10, 40, 60)
retrieval_rate_roberta = (1.5, 15, 45, 65)
retrieval_rate_declutr = (2, 20, 50, 80)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8

rects1 = plt.bar(index, retrieval_rate_bert, bar_width,
                 alpha=opacity,
                 color='b',
                 label='BERT Base Uncased')

rects2 = plt.bar(index + bar_width, retrieval_rate_roberta, bar_width,
                 alpha=opacity,
                 color='g',
                 label='RoBERTa Base')

rects3 = plt.bar(index + 2 * bar_width, retrieval_rate_declutr, bar_width,
                 alpha=opacity,
                 color='r',
                 label='DeCLUTR')

plt.xlabel('k = Number of Candidates')
plt.ylabel('Retrieval Rate')
plt.title('Retrieval Rate by Base Encoder')
plt.xticks(index + bar_width, ('k = 1', 'k = 16', 'k = 32', 'k = 64'))
plt.legend()

plt.tight_layout()
plt.show()
