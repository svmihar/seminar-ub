from sklearn import datasets
import seaborn as sns; sns.set(style='ticks',color_codes=True)
import pandas as pd
import matplotlib.pyplot as plt

# data = datasets.load_iris()
# df = pd.DataFrame(data.data, columns=data.feature_names)
# df['target'] = data.target
df = sns.load_dataset('iris')
sns.pairplot(df,kind='reg',  hue='species',palette="husl",markers=['*','o','s'])
plt.show()
