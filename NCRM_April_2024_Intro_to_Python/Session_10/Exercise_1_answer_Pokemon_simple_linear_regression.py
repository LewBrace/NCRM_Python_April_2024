import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/Filepath_to_file/pokemon_dataset.csv", header=0, index_col=0, encoding="ISO-8859-1")

x_con = sm.add_constant(df['Attack']) # adding a constant
lm = sm.OLS(df['Sp. Atk'],x_con).fit()

predictions = lm.predict(x_con)
print(predictions)

print(lm.summary())

sns.scatterplot(x=df['Attack'], y=df['Sp. Atk'])
#plotting the regression line.
sns.lineplot(x=df['Attack'],y=predictions, color='red')
plt.show()