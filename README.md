# Machine Learning Project with Scikit-Learn

Based on this Kaggle notebook: [Scikit-Learn Project to Start Machine Learning](https://www.kaggle.com/code/kelde9/scikit-learn-project-to-start-machine-learning).

## Installation

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

python -m pip install -r requirements.txt

⚠️ Note on correlation plot ⚠️
The original notebook has an error in this code:

plt.figure(figsize=(15,2))
sns.heatmap(df.corr().iloc[[-1]],
            cmap='RdBu_r',
            annot=True,
            vmin=-1, vmax=1)
plt.show()

df.corr() works only with numeric columns, but the DataFrame includes a string column (alcohol_category), causing:

ValueError: could not convert string to float: 'Low'

To fix it, select only numeric columns before correlation:

df_numeric = df.select_dtypes(include=[np.number])
plt.figure(figsize=(15,2))
sns.heatmap(df_numeric.corr().iloc[[-1]],
            cmap='RdBu_r',
            annot=True,
            vmin=-1, vmax=1)
plt.show()


⚠️ Model Predictions May Differ from the Original Notebook ⚠️
If you run this notebook, you might notice that your results are slightly different from the original Kaggle version.
In this project, we use Logistic Regression to predict the wine quality based on chemical features.

When running:
logisticRegression.predict_proba(X_test.iloc[:1])
We get the following probabilities:
Class (Quality)	Probability
3	1.03%
4	0.46%
5	15.27%
6	48.27%
7	29.33%
8	5.61%
9	~0%

So, the model predicts class 6, since it has the highest probability.

However, when we check the actual label in the test set:
y_test.iloc[:1]
1784    7

Why does this happen?
-Machine Learning models are probabilistic. Predict the most likely class based on training data, but they can still be wrong, especially when classes are similar.

-Wine qualities like 6 and 7 are often hard to distinguish using simple models like Logistic Regression because their features overlap.

-Different random splits of the dataset can produce different results. The Kaggle notebook you followed may have generated different predictions because the train/test split was randomized.

-Your dataset may have different distributions or sampling compared to the original notebook, which naturally leads to different outputs.

✅ Takeaway
Machine Learning models do not always get the exact answer right, especially when working with real-world data and probabilistic outputs.

The important thing is to analyze the predictions, understand the model's confidence, and evaluate performance globally (e.g., using accuracy, precision, recall), not just for individual samples.