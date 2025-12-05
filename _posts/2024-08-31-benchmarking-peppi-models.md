---
layout: post
title: "Benchmarking 4 AI Models On A Custom PepPI Dataset"
description: Assembling four different ML/DL models to compare performance on identifying peptide-protein binding sites.
tags: ai bioinformatics
image: /img/seo/benchmarking-peppi-models.png
thumb: /img/thumb/benchmarking-peppi-models.webp
---
![alt text](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*qqkC_UlOZhi4ZuQYk62iUQ.png)
Ali & Foreman, the Red Sox & the Yankees, Romeo & Juliet. (Wait a minute…)

We love a good historical matchup, and in 2024, few are as important as the AI/LLM race. So, I wanted to throw my hat in the ring—and just so happened to have recently built a [binding site detection dataset](https://ai.gopubby.com/how-i-built-a-custom-protein-binding-dataset-for-machine-learning-in-5-steps-e456b8ab9341) for proteins that I could use as a benchmark.

The task? Given a protein structure, determine which parts of the sequence bind to other peptides.

I assembled four different AI challengers to see which performed the best:

- RandomForest
- XGBoost
- IsolationForest
- Convolutional Neural Network (CNN)

I’ll explain why this matters, why these aren’t language models — and, of course, the winners and losers.

---
## The Background


![Google Trend data for ‘ChatGPT’ — Screenshot by Author](https://miro.medium.com/v2/resize:fit:2000/format:webp/1*x7j4jjS51pj4Eta516gu8g.png)
*Google Trend data for ‘ChatGPT’*

Current hype is around language models — generative AI that can ‘read’ and write text. And the numbers speak for themselves:

- [ChatGPT](https://www.researchgate.net/publication/372212988_Generative_AI_for_Software_Practitioners) reached 100 million users in just two months.
- GPU manufacturer [NVIDIA](https://companiesmarketcap.com/cad/nvidia/marketcap/) is now worth more than the GDP of Canada (my personal home, True North, strong and free).
- It’s estimated AI will add [21% to US GDP](https://www.forbes.com/advisor/business/ai-statistics/) by 2030.

LLMs are going to solve countless issues — **but not all of them**.

**Picking the right AI model for your task is crucial.** And with LLMs, you might just be hiring an elephant to catch a mouse.

**Example**: I do bioinformatics, so I work with biological data. Some of that data is tabular or graphic — meaning LLMs could be useful for analysis, but may **require a disproportionate amount of computing power** for training.

As a classification/detection task, the **compute-to-accuracy ratio for this problem makes a lot more sense if we use smaller models** — like decision trees or CNNs.

### The Dataset

The [PepBDB-ML](https://github.com/MurtoHilali/PepBDB-ML) dataset is derived from the [PepBDB](http://huanglab.phys.hust.edu.cn/pepbdb/) database, a curated collection of peptide-protein complexes from the RCSB [PDB](https://www.rcsb.org/).

![PepBDB screenshot](https://miro.medium.com/v2/resize:fit:2000/format:webp/1*L2atpQGl4v6qWdoRwTN5Xg.png)
*PepBDB, Screenshot*

The dataset is inspired by a paper from [Wardah et al.](https://www.sciencedirect.com/science/article/abs/pii/S0022519320301338?via%3Dihub=), which collected residue-level data on protein sequence windows and turned them into images, using a CNN to classify them as binding or nonbinding.

I generated my own version using structural features from the PepBDB:

![Wardah inspired training data](https://miro.medium.com/v2/resize:fit:2000/format:webp/0*KF4gsrEBJCCTVz4v)

In the visualization above, columns are for amino acids, and the rows represent features like hydrophobicity or accessible surface area.

On the way to creating images like these, however, we end up with tabular data. (It looks like [this](https://gist.github.com/MurtoHilali/3e22448220806a01b608cfa89ab9a7c0#file-peppi_data-csv)).


This type of data is perfect for training a decision-tree style ML model, which explains my choice of contenders.

Each row (and image) is labeled as either binding or nonbinding. Our challengers’ task is to correctly classify each one.

Who are the challengers, you ask? Before we meet our fighters, let’s talk about the metrics we’re grading them on:

### Accuracy
Accuracy is given by the formula:

`(true positives + true negatives) / total number of data points.`

In other words, **how often does the model correctly classify the data?**

It’s crucial to remember that in the context of a peptide-protein binding site classification problem, accuracy isn’t the best metric.

In a given dataset, it’s possible only [5% of the residues are binding](https://www.sciencedirect.com/science/article/pii/S0022519320301338?via%3Dihub=) — that means an algorithm predicting every single residue as non-binding will have a 95% accuracy rate. Highly accurate, minimally useful.

### Precision
Precision is given by the formula:

`true positives / (true positives + false positives)`

This measures how many of the **positive predictions** made by the model are **actually correct**.

### Recall
Recall is given by the formula:

`true positives / (true positives + false negatives).`

This measures how many of the actual positive labels in the data are being correctly identified by the model. **If it’s low, it means we’re missing several binding sites.**

### F1 Score
The F1 score combines precision and recall into one metric, given by the formula:

`2 * ((precision * recall) / (precision + recall)).`

It is a better measure than accuracy, especially when dealing with imbalanced datasets.

We’re going to keep a special eye out for recall: since binding sites are relatively rare compared to non-binding sites, we want to make sure our algorithm has a high sensitivity for detecting them.

>(Note: ‘sensitivity’ and ‘recall’ [mean the same thing](https://www.evidentlyai.com/classification-metrics/accuracy-precision-recall#:~:text=Recall%20can%20also%20be%20called,research%20rather%20than%20machine%20learning.)! Sensitivity is more commonly used in biological contexts).

**Housekeeping**: for the four models below, I split PepBDB-ML into a training, testing, and validation set for the final metrics. The first three models were also hyperparameter-tuned, while I mostly relied on the established hyperparameters from the paper for the CNN.

Now, let’s meet our challengers!

## The Challengers
### Challenger #1: Rumblin’ RandomForest
Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*MYUXecdY2zlyVqUltVQCJg.png)

- **Type**: Ensemble of decision trees.
- **Strengths**: Relatively simple, highly observable, flexible, and typically robust against overfitting.
- **Weaknesses**: It can be computationally expensive for many feature datasets, and it has limited accuracy relative to more sophisticated models.

So how’d it do?

- **Accuracy**: 86.12%
- **Precision**: 49.36%
- **Recall**: 57.23%
- **F1-Score**: 53.01%
Exceedingly average! Here’s a graphical overview:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Q6XxekcePKKUBnao81GQ6w.png)
*Confusion matrix for the RandomForest classifier’s performance | By Author*

As you can see, the high accuracy is largely thanks to the high volume of true positives.

Here’s the code for you to play around with yourself:

```## randomforest.py

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import joblib
from scipy.stats import randint
import matplotlib.pyplot as plt


from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

# Random Forest Classification

## 1. Open up the CSV file and split into features + targets

print('Loading in data...')
pepbdb = pd.read_csv('/path/to/peppi_data.csv')
print('\033[1mLoaded.\033[0m')

# One-hot encode the 'AA' column
pepbdb = pd.get_dummies(pepbdb, columns=['AA'])

X = pepbdb.drop('Binding Indices', axis=1)
y = pepbdb['Binding Indices']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## 2. Hyperparameter tuning
print('Starting hyperparameter tuning...')
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(n_jobs=-1, random_state=42, class_weight='balanced')
rf_random = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=param_dist, 
    n_iter=100, 
    cv=3, 
    verbose=2, 
    random_state=42, 
    n_jobs=-1,
    scoring='recall'
    )
rf_random.fit(X_train, y_train)

print('Best hyperparameters found:')
print(rf_random.best_params_)

best_rf = rf_random.best_estimator_

print('Testing accuracy with the best model...')
y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

with open('/path/to/randomforest/summary_randomForest.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}')


model_path = '/path/to/randomforest/random_forest_model_best.pkl'
joblib.dump(best_rf, model_path)

print('Visualizing...')
for i in range(3):
    tree = best_rf.estimators_[i]
    dot_data = export_graphviz(
        tree,
        feature_names=X_train.columns,
        filled=True,
        max_depth=3,
        impurity=False,
        proportion=True
    )
    graph = graphviz.Source(dot_data)
    
    file = f'/path/to/randomforest/tree_{i}'
    graph.render(file, format='png', cleanup=True)

print("Graphs saved as PNG images.")
print('\033[1mCompleted.\033[0m')
```
As you can see in the code above, one of the coolest parts of RandomForest is seeing the trees themselves:

![](https://miro.medium.com/v2/resize:fit:2000/format:webp/1*TvJg-O1UWYySbc2Ar9PFMA.png)
*Decision tree from RandomForest ensemble | By Author*

### Challenger #2: The Exceptional XGBoost
![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*4hetaWoIbuX2CyXwQf8PBg.png)
- **Type**: Gradient-boosted ensemble decision trees.
- **Strengths**: Gradient boosting involves the sequential building of trees, where each tree corrects on the previous one leading to better predictive performance. Also scales well to large datasets.
- **Weaknesses**: It might overfit to noise as it builds sequentially, and again, it is limited compared to even more sophisticated algorithms.

XGBoost is an algorithm close to my heart, and it was one of my first forays into bio-ML. How’d it do?

- **Accuracy**: 76.54%
- **Precision**: 32.94%
- **Recall**: 69.02%
- **F1-Score**: 44.60%
💔

Heartbroken is an understatement. While our sensitivity (recall) has gone up, we’re still scoring pretty low on every other metric.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*HLozMicBaVDGG_MMBH9vGA.png)
*Confusion matrix for the XGBoost classifier’s performance | By Author*

Here’s the code for your own exploration:
```
## xgboostmodel.py

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint, uniform
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
print('Loading in data...')
pepbdb = pd.read_csv('/path/to/peppi_data.csv')
print('\033[1mLoaded.\033[0m')

# One-hot encode the 'AA' column
pepbdb = pd.get_dummies(pepbdb, columns=['AA'])

X = pepbdb.drop('Binding Indices', axis=1)
y = pepbdb['Binding Indices']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate class weight ratio
num_pos = sum(y_train)
num_neg = len(y_train) - num_pos
scale_pos_weight = num_neg / num_pos

# Hyperparameter tuning
print('Starting hyperparameter tuning...')
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 20),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5)
}

xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_jobs=-1, random_state=42, use_label_encoder=False, scale_pos_weight=scale_pos_weight)
xgb_random = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1, scoring='recall')
xgb_random.fit(X_train, y_train)

print('Best hyperparameters found:')
print(xgb_random.best_params_)

best_xgb = xgb_random.best_estimator_

print('Testing accuracy with the best model...')
y_pred = best_xgb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
fi_score = f1_score(y_test, y_pred)

with open('/path/to/xgboostmodel/summary_xgboost.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}')

# Save the model using XGBoost's save_model method
model_path = '/path/to/xgboostmodel/xgboost_model_best.json'
best_xgb.save_model(model_path)

print('\033[1mCompleted.\033[0m')
```
### Challenger #3: The Inimitatable IsolationForest
![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*fpm4iFYwDxrgDgRSBvp7RQ.png)
- **Type**: An anomaly detection algorithm also based on decision tree ensembles.
- **Strengths**: This architecture isolates anomalies, which could be useful since the rarity of binding sites may allow us to treat them as anomalies.
- **Weaknesses**: The algorithm assumes isolated anomalies — realistically, our anomalies (i.e. the binding sites) tend to be related.

Despite the weaknesses, I thought it might be interesting to test how a model like this performs when we treat binding sites as anomalies. It’s also an unsupervised model, meaning it **doesn't need labeled features** — useful for dirty data scenarios.

By setting the contamination parameter (the predicted rate of anomalies) as our binding site class weight — how’d it do?

- **Accuracy**: 78.74%
- **Precision**: 22.21%
- **Recall**: 22.16%
- **F1-Score**: 22.18%
Stunningly poor! This isn’t very surprising given what we know about the nature of binding sites and the purpose of this model.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*_LVYEnTS34g4P7gv3Bba7g.png)
*Confusion matrix for the IsolationForest classifier’s performance | By Author*

Here’s the code for you to experiment with. IsoForest doesn’t have binary outputs, so we map them to 1 and 0 here.

(Note: this leads to an inflated custom F1 score that still needs debugging.)
```
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
import joblib

from scipy.stats import randint
import matplotlib.pyplot as plt

print('Loading in data...')
pepbdb = pd.read_csv('/path/to/peppi_data.csv')
print('\033[1mLoaded.\033[0m')

# One-hot encode the 'AA' column
pepbdb = pd.get_dummies(pepbdb, columns=['AA'])

X = pepbdb.drop('Binding Indices', axis=1)
y = pepbdb['Binding Indices']

# Calculate the proportion of binding residues (anomalies)
contamination = y.mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Starting hyperparameter tuning...')
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_samples': randint(100, 500),
    'max_features': randint(1, X.shape[1])
}

# Custom scoring function
def custom_f1_score(y_true, y_pred):
    # Convert IsolationForest predictions to binary format
    y_pred_binary = np.where(y_pred == -1, 1, 0)
    return f1_score(y_true, y_pred_binary, average='weighted')

if_base = IsolationForest(contamination=contamination, random_state=42)
if_random = RandomizedSearchCV(
    estimator=if_base, 
    param_distributions=param_dist, 
    n_iter=100, 
    cv=4, 
    verbose=2, 
    random_state=42, 
    n_jobs=-1,
    scoring=make_scorer(custom_f1_score)  # Use custom scoring function
)
if_random.fit(X_train, y_train)

print('Best hyperparameters found:')
print(if_random.best_params_)

best_if = if_random.best_estimator_

print('Testing classification with the best model...')
y_pred = best_if.predict(X_test)

# Convert predictions: -1 (anomaly/binding) to 1, 1 (normal/non-binding) to 0
y_pred_binary = np.where(y_pred == -1, 1, 0)

accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = custom_f1_score(y_test, y_pred_binary)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

with open('/path/to/isolationforest/summary_isolationForest.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}\n')
    f.write(f'F1 Score: {f1}\n')

model_path = '/path/to/isolationforest/isolation_forest_model_best.pkl'
joblib.dump(best_if, model_path)

# Plot anomaly scores
anomaly_scores = -best_if.score_samples(X_test)
plt.figure(figsize=(10, 6))
plt.hist([anomaly_scores[y_test == 0], anomaly_scores[y_test == 1]], 
         label=['Non-binding', 'Binding'], bins=50, stacked=True)
plt.title('Distribution of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Count')
plt.legend()
plt.savefig('/path/to/isolationforest/anomaly_scores_distribution.png')
plt.close()

print("Graphs saved as PNG images.")
print('\033[1mCompleted.\033[0m')
```
### Challenger #4: The Clamouring Convolutional Neural Network
![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*CW2aIrrvtR7c5clTPvV9aQ.png)
- **Type**: A neural network that uses deep learning to classify images.
- **Strengths**: Incredibly useful for processing image data with a high degree of accuracy. This is the kind of AI used in many [self-driving cars](https://neptune.ai/blog/self-driving-cars-with-convolutional-neural-networks-cnn).
- **Weaknesses**: This is the most computationally expensive of our fighters and lacks interpretability — we can’t know why it makes the decisions it makes, unlike RandomForest or XGBoost.

The genesis of this project was recreating [Visual](https://www.sciencedirect.com/science/article/pii/S0022519320301338?via%3Dihub=), a CNN built by Wardah et al. Building on its code, **I modified the model to use PepBDB-ML images**, a larger dataset with more features, including structural data.

How did it do?

- **Accuracy**: 78.59%
- **Precision**: 78.48% (+ 10 pts)*
- **Recall**: 79.32% (+12 pts)*
- **F1-Score**: 50.26%

Not too shabby! We also report an AUC (Area under ROC curve) of 0.86 (+ 5 pts)*, which tells us the model is fairly good at distinguishing between classes at various thresholds.

*(compared to the original paper).

Here’s what the confusion matrix looks like:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*hufe7nv3Ai9FyFFk)
*Confusion matrix for Visual-PepBDB classifier’s performance.*

You’ll notice some totals don’t line up fully; this is because the image dataset is slightly smaller than the tabular dataset. The reasons are discussed in the article I wrote about it.

Again, not great — but a step up from previous iterations.

The code here is a bit more involved, but you can find it at [Visual-PepBDB](https://github.com/MurtoHilali/Visual-PepBDB) and play with it at [Proteinloop](https://murto.co/proteinloop/pages/binding.html).

Press enter or click to view image in full size

![](https://miro.medium.com/v2/resize:fit:4800/format:webp/0*mE2RB0LEq0bVym1a)

Why did the CNN do better than everything else?

- Deep learning lets the model pick up on **more nuanced relationships** between data points.
- PepBDB-ML’s image dataset includes 7x41 windows of residues instead of the single entries we see in the table. This lets the **CNN capture and understand more context around residues**, providing more information to the model.

**We have our winner** — Visual-PepBDB, or more broadly, the Convolutional Neural Network!

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*9VHsV_29ufDT4fpOlI35PQ.png)
## Key Takeaways
- CNNs are by far the winner when it comes to detecting peptide-protein binding sites in PepBDB-ML — at least compared to RandomForest, XGBoost, and IsoForest.
- LLMs are tempting, trendy, and incredibly useful — but they may not be the right tools for your specific task. Traditional ML models can still be highly accurate, inexpensive, and more suited to your problem.

**If you’re in biology** — I highly recommend exploring ML methods to see how they can augment your research. They don’t need to be cutting-edge, as code from 5+ years could be a game changer for your work. AI is much more approachable than you think!

**If you’re in ML/DL/AI**—consider your entire toolbox, and apply it to biological problems! It’s filled with opportunities to apply classics and SOTA models.

If you made it all the way here, thanks for reading! Let me know what you think.