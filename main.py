import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn import metrics, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import zero_one_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from mlxtend.evaluate import bias_variance_decomp

from subprocess import call
last_column = 'label'

data_set = pd.read_csv("COMP4388-Dataset1.csv")

print(data_set.describe().to_string())


# --------------- correlation finder ------------

corrMatrix = data_set.corr()
fig, ax = plt.subplots(figsize=(10, 10))
sn.heatmap(corrMatrix, annot=True, linewidths=.5, ax=ax, linecolor='grey')  # seaborn
plt.title("Correlation OF 'Label'")  # Correlation plot
plt.show()
data_set[data_set.columns[1:]].corr()[last_column][:]
s1 = corrMatrix[last_column].sort_values(ascending=False)
print("----------The correlation values (for 'Label')------------------------------------\n")
print(s1.to_string())
################

# --------------- Drawing ------------

x_no_density_plot = data_set[data_set[last_column] == 0]
x_yes_density_plot = data_set[data_set[last_column] == 1]

x_no_density_plot.x.plot.density(color='green')  # 0

x_yes_density_plot.x.plot.density(color='red')  # 1
plt.title('Density plot: "X"')
plt.xlabel('X-param')
plt.show()
###
## for y

y_no_density_plot = data_set[data_set[last_column] == 0]
y_yes_density_plot = data_set[data_set[last_column] == 1]

y_no_density_plot.y.plot.density(color='green')  # 0

y_yes_density_plot.y.plot.density(color='red')  # 1
plt.title('Density plot: "Y"')
plt.xlabel('Y-param')
plt.show()
###
## for ID

y_no_density_plot = data_set[data_set[last_column] == 0]
y_yes_density_plot = data_set[data_set[last_column] == 1]

y_no_density_plot.ID.plot.density(color='green')  # 0

y_yes_density_plot.ID.plot.density(color='red')  # 1
plt.title('Density plot: "ID"')
plt.xlabel('ID-param')
plt.show()
########################################

X = data_set.drop(columns=[last_column])
y = data_set[last_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f'----------------------------------\n'
      f'Shape(number of rows for each one):\ntrain:{X_train.shape}\n'
      f'test: {X_test.shape}')

print(f'----------------------------------')
print(f'----Random Forest Classifier----')

classifier_randomForest = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,n_estimators=100)

# ######### Bias and variance ##############
mse1, bias1, var1 = bias_variance_decomp(classifier_randomForest, X_train=X_train.values, y_train=y_train.values, X_test=X_test.values, y_test=y_test.values, loss='mse', num_rounds=200, random_seed=1)
# summarize results
print('MSE: %.3f' % mse1)
print('Bias: %.3f' % bias1)
print('Variance: %.3f' % var1)
print(f'----------------------------------')

#######################
classifier_randomForest.fit(X_train, y_train)

y_predict = classifier_randomForest.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))
print("Precision: ", metrics.precision_score(y_test, y_predict))
print("Recall: ", metrics.recall_score(y_test, y_predict))
print(f'Error Rate= {zero_one_loss(y_test, y_predict)}')

print(f'RandomForestClassifier_classification_report:\n{metrics.classification_report(y_test, y_predict)}')
print(f'RandomForestClassifier_confusion_matrix:\n{metrics.confusion_matrix(y_test, y_predict)}')

estimator = classifier_randomForest.estimators_[5]

tree.export_graphviz(estimator, out_file='graph_RandomForest_proj3.dot'
                     , feature_names=X_train.columns,
                     class_names=['0', '1'],
                     label='all',
                     rounded=True,
                     filled=True)
# print(f'score:{classifier_randomForest.oob_score_}')


# plt.figure(figsize=(80,40))
# plot_tree(classifier_randomForest.estimators_[5], feature_names = X.columns,class_names=['ok', "Not ok"],filled=True);
# plt.show()

#####

print("------------------Decision Tree Classifier--------------------------")
model = DecisionTreeClassifier()


# ######### Bias and variance ##############
mse, bias, var = bias_variance_decomp(model, X_train=X_train.values, y_train=y_train.values, X_test=X_test.values, y_test=y_test.values, loss='mse', num_rounds=200, random_seed=1)
# summarize results
print('MSE: %.3f' % mse)
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)
#######################
print(f'----------------------------------')


model.fit(X_train, y_train)
y_predict = model.predict(X_test)


print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))
print("Precision: ", metrics.precision_score(y_test, y_predict))
print("Recall: ", metrics.recall_score(y_test, y_predict))
print(f'Error Rate= {zero_one_loss(y_test, y_predict)}')

print(f'DecisionTree_classification_report:\n{metrics.classification_report(y_test, y_predict)}')
print(f'DecisionTree_confusion_matrix:\n{metrics.confusion_matrix(y_test, y_predict)}')

# this is for the graph of the tree
tree.export_graphviz(model, out_file='graph_decision_tree_proj3.dot'
                     , feature_names=X_train.columns,
                     class_names=['0', '1'],
                     label='all',
                     rounded=True,
                     filled=True)

