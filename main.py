"""
605.649 Introduction to Machine Learning
Dr. Donnelly
Programming Project #02
20220925
Jacob M. Lundeen

The purpose of this assignment is to give you some hands-on experience implementing a nonparametric
algorithm to perform classification and regression. Specifically, you will be implementing a k-nearest neighbor
classifier and regressor. In this exploration, be very careful with how the attributes are handled. Nearest
neighbor methods work best with numeric attributes, so some care will need to be taken to handle categorical
(i.e., discrete) attributes. One way of doing that is with the Value Difference Metric.

In this project, you will also be implementing both the edited nearest-neighbor and condensed nearest neighbor
algorithms.

Also in this project (and all future projects), the experimental design we will use 5x2 cross-validation.
You should have written code to support this in Project #1. Note that stratification is not expected for the
regression data sets. You should be sure to sample uniformly across all of the response values (i.e. targets)
when creating your folds. One approach for doing that (that’s not particularly random) is to sort the data
on the response variable and take every fifth point for a given fold.

Let’s talk about tuning. Basically, you will use the process you set up in Project #1. Suppose the entire
data set has 1,000 examples. Pull out 20% (i.e. 200 examples) for tuning, testing various parameter settings
via 5x2 cross validation with those 200 examples. After choosing the best parameters, use them with 5x2
cross validation using only the 80% (i.e. 800 examples). Report the results from the experiment by averaging
over the five held-out folds from the 80%.
"""
import math
from collections import Counter
from timeit import default_timer as timer

import pandas as pd
import numpy as np
from numpy.linalg import norm
from statistics import mean


# Function to read in the data set. For those data sets that do not have header rows, this will accept a tuple of
# column names. It is defaulted to fill in NA values with '?'.
def read_data(data, names=(), fillna=True):
    if not names:
        return pd.read_csv(data)
    if not fillna:
        return pd.read_csv(data, names=names)
    else:
        return pd.read_csv(data, names=names, na_values='?')


# The missing_values() function takes in the data set and the column name and then fills in the missing values of the
# column with the column mean. It does this 'inplace' so there is no copy of the data set made.
def missing_values(data, column_name):
    data[column_name].fillna(value=data[column_name].mean(), inplace=True)


# The cat_data() function handles ordinal and nominal categorical data. For the ordinal data, we use a mapper that maps
# the ordinal data to integers so they can be utilized in the ML algorithms. For nominal data, Pandas get_dummies()
# function is used.
def cat_data(data, var_name='', ordinal=False, data_name=''):
    if ordinal:
        if data_name == 'cars':
            buy_main_mapper = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
            door_mapper = {'2': 2, '3': 3, '4': 4, '5more': 5}
            per_mapper = {'2': 2, '4': 4, 'more': 5}
            lug_mapper = {'small': 0, 'med': 1, 'big': 2}
            saf_mapper = {'low': 0, 'med': 1, 'high': 2}
            class_mapper = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
            mapper = [buy_main_mapper, buy_main_mapper, door_mapper, per_mapper, lug_mapper, saf_mapper, class_mapper]
            count = 0
            for col in data.columns:
                data[col] = data[col].replace(mapper[count])
                count += 1
            return data
        elif data_name == 'forest':
            month_mapper = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
                            'oct': 10, 'nov': 11, 'dec': 12}
            day_mapper = {'sun': 1, 'mon': 2, 'tue': 3, 'wed': 4, 'thu': 5, 'fri': 6, 'sat': 7}
            data.month = data.month.replace(month_mapper)
            data.day = data.day.replace(day_mapper)
            return data
        elif data_name == 'cancer':
            class_mapper = {2: 0, 4: 1}
            data[var_name] = data[var_name].replace(class_mapper)
            return data
    else:
        return pd.get_dummies(data, columns=var_name, prefix=var_name)


# The discrete() function transforms real-valued data into discretized values. This function provides the ability to do
# both equal width (pd.cut()) and equal frequency (pd.qcut()). The function also provides for discretizing a single
# feature or the entire data set.
def discrete(data, equal_width=True, num_bin=20, feature=""):
    if equal_width:
        if not feature:
            for col in data.columns:
                data[col] = pd.cut(x=data[col], bins=num_bin)
            return data
        else:
            data[feature] = pd.cut(x=data[feature], bins=num_bin)
            return data
    else:
        if not feature:
            for col in data.columns:
                data[col] = pd.qcut(x=data[col], q=num_bin, duplicates='drop')
            return data
        else:
            data[feature] = pd.qcut(x=data[feature], q=num_bin)
            return data


# The standardization() function performs z-score standardization on a given train and test set. The function
# will standardize either an individual feature or the entire data set. If the standard deviation of a variable is 0,
# then the variable is constant and adds no information to the regression, so it can be dropped from the data set.
def standardization(train, test=pd.DataFrame(), feature=''):
    if test.empty:
        for col in train.columns:
            if train[col].std() == 0:
                train.drop(col, axis=1, inplace=True)
            else:
                train[col] = (train[col] - train[col].mean()) / train[col].std()
        return train
    elif not feature:
        for col in train.columns:
            if train[col].std() == 0:
                train.drop(col, axis=1, inplace=True)
                test.drop(col, axis=1, inplace=True)
            else:
                test[col] = (test[col] - train[col].mean()) / train[col].std()
                train[col] = (train[col] - train[col].mean()) / train[col].std()
        return train, test
    else:
        test[feature] = (test[feature] - train[feature].mean()) / train[feature].std()
        train[feature] = (train[feature] - train[feature].mean()) / train[feature].std()
        return train, test


# The cross_val() function performs the cross validation of the data set. For classification, the data is stratified
# (to include the validation set).
def cross_val(data, class_var, pred_type, k=5, validation=False):
    if pred_type == 'classification':
        val_split, data_splits = class_split(data, k, class_var, validation)
    elif pred_type == 'regression':
        val_split, data_splits = reg_split(data, k, validation)
    else:
        print("Please provide a prediction choice.")
        return
    # Here is where the program rotates through the k-folds of the data. The For loop identifies each test set, removes
    # it from the folds, and uses the rest of the folds for the training set. Regression data (both training and test)
    # are standardized. The results, validation data set, and test set are then returned.
    results = []
    for item in range(k):
        train = data_splits.copy()
        test = data_splits[item]
        del train[item]
        train = pd.concat(train, sort=False)
        if pred_type == 'regression':
            train, test = standardization(train.copy(), test.copy())
        results.append(np.full(len(test), null_model(train.copy(), class_var, pred_type=pred_type)))
    if pred_type == 'regression':
        norm_data = standardization(pd.concat(data_splits))
        return val_split, norm_data, [item for subresult in results for item in subresult]
    else:
        return val_split, pd.concat(data_splits), [item for subresult in results for item in subresult]


# The class_split() function handles splitting and stratifying classification data. Returns validation set and
# data_splits.
def class_split(data, k, class_var, validation=False):
    # Group the data set by class variable using pd.groupby()
    grouped = data.groupby([class_var])
    grouped_l = []
    data_splits = []
    # Create stratified validation set using np.split(). 20% from each group is appended to the validation set, the rest
    # will be used for the k-folds.
    if validation:
        grouped_val = []
        grouped_dat = []
        for name, group in grouped:
            val, dat = np.split(group, [int(0.2 * len(group))])
            grouped_val.append(val)
            grouped_dat.append(dat)
        # Split the groups into k folds
        for i in range(len(grouped_dat)):
            grouped_l.append(np.array_split(grouped_dat[i], k))
    else:
        for name, group in grouped:
            grouped_l.append(np.array_split(group.iloc[np.random.permutation(np.arange(len(group)))], k))
        for i in range(len(grouped_l)):
            for j in range(len(grouped_l[i])):
                grouped_l[i][j].reset_index(inplace=True, drop=True)
        for i in range(k):
            temp = grouped_l[0][i]
            for j in range(1, len(grouped_l)):
                temp = pd.concat([temp, grouped_l[j][i]], ignore_index=True)
            data_splits.append(temp)
    # Reset indices of the folds
    for item in range(len(grouped_l)):
        for jitem in range(len(grouped_l[item])):
            grouped_l[item][jitem].reset_index(inplace=True, drop=True)
    # Combine folds from each group to create stratified folds
    for item in range(k):
        tempo = grouped_l[0][item]
        for jitem in range(1, len(grouped_l)):
            tempo = pd.concat([tempo, grouped_l[jitem][item]], ignore_index=True)
        data_splits.append(tempo)
    if validation:
        grouped_val = pd.concat(grouped_val)
    else:
        grouped_val = 0
    return grouped_val, pd.concat(data_splits)


# The reg_split() function creates the k-folds for regression data.
def reg_split(data, k, validation=False):
    # Randomize the data first
    df = data.sample(frac=1, random_state=42).reset_index(drop=True)
    # If a validation set is required, divide the data set 20/80 and return the sets
    if validation:
        val_fold, data_fold = np.split(df, [int(.2 * len(df))])
        if k == 1:
            return val_fold, data_fold.reset_index(drop=True)
        else:
            data_fold = np.array_split(data_fold, k)
            return val_fold, data_fold
    # If no validation set is required, split the data by the requested k
    else:
        data_fold = np.array_split(df, k)
        val_fold = 0
        return val_fold, data_fold


# The k2_cross() function performs Kx2 cross validation.
def k2_cross(data, k, class_var, pred_type, sigma=0.5, epsilon=0.5, reduce=''):
    reduce = reduce
    results = []
    count = 0
    if pred_type == 'regression':
        data = standardization(data)
    if reduce == 'cnn':
        data = cnn(data, class_var, pred_type, epsilon)
    elif reduce == 'enn':
        data = enn(data, pred_type, class_var, epsilon)

    # As we loop over k, we randomize each loop and then split the data 50/50 into train and test sets (standardizing
    # when doing regression). The learning algorithm is trained on the training set first and then tested on the test
    # set. They are then flipped (trained on the test set and tested on the train set). So we get 2k experiments.
    while count < 5:
        rand_df = data.sample(frac=1, random_state=42).reset_index(drop=True)
        if count == 0:
            file1.write('The length of the data set is: ' + str(len(rand_df)) + '\n')
            dfs = np.array_split(rand_df, 2)
            train = dfs[0]
            test = dfs[1]
            file1.write('The length of the training set is: ' + str(len(train)) + '\n')
            file1.write('The length of the test set is: ' + str(len(test)) + '\n')
        else:
            dfs = np.array_split(rand_df, 2)
            train = dfs[0]
            test = dfs[1]
        pred, true = knn_predict(train.copy(), test.copy(), k, pred_type=pred_type, target=class_var, sigma=sigma)
        if pred_type == 'regression':
            pred = reg_adjust(pred, true, epsilon)
        results.append(eval_metrics(true, pred, pred_type))
        pred, true = knn_predict(test.copy(), train.copy(), k, pred_type=pred_type, target=class_var, sigma=sigma)
        if pred_type == 'regression':
            pred = reg_adjust(pred, true, epsilon)
        results.append(eval_metrics(true, pred, pred_type))
        count += 1
    metric1 = []
    metric2 = []
    metric3 = []
    for lst in results:
        metric1.append(lst[0])
        metric2.append(lst[1])
        metric3.append(lst[2])
    final_metrics = [mean(metric1), mean(metric2), mean(metric3)]
    return final_metrics


# The reg_adjust() algorithm takes an accurately predicted example during regression and sets the predicted value to be
# equal to the true value
def reg_adjust(pred, true, epsilon):
    for i in range(len(pred)):
        if (true[i] - epsilon < pred[i]) & (pred[i] < true[i] + epsilon):
            pred[i] = true[i]
    return pred


# Calculate the R2 score
def r2_score(true, pred):
    ss_t = 0
    ss_r = 0
    mean_true = np.mean(true)
    for i in range(len(true)):
        ss_r += (true[i] - pred[i]) ** 2
        ss_t += (true[i] - mean_true) ** 2
    if ss_t == 0:
        mse = 1
    else:
        mse = round(1 - (ss_r / ss_t), 3)
    return mse


# The eval_metrics() function returns the classification or regression metrics.
def eval_metrics(true, predicted, eval_type='regression'):
    # For regression, we create the correlation matrix and then calculate the R2, Person's Correlation, and MSE.
    if eval_type == 'regression':
        r2_s = r2_score(true, predicted)
        persons = round(pd.Series(true).corr(pd.Series(predicted)), 3)
        mse = round(np.square(np.subtract(true, predicted)).mean(), 3)
        return r2_s, persons, mse
    # For classification, we calculate Precision, Recall, and F1 scores.
    elif eval_type == 'classification':
        total_examp = len(true)
        mc_weights = []
        precision = []
        recall = []
        f_1 = []
        count = 0
        for label in np.unique(true):
            class_len = np.sum(true == label)
            true_pos = np.sum((true == label) & (predicted == label))
            true_neg = np.sum((true != label) & (predicted != label))
            false_pos = np.sum((true != label) & (predicted == label))
            false_neg = np.sum((true == label) & (predicted != label))
            if true_pos & false_pos == 0:
                precision.append(0)
            else:
                precision.append(true_pos / (true_pos + false_pos))
            if true_pos + false_neg == 0:
                recall.append(0)
            else:
                recall.append(true_pos / (true_pos + false_neg))
            if precision[count] + recall[count] == 0:
                f_1.append(0)
            else:
                if len(np.unique(true)) > 1:
                    f_1.append((class_len / total_examp) * 2 * (precision[count] * recall[count]) / (
                                precision[count] + recall[count]))
                else:
                    f_1.append(2 * (precision[count] * recall[count]) / (precision[count] + recall[count]))
            count += 1
        if count > 1:
            return mean(precision), mean(recall), mean(f_1)
        else:
            return sum(precision), sum(recall), sum(f_1)
    else:
        print("Please choose a prediction method.")
        return


# The null_model() function will return the mean value of the target variable or the most common class for
# classification.
def null_model(train, class_var, pred_type="regression"):
    if pred_type == 'regression':
        return np.mean(train[class_var])
    elif pred_type == 'classification':
        unique_elem, count = np.unique(train[class_var], return_counts=True)
        return unique_elem[count == count.max()]
    else:
        print("Please choose a prediction method.")
        return


# The KNN classifier/regressor function. Using a reduction method is optional.
def knn_predict(train, test, k=5, pred_type='regression', sigma=0.5, target=''):
    X_train = train.drop(target, axis=1).to_numpy()
    y_train = train[target]

    # Create the test X,y sets
    X_test = test.drop(target, axis=1).to_numpy()
    y_test = test[target]
    y_hat_test = []
    if pred_type == 'regression':
        file2 = open('KNN Distances.txt', 'w')
        file3 = open('KNN Regression.txt', 'w')
    elif pred_type == 'classification':
        file4 = open('KNN Classification.txt', 'w')
    # We loop over each test point, calculating the distance from that test point to each train point
    count = 0
    for test_point in X_test:

        distances = []
        for train_point in X_train:
            dist = distance(train_point, test_point)
            if pred_type == 'regression':
                file2.write('The Euclidean distance calculated is: ' + str(dist) + '\n')
            distances.append(dist)

        # We create a DataFrame from the distances calculated, sort them, and then save the small k distances
        distsdf = pd.DataFrame(data=distances, columns=['dist'], index=y_train.index)
        nndf = distsdf.sort_values(by=['dist'], axis=0)[:k]
        if pred_type == 'regression':
            if count == 0:
                print("Below are all of the distances and then those distances ordered and reduced to k NN for Regression")
                print(test_point)
                print(distsdf)
                print(nndf)
        elif pred_type == 'classification':
            if count == 0:
                print("Below are all of the distances and then those distances ordered and reduced to k NN for Classification")
                print(test_point)
                print(distsdf)
                print(nndf)
        # print(test_point)
        # With the shortest distances found, we pass the class of those shortest distances to the Counter() function,
        # which tells us which class is the most common class. For regression, we take the mean of the k shortest
        # distances.
        if pred_type == 'classification':
            counter = Counter(y_train[nndf.index])
            pred = counter.most_common()[0][0]
        elif pred_type == 'regression':
            nndf = rbf(nndf, sigma)
            if pred_type == 'regression':
                if count == 0:
                    print("Below is the calculation of the Kernel Function on the values of the k NN.")
                    print(nndf)
            pred = nndf.mul(y_train[nndf.index], axis=0)
            pred = mean(pred['dist'])
        y_hat_test.append(pred)
        count += 1
    # file.close()
    return y_hat_test, y_test.tolist()


# Calculate the Euclidean distance between two examples
def distance(train_point, test_point):
    return norm(train_point - test_point)


# The Radial Basis Function, or Gaussian Kernel, used to weigh the values for the k nearest neighbors.
def rbf(nndf, sigma):
    return np.exp(-(nndf ** 2) / (2 * (sigma ** 2)))


# The CNN reduction function. Called within the KNN predictor if the correct variable is passed.
def cnn(train, class_var, pred_type, epsilon):
    # Set up our initial variables, splitting the training set into X_train and y_train. The Z and y data sets are set to
    # the first example in the training set.
    names = train.columns.values.tolist()
    X_train = train.drop(class_var, axis=1)
    y_train = train[class_var].reset_index(drop=True)
    Z = X_train.iloc[[0]]
    X_train.drop(0, inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    y = [y_train[0]]
    y_train.drop(axis=0, index=0, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    previous = 0
    new = len(Z)
    print('The length of the condensed set starts at: ' + str(len(Z)))
    # file.write("\nThe size of the data set before CNN was: %3.f" % len(train))
    # As long as examples continue to be added to the Z and y sets, continue to make passes through the training set
    while previous < new:
        t = []
        previous = new
        # Iterate through each row in the training set
        count = 0
        for index, train_point in X_train.iterrows():
            distances = []
            # Iterate through each row in the Z set and calculate the distances
            for zindex, Z_point in Z.iterrows():
                dist = distance(train_point.values, Z_point.values)
                distances.append(dist)
            # Put the distances into a DF and sort by shortest to longest and keep only the shortest distance. This
            # keeps the index of the shortest distance so it can be used to add to the Z and y DFs.
            distsdf = pd.DataFrame(data=distances, columns=['dist'])
            nndf = distsdf.sort_values(by=['dist'], axis=0)[0:1]
            # The predicted value is not equal to the true value, add it to the Z and y DFs, otherwise continue on
            if pred_type == 'classification':
                pred = y[nndf.index.values.astype(int)[0]]
                if y_train.loc[index] != pred:
                    train_point = train_point.values
                    train_point = pd.DataFrame(train_point.reshape(1, -1), columns=names[:-1])
                    Z = pd.concat([Z, train_point], axis=0, ignore_index=True)
                    y.append(y_train.iloc[index])
            elif pred_type == 'regression':
                pred = y[nndf.index.values.astype(int)[0]]
                if (y_train.iloc[index] - epsilon <= pred) and (pred <= y_train.iloc[index] + epsilon):
                    t.append(train_point)
                else:
                    print("The below point has been added to the condensed set during CNN")
                    print(train_point)
                    train_point = train_point.values
                    train_point = pd.DataFrame(train_point.reshape(1, -1), columns=names[:-1])
                    Z = pd.concat([Z, train_point], axis=0, ignore_index=True)
                    y.append(y_train.iloc[index])
                    print(str(len(Z)))
        X_train = pd.DataFrame(t, columns=names[:-1])
        new = len(Z)
        count += 1
    # file.write("\nThe size of the data set after CNN is: %3.f" % len(Z))
    # Put Z and y into a DF and return it
    y = pd.Series(y.copy())
    Z = pd.DataFrame(np.array(Z))
    Z = pd.concat([Z, y], axis=1, ignore_index=True)
    Z.columns = names
    return Z


# The ENN reduction function. Called within the KNN predictor if the correct variable is passed.
def enn(train, method, target, epsilon):
    # Set up the initial variables by splitting the training set into X_train and y_train. The y array keeps track of
    # the original true values
    names = train.columns.values.tolist()
    X_train = train.drop(target, axis=1)
    y_train = train[target].reset_index(drop=True)
    y = y_train.to_numpy()
    previous = len(X_train)
    new = 0
    print('The starting size of the training set is: ' + str(len(X_train)))
    # file.write("\nThe size of the data set before ENN was: %3.f" % previous)
    # As long as there is significant change between passes, keep going
    while (previous - new) > 2:
        new = previous
        # print(previous, new)
        # Iterate over each training point
        for index, train_point in X_train.iterrows():
            distances = []
            # Iterate over each training point
            for iindex, point in X_train.iterrows():
                # If the training point is the same, skip it
                if iindex == index:
                    continue
                dist = distance(train_point.values, point.values)
                distances.append(dist)
            # print(distances)
            # Put all the distances into a DF and sort. Return only the shortest distance
            distsdf = pd.DataFrame(data=distances, columns=['dist'])
            nndf = distsdf.sort_values(by=['dist'], axis=0)[0:1]
            # print(nndf)
            # If the predicted value is incorrect, delete it from the training set
            if method == 'classification':
                pred = y[nndf.index.values.astype(int)[0]]
                if y_train.loc[index] != pred:
                    print("The follow point has been removed from the training set using ENN")
                    print(train_point)
                    X_train.drop(index, inplace=True)
                    y_train.drop(index, inplace=True)
                    print(str(len(X_train)))
            elif method == 'regression':
                pred = y[nndf.index.values.astype(int)[0]]
                if math.isnan(pred):
                    pred = 0
                if (y_train.loc[index] - epsilon <= pred) and (pred <= y_train.loc[index] + epsilon):
                    continue
                else:
                    X_train.drop(index, inplace=True)
                    y_train.drop(index, inplace=True)
        previous = len(X_train)
    # file.write("\nThe size of the data set after ENN is: %3.f" % len(X_train))
    Z = pd.concat([X_train, y_train], axis=1, ignore_index=True)
    Z.columns = names
    return Z


# The hyper_tune() function is used to tune the hyperparameters of the KNN algorithm
def hyper_tune(data, class_var, pred_type):
    # Create lists of values for k, sigma, and epsilon and loop over to determine parameters (lowest MSE or highest F1)
    ks = [i for i in np.arange(1, 10, 1)]
    if pred_type == 'regression':
        sigma = [round(i, 3) for i in np.linspace(0.001, 5, 10)]
        epsilon = [round(i, 3) for i in np.linspace(0.0, 1.0, 10)]
        df_hyper = pd.DataFrame(columns=['K', 'Sigma', 'Epsilon', 'MSE'])
        for k in ks:
            for sig in sigma:
                for ep in epsilon:
                    results = k2_cross(data=data, k=k, class_var=class_var, pred_type=pred_type, sigma=sig, epsilon=ep)
                    temp = pd.DataFrame(data={'K': k, 'Sigma': sig, 'Epsilon': ep, 'MSE': round(results[2], 3)},
                                        index=[0])
                    df_hyper = pd.concat([df_hyper, temp], ignore_index=True)
        mse_min = df_hyper[df_hyper.MSE == df_hyper.MSE.min()]
        mse_min.reset_index(drop=True)
        return mse_min.iat[0, 0], mse_min.iat[0, 1], mse_min.iat[0, 2]
    else:
        df_hyper = pd.DataFrame(columns=['K', 'F1'])
        for k in ks:
            results = k2_cross(data, k, class_var, pred_type)
            temp = pd.DataFrame(data={'K': k, 'F1': round(results[2], 3)}, index=[0])
            df_hyper = pd.concat([df_hyper, temp], ignore_index=True)
        f1_max = df_hyper[df_hyper.F1 == df_hyper.F1.max()]
        f1_max.reset_index(drop=True)
        return f1_max.iat[0, 0]


if __name__ == '__main__':
    # This first section read in the six data sets. 5 of the 6 data sets must have their column names hardcoded
    # (the forest data set is the only one that doesn't). A tuple is created with the column names and then passed
    # to the read_data() function along with the name of the data set. The house data is the only data set that does not
    # need missing values changed to '?'.
    ab_names = ('sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight',
                'shell_weight', 'rings')
    abalone = read_data('abalone.data', ab_names)
    cancer_names = ('code_num', 'clump_thick', 'unif_size', 'unif_shape', 'adhesion', 'epithelial_size', 'bare_nuclei',
                    'bland_chromatin', 'norm_nucleoli', 'mitosis', 'class')
    cancer = read_data('breast-cancer-wisconsin.data', cancer_names)
    car_names = ('buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class')
    cars = read_data('car.data', car_names)
    forest = read_data('forestfires.csv')
    house_names = ('class', 'infants', 'water_sharing', 'adoption_budget', 'physician_fee', 'salvador_aid',
                   'religious_schools', 'andit_sat_ban', 'aid_nic_contras', 'mx_missile', 'immigration',
                   'synfuels_cutback', 'edu_spending', 'supderfund_sve', 'crime', 'duty_free', 'export_admin_africa')
    house = read_data('house-votes-84.data', house_names, fillna=False)
    machine_names = ('vendor', 'model', 'myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax', 'prp', 'erp')
    machine = read_data('machine.data', machine_names)
    missing_values(cancer, 'bare_nuclei')
    # Here we handle categorical data
    abalone = cat_data(abalone, [ab_names[0]])
    temp = abalone['rings']
    abalone = abalone.drop(columns='rings')
    abalone.insert(loc=len(abalone.columns), column='rings', value=temp)
    cancer = cat_data(cancer, var_name='class', data_name='cancer', ordinal=True)
    cars = cat_data(cars, var_name=car_names, data_name='cars', ordinal=True)
    forest = cat_data(forest, data_name='forest', ordinal=True)
    house = cat_data(house, var_name=list(house_names))
    temp = house['class_republican']
    house = house.drop(columns='class_republican')
    house.insert(loc=len(house.columns), column='class_republican', value=temp)
    machine = cat_data(machine, var_name=list(machine_names[0:2]))
    temp = machine['erp']
    machine = machine.drop(columns='erp')
    machine.insert(loc=len(machine.columns), column='erp', value=temp)

    # Here we print the results of the Null Model using Kx2 CV
    class_var = 'erp'
    pred_type = 'regression'
    data = machine
    dname = 'Machine'
    file1 = open("Results.txt", 'w', encoding='utf-8')
    # file = open(str(dname) + "_results.txt", 'w', encoding='utf-8')
    # file.write("The " + str(dname) + " data, with target variable " + class_var + ", is a " + pred_type + " problem.")
    val_set, data_set = reg_split(data=data, k=1, validation=True)
    val_set = standardization(val_set)
    # start = timer()
    # k, sigma, epsilon = hyper_tune(data=val_set, class_var=class_var, pred_type=pred_type)
    # end = timer()
    # time = round((end - start) / 60, 3)
    # file.write("\nHyperparameter tuning took: " + str(time) + " minutes.")
    k = 2
    sigma = 0.447
    epsilon = 0.5
    # file.write('\nThe hyperparameter K is tune to: ' + str(k))
    # file.write('\nThe hyperparameter \u03C3 is tuned to: ' + str(sigma))
    # file.write('\nThe hyperparameter \u03B5 is tuned to: ' + str(epsilon) + "\n")
    # start = timer()
    results = k2_cross(data_set, k, class_var, pred_type, sigma, epsilon, reduce='cnn')
    # end = timer()
    # time = round((end - start) / 60, 3)
    # file.write("\nKNN with no reduction took: " + str(time) + " minutes.")
    # file.write("\nThe R\u00b2 is: " + str(round(results[0], 3)))
    # file.write("\nThe Pearsons Correlation is: " + str(round(results[1], 3)))
    # file.write("\nThe MSE is: " + str(round(results[2], 3)) + "\n")
    # start = timer()
    # results = k2_cross(data_set, k, class_var, pred_type, sigma, epsilon, reduce='cnn')
    # end = timer()
    # time = round((end - start) / 60, 3)
    # file.write("\nKNN with CNN reduction took: " + str(time) + " minutes.")
    # file.write("\nThe R\u00b2 is: " + str(round(results[0], 3)))
    # file.write("\nThe Pearsons Correlation is: " + str(round(results[1], 3)))
    # file.write("\nThe MSE is: " + str(round(results[2], 3)) + "\n")
    # start = timer()
    # results = k2_cross(data_set, k, class_var, pred_type, sigma, epsilon, reduce='enn')
    # end = timer()
    # time = round((end - start) / 60, 3)
    # file.write("\nKNN with ENN reduction took: " + str(time) + " minutes.")
    # file.write("\nThe R\u00b2 is: " + str(round(results[0], 3)))
    # file.write("\nThe Pearsons Correlation is: " + str(round(results[1], 3)))
    # file.write("\nThe MSE is: " + str(round(results[2], 3)) + "\n")

    class_var = 'class_republican'
    pred_type = 'classification'
    data = house
    dname = 'House'
    val_set, data_set = class_split(data, 1, class_var, True)
    # start = timer()
    # k = hyper_tune(val_set, class_var, pred_type)
    # end = timer()
    # time = round((end - start) / 60, 3)
    # file.write("\nHyperparameter tuning took: " + str(time) + " minutes.")
    # file.write("\nThe hyperparameter K is tune to: " + str(k) + "\n")
    k = 6
    # start = timer()
    results = k2_cross(data_set, k, class_var, pred_type, reduce='enn')
    # end = timer()
    # time = round((end - start) / 60, 3)
    # file.write("\nKNN with no reduction took: " + str(time) + " minutes.")
    # file.write('\nThe Precision score is: ' + str(round(results[0], 3)))
    # file.write('\nThe Recall score is: ' + str(round(results[1], 3)))
    # file.write('\nThe F1 score is: ' + str(round(results[2], 3)) + "\n")
    # start = timer()
    # results = k2_cross(data_set, k, class_var, pred_type, reduce='cnn')
    # end = timer()
    # time = round((end - start) / 60, 3)
    # file.write("\nKNN with CNN reduction took: " + str(time) + " minutes.")
    # file.write('\nThe Precision score is: ' + str(round(results[0], 3)))
    # file.write('\nThe Recall score is: ' + str(round(results[1], 3)))
    # file.write('\nThe F1 score is: ' + str(round(results[2], 3)) + "\n")
    # start = timer()
    # results = k2_cross(data_set, k, class_var, pred_type, reduce='enn')
    # end = timer()
    # time = round((end - start) / 60, 3)
    # file.write("\nKNN with ENN reduction took: " + str(time) + " minutes.")
    # file.write('\nThe Precision score is: ' + str(round(results[0], 3)))
    # file.write('\nThe Recall score is: ' + str(round(results[1], 3)))
    # file.write('\nThe F1 score is: ' + str(round(results[2], 3)))

    # file.close()
