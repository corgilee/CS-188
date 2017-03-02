"""
Author      : Yi-Chieh Wu, Sriram Sankararman
Description : Twitter
"""

from string import punctuation
import collections
import numpy as np

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.

    Parameters
    --------------------
        fname  -- string, filename

    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile):
    """
    Writes your label vector to the given file.

    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """

    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return

    np.savetxt(outfile, vec)


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.

    Parameters
    --------------------
        input_string -- string of characters

    Returns
    --------------------
        words        -- list of lowercase "words"
    """

    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.

    Parameters
    --------------------
        infile    -- string, filename

    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """

    word_list = {}
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
        words = [set(extract_words(line)) for line in fid]
        counter = 0
        for uw in words:
            for word in uw:
                if word not in word_list.keys():
                    word_list[word] = counter
                    counter+=1
        # print word_list
        print "max: " + str(max(word_list.values()))
        print "len: " + str(len(word_list.values()))



        ### ========== TODO : END ========== ###

    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.

    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)

    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """

    num_lines = sum(1 for line in open(infile,'rU'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    word_dict = extract_dictionary(infile)
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix
        count = 0
        for line in fid:
            this_tweet_words = set(extract_words(line))
            li = map(lambda l: 1 if l in this_tweet_words else 0, word_dict.keys())
            feature_matrix[count, :] = np.array(li)
            count+=1


        ### ========== TODO : END ========== ###
    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the
    true labels and the predicted labels.

    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1-score', 'auroc', 'precision',
                           'sensitivity', 'specificity'

    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1
    # print "metric in use: " + metric
    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    assert(metric in metric_list)
    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_label)
    elif metric == "f1_score":
        return metrics.f1_score(y_true, y_label)
    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, y_label)
    elif metric == "precision":
        return metrics.precision_score(y_true, y_label)
    elif metric == "sensitivity":
        cm = metrics.confusion_matrix(y_true, y_label)
        return cm[1][1]/float(cm[1][1] + cm[1][0])
    else:
        cm = metrics.confusion_matrix(y_true, y_label)
        return cm[0][0]/float(cm[0][0] + cm[0][1])


    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.

    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """

    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance
    perf = 0.0
    perf = []


    for train_index, test_index in kf:
        # get split data

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # train and test
        clf.fit(X_train, y_train)
        preds = clf.decision_function(X_test)
        # print "Metric in use: " + str(metric)
        perf.append(performance(y_true=y_test, y_pred=preds, metric = metric))
    # average the predictions
    return np.mean(np.array(perf))


    ### ========== TODO : END ========== ###


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.

    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """

    print 'Linear SVM Hyperparameter Selection based on ' + str(metric) + ':'
    C_range = 10.0 ** np.arange(-3, 3)

    ### ========== TODO : START ========== ###
    # part 2c: select optimal hyperparameter using cross-validation
    d = {}
    # for each hyperparameter
    for c in C_range:
        # run k-fold CV
        sclf = SVC(kernel="linear", C=c)
        # get the mean performance across k folds
        # print "metric in use: " + metric
        mean_perf_c = cv_performance(clf=sclf, X=X, y=y,kf=kf, metric=metric)
        d[c] = mean_perf_c

    best_c, perf = -1, -1
    for k, v in d.items():
        if v > perf:
            best_c, perf = k, v

    return best_c, d



    # return 1.0
    ### ========== TODO : END ========== ###


def select_param_rbf(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.

    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric  -- string, option used to select performance measure

    Returns
    --------------------
        gamma, C -- tuple of floats, optimal parameter values for an RBF-kernel SVM
    """

    print 'RBF SVM Hyperparameter Selection based on ' + str(metric) + ':'

    ### ========== TODO : START ========== ###
    # part 3b: create grid, then select optimal hyperparameters using cross-validation
    C_range = 10.0 ** np.arange(-3, 3)
    g_range = 10.0 ** np.arange(-3, 3)
    d = {}
    for c in C_range:
        for g in g_range:
            sclf = SVC(kernel="rbf", gamma = g, C=c)
            mean_perf_c_g = cv_performance(clf=sclf, X=X, y=y,kf=kf, metric=metric)
            d[(c,g)] = mean_perf_c_g

    best_c_g, perf = (-1, -1), -1
    for k, v in d.items():
        if v > perf:
            best_c_g, perf = k, v
    return best_c_g, d




    ### ========== TODO : END ========== ###


def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.

    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure

    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 4b: return performance on test data by first computing predictions and then calling performance
    preds = clf.decision_function(X)

    score = performance(y_true=y, y_pred=preds, metric=metric)
    return score
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################

def main() :
    np.random.seed(1234)

    # read the tweets and its labels
    dictionary = extract_dictionary('../data/tweets.txt')
    # print dictionary
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    # exit()
    y = read_vector_file('../data/labels.txt')

    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]

    ### ========== TODO : START ========== ###
    # part 1c: split data into training (training + cross-validation) and testing set
    X_train, y_train = X[:560, :], y[:560]
    X_test, y_test = X[560:, :], y[560:]

    # part 2b: create stratified folds (5-fold CV)
    kf = StratifiedKFold(y=y_train, n_folds=5)



    # part 2d: for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    best_c_dict = {}
    for metric in metric_list:
        best_c, d = select_param_linear(X=X_train, y=y_train, kf=kf, metric=metric)
        # save best Cs
        if best_c in best_c_dict:
            best_c_dict[best_c]+=1
        else:
            best_c_dict[best_c] = 1

        d = collections.OrderedDict(sorted(d.items()))

        print "best value of C found: " + str(best_c) + " with score: " + str(d[best_c])
        print d.items()
        print "\n"
    # part 3c: for each metric, select optimal hyperparameter for RBF-SVM using CV
    best_dict = {}
    for metric in metric_list:
        best_c_g, d = select_param_rbf(X=X_train, y=y_train, kf=kf, metric=metric)

        print "best combination of hyperparameters: " + str(best_c_g) + " with score: " + str(d[best_c_g])
        for k, v in d.items():
            assert(d[best_c_g] >= v)
            if d[best_c_g] == v:
                pass
                # print "params: " + str(k) + ", score: " + str(v)
        if best_c_g in best_dict:
            best_dict[best_c_g]+=1
        else:
            best_dict[best_c_g] = 1
        print "\n"

    # part 4a: train linear- and RBF-kernel SVMs with selected hyperparameters
    print "best C's: "
    print best_c_dict.items()
    print "best tupes: "
    print best_dict.items()

    best_c, num = -1, -1
    for k, v in best_c_dict.items():
        if v > num:
            best_c, num = k, v
    print "best c for linear: " + str(best_c) + " occured: " + str(num) + " times"

    best_tup, num = -1, -1
    for k, v in best_dict.items():
        if v > num:
            best_tup, num = k, v
    print "best pair for rbf: " + str(best_tup) + " occured: " + str(num) + " times"

    assert(best_c == 100.0) #linear
    assert(best_tup[0] == 100) #rbf C
    assert(best_tup[1] == 0.01) #rbf gamma

    print "using c for linear: " + str(best_c)
    linear = SVC(C=best_c, kernel="linear")
    print "using c: " + str(best_tup[0]) + " gamma: " + str(best_tup[1]) + " for gamma"
    rbf = SVC(C=best_tup[0], gamma = best_tup[1], kernel="rbf")
    # fit
    linear.fit(X_train, y_train)
    rbf.fit(X_train, y_train)
    # pred
    linear_pred, rbf_pred = linear.decision_function(X=X_test), rbf.decision_function(X=X_test)





    # part 4c: report performance on test data
    print "gathering final metrics! "
    for metric in metric_list:
        lin_score = performance(y_true=y_test, y_pred=linear_pred, metric=metric)
        rbf_score = performance(y_true=y_test, y_pred=rbf_pred, metric = metric)
        print "Metric: " + metric + ", " + "linear score: " + str(lin_score) + " ," + "rbf score: " + str(rbf_score)




    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()
