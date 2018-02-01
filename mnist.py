from __future__ import unicode_literals, division

import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from sklearn.base import clone
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

'''based on: https://github.com/davefernig/alp'''

DEBUG = False


class ActiveLearning(object):
    def __init__(self, classifier_type, committee_number, sampling_methods):
        if classifier_type == 'LR':
            clf = [LogisticRegression()]
        else:
            # SVC params -
            # https://github.com/ksopyla/svm_mnist_digit_classification/blob/master/svm_mnist_classification.py
            clf = [SVC(probability=True, C=5, gamma=0.05)]
        clf += [LogisticRegression() for _ in range(committee_number)]  # for voting
        self.clf = clf
        self.sampling_methods = sampling_methods

    @staticmethod
    def sampling(clf, strategy, x_unlabeled, x_labeled, y_labeled, num_queries):
        if x_labeled.shape[0] == 0:
            strategy = 'random'

        if strategy == 'random':
            idx = np.array(range(x_unlabeled.shape[0]))
            np.random.shuffle(idx)
        else:
            for model in clf:
                model.fit(x_labeled, y_labeled)

            if strategy == 'least_confident':  # uncertainty -- least confident
                probs = clf[0].predict_proba(x_unlabeled)
                scores = 1 - np.amax(probs, axis=1)

            elif strategy == 'max_margin':  # uncertainty -- max margin
                probs = clf[0].predict_proba(x_unlabeled)
                margin = np.partition(-probs, 1, axis=1)
                scores = -np.abs(margin[:, 0] - margin[:, 1])

            elif strategy == 'entropy':  # uncertainty -- entropy
                probs = clf[0].predict_proba(x_unlabeled)
                scores = np.apply_along_axis(entropy, 1, probs)

            elif strategy == 'average_kl_divergence':  # query_by_committee -- average_kl_divergence
                preds = []
                for model in clf:
                    preds.append(model.predict_proba(x_unlabeled))
                consensus = np.mean(np.stack(preds), axis=0)
                divergence = [entropy(consensus.T, y_out.T) for y_out in preds]
                scores = np.apply_along_axis(np.mean, 0, np.stack(divergence))

            elif strategy == 'centers_confidence':  # uncertainty -- least confident, using clustering, distance as measurement
                opt_centroids = [x_labeled[y_labeled == k] for k in np.unique(y_labeled)]
                classes = []
                for _ in range(10):
                    centroids = [random.choice(a) for a in opt_centroids]
                    classes.append(np.array(
                        [np.argmin([np.dot(x_i - y_k, x_i - y_k) for y_k in centroids]) for x_i in x_unlabeled]))
                scores = np.std(classes, axis=0)

            elif strategy == 'centers_distances':  # uncertainty -- least confident, using clustering, distance as measurement
                centroids = [x_labeled[y_labeled == k].mean(axis=0) for k in np.unique(y_labeled)]
                distances_x_unlabeld = np.array(
                    [[np.dot(x_i - y_k, x_i - y_k) for y_k in centroids] for x_i in x_unlabeled])
                scores = (np.min(distances_x_unlabeld, axis=1))

            idx = np.argsort(-scores)  # reversed
        return idx[:num_queries]

    def sample_and_check_performance(self, x_labeled, x_test, x_unlabeled, y_labeled, y_test, y_oracle,
                                     num_queries, strategy):
        main_clf = clone(self.clf[0])
        clf = [clone(model) for model in self.clf]
        clf_ind = self.sampling_methods[strategy]

        idx = self.sampling(clf[clf_ind[0]:clf_ind[1]], strategy, x_unlabeled, x_labeled, y_labeled, num_queries)
        x_augmented = np.concatenate((x_labeled, x_unlabeled[idx, :]))
        y_augmented = np.concatenate((y_labeled, y_oracle[idx]))

        main_clf.fit(x_augmented, y_augmented)
        preds = main_clf.predict(x_test)
        del clf
        accuracy, precision, recall = self.multiclass_performance(y_test, preds, np.unique(y_test))

        x_unlabeled = np.delete(x_unlabeled, idx, axis=0)
        y_oracle = np.delete(y_oracle, idx)
        x_labeled, y_labeled = x_augmented, y_augmented

        return accuracy, precision, x_labeled, y_labeled, x_unlabeled, y_oracle

    @staticmethod
    def multiclass_performance(y, preds, classes):
        tp = np.array([np.sum(np.logical_and(y == c, preds == c)) for c in classes])
        tn = np.array([np.sum(np.logical_and(y != c, preds != c)) for c in classes])
        fp = np.array([np.sum(np.logical_and(y != c, preds == c)) for c in classes])
        fn = np.array([np.sum(np.logical_and(y == c, preds != c)) for c in classes])

        accuracy, precision, recall = np.mean((tp + tn) / (tp + tn + fp + fn + 0.01)), np.mean(tp / (tp + fp + 0.01)), \
                                      np.mean(tp / (tp + tn + 0.01))
        # print confusion_matrix(y, preds)

        return accuracy, precision, recall


class Simulation(object):
    def __init__(self, sampling_methods, num_queries_list, iter_num, data_name):
        self.sampling_methods, self.num_queries_list, self.iter_num = sampling_methods, num_queries_list, iter_num
        self.results = {strategy: {'accuracy': [], 'precision': []} for strategy in sampling_methods}
        self.x_labeled, self.x_test, self.x_unlabeled, self.y_labeled, self.y_test, self.y_oracle = self.init_dataset(
            data_name)

    def plot_results(self, plt_name):
        num_queries_list = np.cumsum(self.num_queries_list)
        for ii, strategy in enumerate(self.sampling_methods):
            if strategy != 'random':
                color_b = 1.0 * ii / len(self.sampling_methods)
                color_g = 1 - color_b
            else:
                color_b = 0.0
                color_g = 0.0
            plt.subplot(131)
            plt.plot(num_queries_list, self.results[strategy]['accuracy'], color=(0.0, color_g, color_b))
            plt.subplot(133)
            plt.plot(num_queries_list, self.results[strategy]['precision'], color=(0.0, color_g, color_b))
        plt.subplot(131)
        plt.ylabel('Accuracy')
        plt.xlabel('Number of Queries')
        plt.title('MNIST')
        plt.legend([ii.replace('_', ' ').capitalize() for ii in self.sampling_methods], loc=4)
        plt.subplot(133)
        plt.ylabel('Precision')
        plt.xlabel('Number of Queries')
        plt.savefig(plt_name)
        plt.show()

    @staticmethod
    def init_dataset(data_name):
        dataset = fetch_mldata(data_name)
        x, y = dataset.data, dataset.target
        x = x.astype('float')
        x = (x - np.min(x)) / (np.max(x) - np.min(x))

        if DEBUG:
            idx_init = list(range(x.shape[0]))
            np.random.shuffle(idx_init)
            idx_init = idx_init[:5000]
            x, y = x[idx_init, :], y[idx_init]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7)
        x_unlabeled, x_labeled, y_oracle, y_labeled = train_test_split(x_train, y_train, test_size=0)

        return x_labeled, x_test, x_unlabeled, y_labeled, y_test, y_oracle

    def simulate(self, plt_name, classifier_type, committee_number):
        simulate_al = ActiveLearning(classifier_type, committee_number, self.sampling_methods)

        for num_queries in self.num_queries_list:
            for strategy in self.sampling_methods:
                precision = []
                accuracy = []
                x_labeled, x_test, x_unlabeled, y_labeled, y_test, y_oracle = \
                    self.x_labeled.copy(), self.x_test.copy(), self.x_unlabeled.copy() \
                        , self.y_labeled.copy(), self.y_test.copy(), self.y_oracle.copy()
                for _ in range(self.iter_num):
                    # sample and check performance
                    iter_accuracy, iter_precision, x_labeled, y_labeled, x_unlabeled, y_oracle \
                        = simulate_al.sample_and_check_performance(x_labeled, x_test, x_unlabeled, y_labeled, y_test,
                                                                   y_oracle, num_queries, strategy)

                    precision.append(iter_precision)
                    accuracy.append(iter_accuracy)
                self.results[strategy]['accuracy'].append(np.mean(accuracy))
                self.results[strategy]['precision'].append(np.mean(precision))

        self.plot_results(plt_name)


def main():
    sampling_methods = {'random': (0, 0), 'average_kl_divergence': (1, None), 'entropy': (0, 1), 'max_margin': (0, 1),
                        'least_confident': (0, 1), 'centers_confidence': (0, 0), 'centers_distances': (0, 0)}

    # sampling_methods = {'centers':(0,0)}
    num_queries_list_acc = [50, 75, 100, 125, 250, 375, 500]
    if DEBUG:
        num_queries_list_acc = [50, 100, 200]
    num_queries_list = [num_queries_list_acc[0]]
    for ii in range(1, len(num_queries_list_acc)):
        num_queries_list += [num_queries_list_acc[ii] - num_queries_list_acc[ii - 1]]

    committee_number = 5
    iter_num = 10
    classifier_types = ['LR', 'SVC']

    simulation = Simulation(sampling_methods, num_queries_list, iter_num, data_name='MNIST original')

    for classifier_type in classifier_types:
        plt_name = 'MNIST {0} iter_num:{1} classifier_type:{2} committee_number:{3}.jpg'.format('all', iter_num,
                                                                                                classifier_type,
                                                                                                committee_number)
        simulation.simulate(plt_name=plt_name, classifier_type=classifier_type, committee_number=committee_number)


if __name__ == '__main__':
    main()
