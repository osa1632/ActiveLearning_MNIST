import numpy as np
from scipy.stats import entropy
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from mnist_cnn import  CNNClassifiear
import random



class ActiveLearning(object):
    def __init__(self, classifier_type, committee_number, sampling_methods):
        if classifier_type == 'LR':
            clf = [LogisticRegression()]
        elif classifier_type == 'SVC':
            # SVC params -
            # https://github.com/ksopyla/svm_mnist_digit_classification/blob/master/svm_mnist_classification.py
            clf = [SVC(probability=True, C=5, gamma=0.05)]
        elif classifier_type == 'MLP':
            # MLP Params:
            clf = [MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=1e-4,
                     tol=1e-4, learning_rate_init=.1,early_stopping=True)]
        elif classifier_type=='CNN':
            clf = [CNNClassifiear]
        else:
            raise NotImplementedError("classifier type should be: LR,MLP,SVC or CNN")

        clf += [LogisticRegression() for _ in range(committee_number)]  # for voting
        self.clf = clf
        self.sampling_methods = sampling_methods
        self.committee_number = committee_number

    def sampling(self,clf, strategy, x_unlabeled, x_labeled, y_labeled, num_queries):
        if x_labeled.shape[0] == 0:
            strategy = 'random'

        if strategy == 'random':
            idx = np.random.choice(x_unlabeled.shape[0], num_queries, replace=False)
        else:
            if strategy == 'least_confident':  # uncertainty -- least confident
                probs = clf[0].predict_proba(x_unlabeled)
                scores = 1 - np.amax(probs, axis=1)

            elif strategy == 'BvsSB':  # uncertainty -- Best vs Second Best
                probs = clf[0].predict_proba(x_unlabeled)
                probs.sort(axis=-1)
                scores = np.apply_along_axis(lambda x:x[-2]-x[-1], 1, probs)

            elif strategy == 'entropy':  # uncertainty -- entropy
                probs = clf[0].predict_proba(x_unlabeled)
                scores = np.apply_along_axis(entropy, 1, probs)

            elif strategy == 'average_kl_divergence':  # query_by_committee -- average_kl_divergence
                probs = []
                for model in clf:
                    model.fit(x_labeled, y_labeled)
                    probs.append(model.predict_proba(x_unlabeled))
                consensus = np.mean(np.stack(probs), axis=0)
                divergence = [entropy(consensus.T, y_out.T) for y_out in probs]
                scores = np.apply_along_axis(np.mean, 0, np.stack(divergence))


            elif strategy == 'centers_confidence':  # uncertainty -- least confident, using clustering, distance as measurement
                opt_centroids = [np.array([np.reshape(x,(784,-1)) for x,y in zip(x_labeled,y_labeled) if y == k]) for k in np.unique(y_labeled)]
                classes = []
                for _ in range(self.committee_number):
                    centroids = [np.mean(a[np.random.choice(a.shape[0],min(20,a.shape[0]),replace=False),:,:],axis=0) for a in opt_centroids]
                    classes.append(np.array(
                        [np.argmin([np.dot(((x_i - y_k).T), ((x_i - y_k).T))
                                    for y_k in centroids]) for e,x_i in enumerate(x_unlabeled)]))
                scores = np.apply_along_axis(lambda x:len(np.unique(x)),axis=0,arr=classes)

            elif strategy == 'centers_confidence_pca':  # uncertainty -- least confident, using clustering, distance as measurement using pca

                pca=ActiveLearning.get_pca(x_labeled, x_unlabeled)
                x_unlabeled_pca = [pca(x_i) for x_i in x_unlabeled]
                opt_centroids = [np.array([np.reshape(x,(784,-1)) for x,y in zip(x_labeled,y_labeled) if y == k]) for k in np.unique(y_labeled)]
                classes = []
                for _ in range(3):
                     centroids = [np.mean(pca(a[np.random.choice(a.shape[0],min(3,a.shape[0]),replace=False),:,:]),axis=1) for a in opt_centroids]
                     classes.append(
                         np.array([np.argmin(
                             [np.dot((x_i - np.reshape(y_k,(-1,))).T, (x_i - np.reshape(y_k,(-1,)))) for y_k in centroids])
                                   for e,x_i in enumerate(x_unlabeled_pca)]))
                scores = np.apply_along_axis(lambda x:-len(np.unique(x)),axis=0,arr=classes)

            elif strategy == 'centers_distances':  # uncertainty -- least confident, using clustering, distance as measurement
                centroids = [x_labeled[y_labeled == k].mean(axis=0) for k in np.unique(y_labeled)]
                distances_x_unlabeld = np.array(
                    [[np.dot(x_i - y_k, x_i - y_k) for y_k in centroids] for x_i in x_unlabeled])
                scores = (np.min(distances_x_unlabeld, axis=1))

            elif strategy == 'centers_distances_pca':  # uncertainty -- least confident, using clustering, distance as measurement

                pca=ActiveLearning.get_pca(x_labeled, x_unlabeled)
                centroids = [x_labeled[y_labeled == k].mean(axis=0) for k in np.unique(y_labeled)]
                distances_x_unlabeld = np.array(
                    [[np.dot(pca((x_i - y_k).T), pca((x_i - y_k).T))
                                    for y_k in centroids] for x_i in x_unlabeled])
                scores = (np.min(distances_x_unlabeld, axis=1))

            elif strategy == 'centers_confidence_pca_least_confident':  # uncertainty -- least confident, using clustering, distance as measurement using pca

                pca=ActiveLearning.get_pca(x_labeled, x_unlabeled)
                x_unlabeled_pca = [pca(x_i) for x_i in x_unlabeled]
                opt_centroids = [np.array([np.reshape(x,(784,-1)) for x,y in zip(x_labeled,y_labeled) if y == k]) for k in np.unique(y_labeled)]
                classes = []
                for _ in range(3):
                     centroids = [np.mean(pca(a[np.random.choice(a.shape[0],min(3,a.shape[0]),replace=False),:,:]),axis=1) for a in opt_centroids]
                     classes.append(
                         np.array([np.argmin(
                             [np.dot((x_i - np.reshape(y_k,(-1,))).T, (x_i - np.reshape(y_k,(-1,)))) for y_k in centroids])
                                   for e,x_i in enumerate(x_unlabeled_pca)]))
                scores = np.apply_along_axis(lambda x:len(np.unique(x)),axis=0,arr=classes)
                idx = np.argsort(scores) [:num_queries*2]

                probs = clf[0].predict_proba(x_unlabeled[idx,:])
                # scores = np.apply_along_axis(entropy, 1, probs)
                scores = 1 - np.amax(probs, axis=1)

            idx = np.argsort(-scores)  # reversed
        return idx[:num_queries]

    @staticmethod
    def get_pca(x_labeled, x_unlabeled):
        if not hasattr(ActiveLearning.get_pca, 'pca'):
            x = np.concatenate((x_labeled, x_unlabeled)).T
            xmean = np.mean(x)
            x = [a - xmean for a in x]
            u, _, _ = np.linalg.svd(x, full_matrices=False)
            u_pca = (np.transpose(u)[0:50])
            del u
            ActiveLearning.get_pca.pca = lambda data: np.dot(u_pca, data)
        return ActiveLearning.get_pca.pca

    def sample_and_check_performance(self, x_labeled, x_test, x_unlabeled, y_labeled, y_test, y_oracle,
                                     num_queries, strategy):
        main_model = self.clf[0]
        clf_ind = self.sampling_methods[strategy]

        idx = self.sampling(self.clf[clf_ind[0]:clf_ind[1]], strategy, x_unlabeled, x_labeled, y_labeled, num_queries)
        x_augmented = np.concatenate((x_labeled, x_unlabeled[idx, :]))
        y_augmented = np.concatenate((y_labeled, y_oracle[idx]))

        # print x_augmented.shape[0]

        main_model.fit(x_augmented, y_augmented)
        preds = main_model.predict(x_test)
        accuracy, precision, recall = self.multiclass_performance(y_test, preds, np.unique(y_test))

        x_unlabeled = np.delete(x_unlabeled, idx, axis=0)
        y_oracle = np.delete(y_oracle, idx)
        x_labeled, y_labeled = x_augmented, y_augmented

        return accuracy, precision, x_labeled, y_labeled, x_unlabeled, y_oracle

    @staticmethod
    def multiclass_performance(y, preds, classes):
        '''see: https://stackoverflow.com/questions/47899463/how-to-extract-false-positive-false-negative-from-a-confusion-matrix-of-multicl'''
        cfm = np.matrix([[np.sum(np.logical_and(y==c_y, preds==c_pred)) for c_pred in classes] for c_y in classes])
        tp = np.diag(cfm)

        fp = []
        for ii in classes.astype('uint8'):
            fp.append(np.sum(cfm[:, ii]) - cfm[ii,ii])
        fp = np.array(fp)

        fn = []
        for ii in classes.astype('uint8'):
            fn.append(np.sum(cfm[ii, :]) - cfm[ii,ii])
        fn = np.array(fn)

        tn = []
        for ii in classes.astype('uint8'):
            temp = np.delete(cfm, ii, 0)  # delete ith row
            temp = np.delete(temp, ii, 1)  # delete ith column
            tn.append(np.sum(np.sum(temp)))
        tn = np.array(tn)

        accuracy, precision, recall = np.mean((tp + tn) / np.mean(tp + tn + fp + fn + 0.01)), np.mean(tp / (tp + fp + 0.01)), \
                                      np.mean(tp / (tp + tn + 0.01))
        # print confusion_matrix(y, preds)

        return accuracy, precision, recall
