import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from ActiveLearning import ActiveLearning
from configure import DEBUG
import time

class Simulation(object):
    def __init__(self, sampling_methods, num_queries_list, iter_num, data_name):
        self.sampling_methods, self.num_queries_list, self.iter_num = sampling_methods, num_queries_list, iter_num
        self.x_labeled, self.x_test, self.x_unlabeled, self.y_labeled, self.y_test, self.y_oracle = self.init_dataset(
            data_name)

    def plot_results(self, plt_name):
        num_queries_list = np.cumsum(self.num_queries_list)
        styles = ['-','--','-.']
        for ii, strategy in enumerate(self.sampling_methods):
            if strategy != 'random':
                color_b = 1.0 * ii / len(self.sampling_methods)
                color_g = 1 - color_b
                style = styles[ii%len(styles)]
            else:
                color_b = 0.0
                color_g = 0.0
                style = '-'
            plt.subplot(131)
            plt.plot(num_queries_list, self.results[strategy]['accuracy'], color=(0.0, color_g, color_b),linestyle=style)
            plt.subplot(133)
            plt.plot(num_queries_list, self.results[strategy]['precision'], color=(0.0, color_g, color_b),linestyle=style)
        plt.subplot(131)
        plt.ylabel('Accuracy')
        plt.xlabel('Number of Queries')
        plt.subplot(133)
        plt.ylabel('Precision')
        plt.xlabel('Number of Queries')

        plt.subplot(132)
        for ii, strategy in enumerate(self.sampling_methods):
            if strategy != 'random':
                color_b = 1.0 * ii / len(self.sampling_methods)
                color_g = 1 - color_b
                style = styles[ii%len(styles)]
            else:
                color_b = 0.0
                color_g = 0.0
                style = '-'
            plt.plot([_*0 for _ in self.results[strategy]['accuracy']], color=(0.0, color_g, color_b), linestyle=style)
        plt.legend([''.join([_[0] for _ in ii.replace('_', ' ').split()]).upper()*(len(ii)>10)+
                    ii.replace('_', ' ').title()*(len(ii)<=10) for ii in self.sampling_methods], loc=4)

        # plt.legend([ii.replace('_', ' ').capitalize() for ii in self.sampling_methods], loc=4)

        plt.ylim([0,1])
        plt.title('MNIST')
        plt.savefig("./res/"+plt_name)
        plt.close()

    @staticmethod
    def init_dataset(data_name,data_home='./data'):
        dataset = fetch_mldata(data_name,data_home=data_home)
        x, y = dataset.data, dataset.target
        x = x.astype('float')
        x = (x - np.min(x)) / (np.max(x) - np.min(x))

        if DEBUG:
            idx_init = list(range(x.shape[0]))
            np.random.shuffle(idx_init)
            idx_init = idx_init[:5000]
            x, y = x[idx_init, :], y[idx_init]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        x_unlabeled, x_labeled, y_oracle, y_labeled = train_test_split(x_train, y_train, test_size=0)

        return x_labeled, x_test, x_unlabeled, y_labeled, y_test, y_oracle

    def simulate(self, plt_name, classifier_type, committee_number):
        simulate_al = ActiveLearning(classifier_type, committee_number, self.sampling_methods)
        self.results = {strategy: {'accuracy': [], 'precision': []} for strategy in self.sampling_methods}

        for strategy in self.sampling_methods:
            precision = [list() for _ in range(self.iter_num)]
            accuracy = [list() for _ in range(self.iter_num)]
            for _iter in range(self.iter_num):
                t0=time.time()
                x_labeled, x_test, x_unlabeled, y_labeled, y_test, y_oracle = \
                    self.x_labeled.copy(), self.x_test.copy(), self.x_unlabeled.copy() \
                        , self.y_labeled.copy(), self.y_test.copy(), self.y_oracle.copy()

                for num_queries in self.num_queries_list:
                    # sample and check performance
                    query_accuracy, query_precision, x_labeled, y_labeled, x_unlabeled, y_oracle \
                        = simulate_al.sample_and_check_performance(x_labeled, x_test, x_unlabeled, y_labeled, y_test,
                                                                   y_oracle, num_queries, strategy)

                    accuracy[_iter].append(query_accuracy)
                    precision[_iter].append(query_precision)
                print classifier_type,strategy,_iter,time.time()-t0
            self.results[strategy]['accuracy']=(np.mean(accuracy,axis=0))
            self.results[strategy]['precision']=(np.mean(precision,axis=0))


        self.plot_results(plt_name)

