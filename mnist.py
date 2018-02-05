from __future__ import unicode_literals, division

import numpy as np
from Simulation import Simulation
from configure import DEBUG



def main():
    '''
    inspired by: https://github.com/davefernig/alp
    active learning framework:
    arguments:
        sampling_methods
            'random'
            'average_kl_divergence'
            'entropy'
            'max_margin'
            'least_confident'
            'centers_confidence'
            'centers_distances':
            centers_confidence_pca'
            'centers_distances_pca'
        iter_num
        num_queries_list
        classifier_types
            LR - logistic regression ('soft max')
            SVC - Support vector classifier
            MLP - Fully connected (one hiden layer)
            CNN - 2 Conv. layeers (3x3 kernel, stride 1, relu), max-poling, 1 Fully connected ('relu'), soft-max layer
    '''


    sampling_methods = {'random': (0, 0),'average_kl_divergence': (1, None), 'least_confident': (0, 1),
                        'centers_confidence_pca': (0, 0),
                        }

    num_queries_list_acc = [int(ii) for ii in np.linspace(50,500,4)]
    if DEBUG:
        num_queries_list_acc = [50, 100, 200]

    num_queries_list = [50,num_queries_list_acc[0]]
    for ii in range(1, len(num_queries_list_acc)):
        num_queries_list += [num_queries_list_acc[ii] - num_queries_list_acc[ii - 1]]

    committee_numbers = [5,10,21]
    iter_num = 5
    classifier_types = ['LR','CNN']

    if DEBUG:
        classifier_types = ['LR']

    simulation = Simulation(sampling_methods, num_queries_list, iter_num, data_name='MNIST original')

    for classifier_type in classifier_types:
        for committee_number in committee_numbers:
            plt_name = 'MNIST {0} iter_num:{1} classifier_type:{2} committee_number:{3}{4}.jpg'.format('all', iter_num,
                                                                                                classifier_type,
                                                                                                committee_number,' Debug'*DEBUG)
        simulation.simulate(plt_name=plt_name, classifier_type=classifier_type, committee_number=committee_number)


if __name__ == '__main__':
    main()
