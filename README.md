Active Learning Tool for MNIST

Framework to test perfomence for MNIST using differnet query algorithms


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

By Osher Arbib
