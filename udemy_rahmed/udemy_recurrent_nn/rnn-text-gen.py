import tensorflow as tf

import numpy as np
import os
import time



if __name__ == "__main__":
    args, unknown = _parse_args() 
    train_data, train_labels, eval_data, eval_labels = _load_data(args.train) 
    
    mnist_classifier = model(train_data, train_labels, eval_data, eval_labels)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        mnist_classifier.save(os.path.join(args.sm_model_dir, '000000002'), 'mnist_model.h5')