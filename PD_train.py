from options import PD_train_options, test_options
from dataloader import data_loader
from model import create_model
from util import visualizer
import time

if __name__=='__main__':
    # get testing options
    opt_feed = test_options.TestOptions().parse()
    # creat a dataset
    dataset = data_loader.dataloader(opt_feed)
    dataset_size = len(dataset) * opt_feed.batchSize
    # create models
    model_pic = create_model(opt_feed)
    model_pic.eval()
    opt = PD_train_options.PD_trainOptions().parse()
    opt.model = 'pdgan'
    model_pd = create_model(opt)
    model_pd.eval()
    # create a visualizer
    visualizer = visualizer.Visualizer(opt)
    keep_training = True
    epoch = 0
    total_iteration = opt.iter_count

    while(keep_training):
        epoch_start_time = time.time()
        epoch += 1
        print('\n Training epoch: %d' % epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_iteration += 1
            model_pic.set_input(data)
            img_p, mask = model_pic.feed()
            model_pd.pd_optimize_parameters(img_p, mask)
            model_pd.update_learning_rate()
