from options import test_options
from dataloader import data_loader
from model import create_model
from util import visualizer
from itertools import islice

n = 2
if __name__ == '__main__':
    if n == 1:
        opt = test_options.TestOptions().parse()
        opt.nsampling = 2
        opt.img_file = '/home/lazy/celeba_hq/val'
        opt.how_many = 2000
        # creat a dataset
        dataset = data_loader.dataloader(opt)
        dataset_size = len(dataset) * opt.batchSize
        print('testing images = %d' % dataset_size)
        # create a model
        model = create_model(opt)
        model.eval()
        # create a visualizer
        visualizer = visualizer.Visualizer(opt)

        for i, data in enumerate(islice(dataset, opt.how_many)):
            model.set_input(data)
            model.test()
    elif n == 2:
        opt = test_options.TestOptions().parse()
        opt.nsampling = 5
        opt.img_file = '/home/lazy/my-Pluralistic-Inpainting/to_test'
        opt.how_many = 5
        # creat a dataset
        dataset = data_loader.dataloader(opt)
        dataset_size = len(dataset) * opt.batchSize
        print('testing images = %d' % dataset_size)
        # create a model
        model = create_model(opt)
        model.eval()
        # create a visualizer
        visualizer = visualizer.Visualizer(opt)

        for i, data in enumerate(islice(dataset, opt.how_many)):
            model.set_input(data)
            model.test()
