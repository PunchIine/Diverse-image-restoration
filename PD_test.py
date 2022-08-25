from options import PD_test_options, test_options
from dataloader import data_loader
from model import create_model
from itertools import islice

if __name__ == "__main__":
    # get testing options
    opt_feed = test_options.TestOptions().parse()
    # creat a dataset
    opt_feed.nsampling = 2
    dataset = data_loader.dataloader(opt_feed)
    dataset_size = len(dataset) * opt_feed.batchSize
    # create models
    model_pic = create_model(opt_feed)
    model_pic.eval()
    opt = PD_test_options.PD_TestOptions().parse()
    opt.nsampling = 1
    # creat a dataset
    dataset = data_loader.dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print("testing images = %d" % dataset_size)
    # create a model
    model = create_model(opt)
    model.eval()
    for i, data in enumerate(islice(dataset, opt.how_many)):
        model_pic.set_input(data)
        img_ps = []
        for _ in range(opt_feed.nsampling):
            img_p, mask, img_truth = model_pic.feed()
            img_ps.append(img_p)
        model.set_input(data)
        num = 0
        for img_p in img_ps:
            num = model.test(img_p, num)
