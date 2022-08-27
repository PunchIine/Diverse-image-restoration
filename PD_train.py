from options import PD_train_options, test_options
from dataloader import data_loader
from model import create_model
from util import visualizer
import time
import torch

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    # get testing options
    opt_feed = test_options.TestOptions().parse()
    # creat a dataset
    opt_feed.batchSize = 2
    dataset = data_loader.dataloader(opt_feed)
    dataset_size = len(dataset) * opt_feed.batchSize
    # create models
    model_pic = create_model(opt_feed)
    model_pic.eval()
    opt = PD_train_options.PD_trainOptions().parse()
    opt.batchSize = 2
    opt.model = "pdgan"
    #    opt.continue_train = True
    model_pd = create_model(opt)
    model_pd.eval()
    # create a visualizer
    visualizer = visualizer.Visualizer(opt)
    keep_training = True
    max_iteration = opt.niter + opt.niter_decay
    epoch = 0
    total_iteration = opt.iter_count

    while keep_training:
        epoch_start_time = time.time()
        epoch += 1
        print("\n Training epoch: %d" % epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_iteration += 1
            model_pic.set_input(data)
            img_p, mask, img_truth = model_pic.feed()
            model_pd.set_input(data)
            model_pd.pd_optimize_parameters(img_p)

            # display images on visdom and save images
            if total_iteration % opt.display_freq == 0:
                visualizer.display_current_results(
                    model_pd.get_current_visuals(), epoch
                )

            # print training loss and save logging information to the disk
            if total_iteration % opt.print_freq == 0:
                losses = model_pd.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, total_iteration, losses, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(total_iteration, losses)

            # save the latest model_pd every <save_latest_freq> iterations to the disk
            if total_iteration % opt.save_latest_freq == 0:
                print(
                    "saving the latest model_pd (epoch %d, total_steps %d)"
                    % (epoch, total_iteration)
                )
                model_pd.save_networks("latest")

            # save the model_pd every <save_iter_freq> iterations to the disk
            if total_iteration % opt.save_iters_freq == 0:
                print("saving the model_pd of iterations %d" % total_iteration)
                model_pd.save_networks(total_iteration)

            if total_iteration > max_iteration:
                keep_training = False
                break

        model_pd.update_learning_rate()
