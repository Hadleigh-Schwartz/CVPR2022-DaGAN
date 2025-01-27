from tqdm import trange
import torch
import cv2

from torch.utils.data import DataLoader

from logger import Logger
from modules.model_dataparallel import DiscriminatorFullModel
import modules.model_dataparallel as MODEL
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
import pdb
from sync_batchnorm import DataParallelWithCallback
from evaluation_dataset import EvaluationDataset

from frames_dataset import DatasetRepeater


class EarlyStopper:
    def __init__(self, patience=1, min_delta=1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss * self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids,opt,writer):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    
    f_before = open("before.txt", "w")
    f_before.write(" ".join([str(x) for x in list(generator.parameters())[:10]]))
    f_before.close()

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                      optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector)
    else:
        start_epoch = 0
    
    f_after = open("after.txt", "w")
    f_after.write(" ".join([str(x) for x in list(generator.parameters())[:10]]))
    f_after.close()
    

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=16,drop_last=True)

    
    generator_full = getattr(MODEL,opt.GFM)(kp_detector, generator, discriminator, train_params,opt)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)
    test_dataset = EvaluationDataset(dataroot='video-preprocessing/vox',pairs_list='data/vox_evaluation_v3.csv')
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = 1,
            shuffle=False,
            num_workers=4)
    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        early_stopper = EarlyStopper(patience=3, min_delta=1.1)
        for epoch in trange(start_epoch, train_params['num_epochs']):
            #parallel
            total = len(dataloader)
            epoch_train_loss = 0
            generator.train(), discriminator.train(), kp_detector.train()
            with tqdm(total=total) as par:
                for i,x in enumerate(dataloader):
                    # x['source'] = x['source'].to(device)
                    # x['driving'] = x['driving'].to(device)
                    if i % 100 == 0: # gen vis evry 100 batches
                        losses_generator, generated = generator_full(x, gen_vis=True)

                        #log the visualizations
                        for g in range(len(generated["generated_vis"])):
                            # cv2.imwrite(f"gen_{total*(epoch-start_epoch) + i*config["train_params"]["batch_size"] + g}.png", generated["generated_vis"][g])
                            # cv2.imwrite(f"mp_{total*(epoch-start_epoch) +  i*config["train_params"]["batch_size"] + g}.png", generated["mp_vis"][g])
                            writer.add_image(f"Generated/{g}", generated["generated_vis"][g][:,:,::-1],
                                 dataformats="HWC", global_step=total*(epoch-start_epoch) + i*config["train_params"]["batch_size"])
                            writer.add_image(f"MediaPipe/{g}", generated["mp_vis"][g][:,:,::-1],
                                 dataformats="HWC", global_step=total*(epoch-start_epoch) + i*config["train_params"]["batch_size"])
                        
                    else:
                        losses_generator, generated = generator_full(x, gen_vis=False)

              
                    loss_values = [val.mean() for val in losses_generator.values()]
                    loss = sum(loss_values)
                    loss.backward()
                    optimizer_generator.step()
                    optimizer_generator.zero_grad()
                    optimizer_kp_detector.step()
                    optimizer_kp_detector.zero_grad()
                    epoch_train_loss+=loss.item()

                    if train_params['loss_weights']['generator_gan'] != 0:
                        optimizer_discriminator.zero_grad()
                        losses_discriminator = discriminator_full(x, generated)
                        loss_values = [val.mean() for val in losses_discriminator.values()]
                        loss = sum(loss_values)

                        loss.backward()
                        optimizer_discriminator.step()
                        optimizer_discriminator.zero_grad()
                    else:
                        losses_discriminator = {}

                    losses_generator.update(losses_discriminator)
                    losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                    for k,v in losses.items():
                        writer.add_scalar(k, v, total*epoch+i)
                    logger.log_iter(losses=losses)
                    par.update(1)
            epoch_train_loss = epoch_train_loss/total
            if (epoch + 1) % train_params['checkpoint_freq'] == 0:
                writer.add_scalar('epoch_train_loss', epoch_train_loss, epoch)
            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector}, inp=x, out=generated)
            generator.eval(), discriminator.eval(), kp_detector.eval()
            if (epoch + 1) % train_params['checkpoint_freq'] == 0:
                epoch_eval_loss = 0
                for i, data in tqdm(enumerate(test_dataloader)):
                    if i > 200: # due to training time constraints, we will make the test dataset a bit smaller
                        break
                    data['source'] = data['source'].cuda()
                    data['driving'] = data['driving'].cuda()
                    losses_generator, generated = generator_full(data) 
                    loss_values = [val.mean() for val in losses_generator.values()]
                    loss = sum(loss_values)
                    epoch_eval_loss+=loss.item()
                epoch_eval_loss = epoch_eval_loss/len(test_dataloader)
                writer.add_scalar('epoch_eval_loss', epoch_eval_loss, epoch)
            if early_stopper.early_stop(epoch_eval_loss):
                print("Early stopping")
                break