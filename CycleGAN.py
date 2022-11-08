import torch.nn.functional as F
import torchvision.transforms as T
from OriginalCycleGAN.Discriminators import PixelDiscriminator as Discriminator
from OriginalCycleGAN.Generators import Resnet as Generator 
from itertools import chain
import torch.nn as nn
import pytorch_lightning as pl
import torch


class CycleGAN(pl.LightningModule):
    def __init__(
            self,
            config: dict,
            wandb_logger,
            list_of_images_for_visual_benchmarking: list,
            **kwargs,
    ):
        super().__init__()
        bat = config["BATCH_SIZE"]
        self.config = config
        self.wandb_logger = wandb_logger
        self.list_of_images_for_visual_benchmarking = list_of_images_for_visual_benchmarking
        # networks
        self.synth_generator = Generator()
        self.real_generator = Generator()
        self.synth_discriminator = Discriminator()
        self.real_discriminator = Discriminator()
        self.l1_loss = nn.L1Loss()
        self.synth_label = torch.zeros(bat)
        self.real_label = torch.ones(bat)
        print("The pl.Module has been initialized")

    def forward(self, synth):
        return self.synth_generator(synth)

    @staticmethod
    def adversarial_loss(disc_output, synth=True):
        if synth:
            ground_truth = torch.zeros_like(disc_output)
        else:
            ground_truth = torch.ones_like(disc_output)
        return F.mse_loss(disc_output, ground_truth)
    
    @staticmethod
    def calcAccuracy(disc_preds, synth=True):
        # disc output is just cnn output - NOT A SINGLE VALUE
        mean_preds = disc_preds.flatten(2).mean(-1)
        thresholded_preds = mean_preds > 0.5
        acc = torch.count_nonzero(thresholded_preds) / thresholded_preds.size()[0]
        if synth:
            return 1-acc
        else:
            return acc


    def discrimTrainStep(self, discrim, is_it_a_synth_disc, real_batch, fake_batch):
        """
        Since I'm training a cycleGAN, it requires very similar steps to be taken on both discriminators.
        """
        name_used_for_logging = "synth_d" if is_it_a_synth_disc else "real_d"
        for param in discrim.parameters():
            param.requires_grad = True
        real_discrim_prediction = discrim(T.RandomRotation(180)(real_batch))
        real_loss = self.adversarial_loss(real_discrim_prediction, synth=is_it_a_synth_disc)
        real_acc = self.calcAccuracy(real_discrim_prediction, synth=is_it_a_synth_disc)
        self.log(f'{name_used_for_logging}_real_acc', real_acc)
        fake_discrim_prediction = discrim(T.RandomRotation(180)(fake_batch))
        fake_acc = self.calcAccuracy(fake_discrim_prediction, synth=(not is_it_a_synth_disc))
        self.log(f'{name_used_for_logging}_fake_acc', fake_acc)
        fake_loss = self.adversarial_loss(fake_discrim_prediction, synth=(not is_it_a_synth_disc))
        d_loss = real_loss + fake_loss
        self.log(f"{name_used_for_logging}_loss", d_loss)
        return d_loss


    def training_step(self, batch, batch_idx, optimizer_idx):
        real_batch, synth_batch = batch
        self.real_label = self.real_label.type_as(real_batch)
        self.synth_label = self.synth_label.type_as(synth_batch)
        # train generator
        if optimizer_idx == 0:
            for param in self.real_discriminator.parameters():
                param.requires_grad = False
            for param in self.synth_discriminator.parameters():
                param.requires_grad = False
            g_loss = 0
            # GANerator fooling discriminators
            generated_into_real = self.real_generator(synth_batch)
            generated_into_synth = self.synth_generator(real_batch)
            real_discriminator_pred = self.real_discriminator(T.RandomRotation(180)(generated_into_real))
            real_g_loss = self.adversarial_loss(real_discriminator_pred, synth=True)
            synth_discriminator_pred = self.synth_discriminator(T.RandomRotation(180)(generated_into_synth))
            synth_g_loss = self.adversarial_loss(synth_discriminator_pred, synth=False)
            g_loss += real_g_loss + synth_g_loss
            # identity loss
            idt_real = self.real_generator(real_batch)
            idt_synth = self.synth_generator(synth_batch)
            g_loss += self.l1_loss(idt_real, real_batch) * self.config['CYCLE_CONSISTENCY_WEIGHT'] * 0.5
            g_loss += self.l1_loss(idt_synth, synth_batch) * self.config['CYCLE_CONSISTENCY_WEIGHT'] * 0.5
            # Cycle consistency loss
            back_to_synth = self.synth_generator(generated_into_real)
            g_loss += self.l1_loss(synth_batch, back_to_synth) * self.config['CYCLE_CONSISTENCY_WEIGHT']
            back_to_real = self.real_generator(generated_into_synth)
            g_loss += self.l1_loss(real_batch, back_to_real) * self.config['CYCLE_CONSISTENCY_WEIGHT']
            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss

        # train real_discriminator
        if optimizer_idx == 1:
            real_batch = T.RandomRotation(180)(real_batch)
            fake_batch = T.RandomRotation(180)(self.real_generator(synth_batch))
            d_loss = self.discrimTrainStep(self.real_discriminator, is_it_a_synth_disc=False, real_batch=real_batch, fake_batch=fake_batch)
            return d_loss

        # train synth_discriminator
        if optimizer_idx == 2:
            real_batch = T.RandomRotation(180)(synth_batch)
            fake_batch = T.RandomRotation(180)(self.synth_generator(real_batch))
            d_loss = self.discrimTrainStep(self.synth_discriminator, is_it_a_synth_disc=True, real_batch=real_batch, fake_batch=fake_batch)
            return d_loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        real_batch, synth_batch = batch
        self.real_label = self.real_label.type_as(real_batch)
        self.synth_label = self.synth_label.type_as(synth_batch)
        fake_loss = self.adversarial_loss(self.real_discriminator(self.real_generator(synth_batch).detach()), synth=True)
        self.log("val_loss", fake_loss)

    def on_validation_epoch_end(self):
        z = self.list_of_images_for_visual_benchmarking.to(self.device)
        # log sampled images
        ganerated_images = self(z)
        list_of_ganerated_ims = [T.ToPILImage()(img_tensor) for img_tensor in ganerated_images]
        self.wandb_logger.log_image(key="generated_images", images=list_of_ganerated_ims)

    def configure_optimizers(self):
        lr = self.config['LEARNING_RATE']
        opt_gs = torch.optim.Adam(chain(self.synth_generator.parameters(), self.real_generator.parameters()), lr=lr)
        opt_real_d = torch.optim.Adam(self.real_discriminator.parameters(), lr=lr)
        opt_synth_d = torch.optim.Adam(self.synth_discriminator.parameters(), lr=lr)
        return [opt_gs, opt_real_d, opt_synth_d]

