import torch
import torchvision.transforms as T


def logTensorImagesViaWandbLogger(tensor_images, wandb_logger, label):
    wandb_logger.log_image(key=label,
                           images=[T.ToPILImage()(img_tensor) for img_tensor in tensor_images])

def getImagesForVisualBenchmarking(data_module, wandb_logger):
	synth_for_visual_benchmarking = None
	real_for_visual_benchmarking = None
	for i in data_module.train_dataloader():
			synth, real = i
			logTensorImagesViaWandbLogger(synth, wandb_logger, "train_synth")
			logTensorImagesViaWandbLogger(real, wandb_logger, "train_real")
			synth_for_visual_benchmarking = synth
			real_for_visual_benchmarking = real
			# use only the first batch
			break
	for i in data_module.val_dataloader():
			synth, real = i
			logTensorImagesViaWandbLogger(synth, wandb_logger, "val_synth")
			logTensorImagesViaWandbLogger(real, wandb_logger, "val_real")
			synth_for_visual_benchmarking = torch.cat(
					[synth, synth_for_visual_benchmarking])
			real_for_visual_benchmarking = torch.cat(
					[real, real_for_visual_benchmarking])
			# use only the first batch
			break
	return synth_for_visual_benchmarking, real_for_visual_benchmarking