# USAGE
# python distributed_inference.py

# import the necessary packages
from pyimagesearch.food_classifier import FoodClassifier
from pyimagesearch import config
from pyimagesearch import create_dataloaders
from torchvision import models
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
import torch

# determine the number of GPUs we have
NUM_GPU = torch.cuda.device_count()
print(f"[INFO] number of GPUs found: {NUM_GPU}...")

# determine the batch size based on the number of GPUs
BATCH_SIZE = config.PRED_BATCH_SIZE * NUM_GPU
print(f"[INFO] using a batch size of {BATCH_SIZE}...")

# define augmentation pipeline
testTransform = transforms.Compose([
	transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.MEAN, std=config.STD)
])

# calculate the inverse mean and standard deviation
invMean = [-m/s for (m, s) in zip(config.MEAN, config.STD)]
invStd = [1/s for s in config.STD]

# define our de-normalization transform
deNormalize = transforms.Normalize(mean=invMean, std=invStd)

# create test data loader
(testDS, testLoader) = create_dataloaders.get_dataloader(config.TEST,
	transforms=testTransform, bs=BATCH_SIZE, shuffle=True)

# load up the DenseNet121 model
baseModel = models.densenet121(pretrained=True)

# initialize our food classifier
model = FoodClassifier(baseModel, len(testDS.classes))

# load the model state
model.load_state_dict(torch.load(config.MODEL_PATH))

# if we have more than one GPU then parallelize the model
if NUM_GPU > 1:
	model = nn.DataParallel(model)

# move the model to the device and set it in evaluation mode
model.to(config.DEVICE)
model.eval()

# grab a batch of test data
batch = next(iter(testLoader))
(images, labels) = (batch[0], batch[1])

# initialize a figure
fig = plt.figure("Results", figsize=(10, 10 * NUM_GPU))

# switch off autograd
with torch.no_grad():
	# send the images to the device
	images = images.to(config.DEVICE)

	# make the predictions
	preds = model(images)

	# loop over all the batch
	for i in range(0, BATCH_SIZE):
		# initalize a subplot
		ax = plt.subplot(BATCH_SIZE, 1, i + 1)

		# grab the image, de-normalize it, scale the raw pixel
		# intensities to the range [0, 255], and change the channel
		# ordering from channels first tp channels last
		image = images[i]
		image = deNormalize(image).cpu().numpy()
		image = (image * 255).astype("uint8")
		image = image.transpose((1, 2, 0))

		# grab the ground truth label
		idx = labels[i].cpu().numpy()
		gtLabel = testDS.classes[idx]

		# grab the predicted label
		pred = preds[i].argmax().cpu().numpy()
		predLabel = testDS.classes[pred]

		# add the results and image to the plot
		info = "Ground Truth: {}, Predicted: {}".format(gtLabel,
			predLabel)
		plt.imshow(image)
		plt.title(info)
		plt.axis("off")

	# show the plot
	plt.tight_layout()
	plt.show()