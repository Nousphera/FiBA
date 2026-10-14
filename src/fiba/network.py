# Modified for the FiBA standalone release, 2026-10-15; see PROVENANCE.md.
import torch
import torchmetrics
import sys
from sklearn.metrics import accuracy_score, confusion_matrix
import logging
from flwr.common.logger import log
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

class LambdaLayer(torch.nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(torch.nn.Module):
	expansion = 1
	def __init__(self, in_planes, planes, stride=1, option='A'):
		super(BasicBlock, self).__init__()
		self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = torch.nn.BatchNorm2d(planes)
		self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = torch.nn.BatchNorm2d(planes)
		self.shortcut = torch.nn.Sequential()

		if stride != 1 or in_planes != planes:
			if option == 'A':
				self.shortcut = LambdaLayer(lambda x: torch.nn.functional.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
			elif option == 'B':
				self.shortcut = torch.nn.Sequential(torch.nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), torch.nn.BatchNorm2d(self.expansion * planes))

	def forward(self, x):
		out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = torch.nn.functional.relu(out)
		return out

class ResNet(torch.nn.Module):
	def __init__(self, num_classes=10):
		super(ResNet, self).__init__()
		self.in_planes = 64
		self.num_blocks = [2,2,2,2]
		self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = torch.nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(BasicBlock, 64, self.num_blocks[0], stride=1)
		self.layer2 = self._make_layer(BasicBlock, 128, self.num_blocks[1], stride=2)
		self.layer3 = self._make_layer(BasicBlock, 256, self.num_blocks[2], stride=2)
		self.layer4 = self._make_layer(BasicBlock, 512, self.num_blocks[3], stride=2)
		self.linear = torch.nn.Linear(512, num_classes)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []

		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return torch.nn.Sequential(*layers)

	def forward(self, x):
		out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = torch.nn.functional.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out

class Model(torch.nn.Module):

	def __init__(self, num_classes=10):
		super(Model, self).__init__()
		self.network = ResNet(num_classes=num_classes)
		self.num_classes = num_classes

	@property
	def dtype(self):
		return self.network.weight.dtype

	@property
	def device(self):
		return next(self.parameters()).device

	def forward(self, x):
		return self.network(x)

	def compile(self, lr=1e-2):

		self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
		self.metric = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes).to(self.device)

	def calculate_asr(self, predicted_labels, target_labels):
		if len(predicted_labels) != len(target_labels):
			raise ValueError("Input lists must have the same length.")

		correct_count = 0

		for pred, target in zip(predicted_labels, target_labels):
			if pred == target:
				correct_count += 1

		accuracy = correct_count / len(predicted_labels)
		return accuracy


	def fit(self, ds, epochs=1, verbose=False):
		self.metric.reset()
		self.train()
		score = []

		for epoch in range(epochs):
			train_loss = 0.0
			for _, (x, y) in enumerate(ds):
				x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True).long()
				self.optimizer.zero_grad()
				logits = self(x)
				y_prob = torch.nn.functional.log_softmax(logits, dim=1)
				_loss = torch.nn.functional.nll_loss(y_prob, y)
				_loss.backward()
				self.optimizer.step()
				train_loss += _loss.item()
				y_preds = torch.argmax(y_prob, dim=1)
				self.metric(y_preds, y)
			train_loss /= len(ds)
			score.append(train_loss)
			acc = self.metric.compute()
		return {'loss': score, 'accuracy': acc}

	def fit_mask(self, ds, epochs=1, verbose=False, mask=None):
		self.metric.reset()
		self.train()
		score = []

		for epoch in range(epochs):
			train_loss = 0.0
			for _, (x, y) in enumerate(ds):
				x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True).long()
				self.optimizer.zero_grad()
				logits = self(x)
				y_prob = torch.nn.functional.log_softmax(logits, dim=1)
				_loss = torch.nn.functional.nll_loss(y_prob, y)
				_loss.backward()
				for name, param in self.named_parameters():
					param.grad.data = mask[name].to(self.device) * param.grad.data
				self.optimizer.step()
				train_loss += _loss.item()
				y_preds = torch.argmax(y_prob, dim=1)
				self.metric(y_preds, y)
			train_loss /= len(ds)
			score.append(train_loss)
			acc = self.metric.compute()
			if verbose:
				print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {100. * acc:.2f}%")
		return {'loss': score, 'accuracy': acc}

	def evaluate(self, ds, attack_setting, poisoned_eval, verbose=False):
		self.metric.reset()
		self.eval()
		test_loss = 0.0

		all_preds = []
		all_labels = []

		all_poisoned = []
		all_poisoned_labels = []

		if poisoned_eval:

			with torch.no_grad():
				for _, (x,y) in enumerate(poisoned_eval):
					# poison x
					poisoned_x = []
					poisoned_y = []
					# TODO remove targeted attack code
					if attack_setting.targeted:
						for i in range(len(x)):
							if y[i] == attack_setting.source_class:
								# Apply the custom function to the individual image
								modified_image = attack_setting.embed_trigger_in_image([x[i]], [y[i]], attack_setting, mixup=False)
								# Predict the label for the modified image
								poisoned_x.append(modified_image[0][0])
								poisoned_y.extend([attack_setting.target_class])
					else:
						for i in range(len(x)):
							if y[i] != attack_setting.target_class:
								# Apply the custom function to the individual image
								poisoned_x.append(x[i])
								poisoned_y.extend([attack_setting.target_class])

					if len(poisoned_x) > 0:
						poisoned_x, poisoned_y = torch.stack(poisoned_x, dim=0), torch.tensor(poisoned_y)
						poisoned_x, poisoned_y = poisoned_x.to(self.device, non_blocking=True), poisoned_y.to(self.device, non_blocking=True).long()
						modified_images_logits = self(poisoned_x)
						modified_images_prob = torch.nn.functional.log_softmax(modified_images_logits, dim=1)
						modified_images_preds = torch.argmax(modified_images_prob, dim=1)
						all_poisoned.extend(modified_images_preds.cpu().numpy())
						all_poisoned_labels.extend(poisoned_y)

		with torch.no_grad():
			for _, (x, y) in enumerate(ds):
				x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True).long()
				logits = self(x)
				y_prob = torch.nn.functional.log_softmax(logits, dim=1)
				test_loss += torch.nn.functional.nll_loss(y_prob, y).item()
				y_preds = torch.argmax(y_prob, dim=1)
				all_preds.extend(y_preds.cpu().numpy())
				all_labels.extend(y.cpu().numpy())
				self.metric(y_preds, y)


		attack_succes_rate = 0
		if poisoned_eval:
			attack_succes_rate = self.calculate_asr(all_poisoned, all_poisoned_labels)

		test_loss /= len(ds)
		acc = self.metric.compute()

		if verbose:
			print(f"Loss: {test_loss:.4f} - Accuracy: {100. * acc:.2f}%")

		 # Calculate the confusion matrix
		confusion = confusion_matrix(all_labels, all_preds)

		# Calculate accuracy per class
		accuracy_per_class = confusion.diagonal() / confusion.sum(axis=1)
		return (test_loss, float(acc.detach().cpu().numpy()), accuracy_per_class, attack_succes_rate)

	def extract_embeddings(self, ds):
		return_nodes = {'network.view': 'network.view'}
		feature_extractor = create_feature_extractor(self, return_nodes=return_nodes)
		embeddings = []
		self.metric.reset()
		self.eval()

		with torch.no_grad():
			for _, (x, y) in enumerate(ds):
				x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True).long()
				features = feature_extractor(x)['network.view']
				embeddings.extend(torch.tensor(features))

		return torch.stack(embeddings)

	def get_activation_map(self, ds):
		self.eval()
		cam = SmoothGradCAMpp(self, input_shape=(3, 32, 32), target_layer='network.layer1')
		activation_mask = torch.zeros(32, 32).to(self.device)

		for _, (x, y) in enumerate(ds):
			x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True).long()
			for image in x:
				logits = self(image.unsqueeze(0))

				# Generate activation maps
				activation_maps = cam(logits.squeeze(0).argmax().item(), logits)
				activation_mask += activation_maps[0].squeeze(0)

		sorted_indices = torch.argsort(activation_mask.flatten(), descending=True)

		top_half_mask = torch.zeros_like(sorted_indices, dtype=torch.float)
		top_half_mask[sorted_indices[:len(sorted_indices)//2]] = 1.0
		top_half_mask = top_half_mask.view(activation_mask.shape)

		fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Adjust figsize as needed

		# Show images on subplots
		axes[0].imshow(to_pil_image(top_half_mask))
		axes[0].set_title('Activation Mask')
		axes[0].axis('off')
		plt.savefig(f'activation maps', bbox_inches='tight')

		return top_half_mask


	def __str__(self):
		return self.__class__.__name__

	def screen_gradients(self, ds):
		self.train()
		gradient = {name: 0 for name, param in self.named_parameters()}
		for _, (x, y) in enumerate(ds):
			self.optimizer.zero_grad()
			x, y = x.to(self.device), y.to(self.device)
			logits = self(x)
			y_prob = torch.nn.functional.log_softmax(logits, dim=1)
			_loss = torch.nn.functional.nll_loss(y_prob, y)
			_loss.backward()
			for name, param in self.named_parameters():
				gradient[name] += param.grad.data
		return gradient
