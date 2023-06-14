import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np

from os import path, makedirs

# Inputs to our networks don't change much so turning on benchmark mode makes sense
torch.backends.cudnn.benchmark = True


class SigmoidMeanErrorScoreBalance:
    """ Keeps track of the number of training step to gradualy increase the weight of FPs relative to FNs """
    def __init__(self, initial_balance=0.1, final_balance=1.0, epochs=100000):
        self.epoch = 0
        self.min = initial_balance
        self.max = final_balance
        self.shift = epochs / 2
        self.stretch = epochs / 10
        self.weight = initial_balance

    def step(self):
        self.epoch += 1
        self.weight = float(np.clip(
            self.min + (self.max - self.min) / (1 + np.exp(-(self.epoch - self.shift) / self.stretch)),
            self.min, self.max
        ))


def dice_error_score(p, t, weights, epsilon=1e-8):
    pf = p.view(-1)
    tf = t.view(-1)
    wf = weights.view(-1)

    i = pf * tf
    n = pf + tf
    return 1 - 2 * (torch.sum(wf * i) + epsilon) / (torch.sum(wf * n) + epsilon)


def mean_error_score(p, t, weights, balance=0.1):
    """ Weighted sum of false positives and false negatives (probabilistic) """
    pf = p.view(-1)
    tf = t.view(-1)
    wf = weights.view(-1)

    fp = pf * (1 - tf)
    fn = (1 - pf) * tf

    return (wf * (balance * fp + fn)).sum() / p.size(0)


def binary_cross_entropy(p, t, epsilon=1e-8):
    """ Binary cross entropy for a single probability value per sample """
    pf = p.view(-1)
    tf = t.view(-1)

    return torch.mean(
        -tf * torch.log(torch.clamp(pf, min=epsilon, max=1)) -\
        (1 - tf) * torch.log(torch.clamp(1 - pf, min=epsilon, max=1))
    )


def mean_absolute_error(p, t):
    return F.l1_loss(p, t)


def mean_squared_error(p, t):
    return F.mse_loss(p, t)


class SoftmaxCrossEntropy:
    def __init__(self):
        self.class_weights = None

    def __call__(self, p, t, weights):
        if self.class_weights is None:
            w = [1] * 25
            w[0] = 0.1
            self.class_weights = torch.tensor(w, dtype=p.dtype, device=p.device)

        ce = F.cross_entropy(p, t.long(), weight=self.class_weights, reduction='none')
        return torch.mean(weights * ce)


def conv3x3(n_in, n_out):
    return [
        nn.Conv3d(n_in, n_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(n_out, affine=True, track_running_stats=False),
        nn.ReLU(inplace=True)
    ]


class Flatten(nn.Module):
    def forward(self, y):
        return y.view(y.size(0), -1)


class ContractionBlock(nn.Module):
    def __init__(self, n_input_channels, n_filters, dropout=None, pooling=True):
        super().__init__()

        layers = []
        if pooling:
            layers.append(nn.MaxPool3d(kernel_size=2))
        layers += conv3x3(n_input_channels, n_filters)
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        layers += conv3x3(n_filters, n_filters)
        self.pool_conv = nn.Sequential(*layers)

    def forward(self, incoming):
        return self.pool_conv(incoming)


class ExpansionBlock(nn.Module):
    def __init__(self, n_input_channels, n_filters, dropout=None):
        super(ExpansionBlock, self).__init__()

        self.upconv = nn.Sequential(
            nn.ConvTranspose3d(n_input_channels, n_filters, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(n_filters, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

        layers = conv3x3(n_filters * 2, n_filters)
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        layers += conv3x3(n_filters, n_filters)
        self.conv = nn.Sequential(*layers)

    def forward(self, incoming, skip_connection):
        y = self.upconv(incoming)
        y = torch.cat([y, skip_connection], dim=1)
        return self.conv(y)


class UNet(nn.Module):
    def __init__(self, n_input_channels, n_filters, n_output_channels, dropout=None, sigmoid=True):
        super().__init__()

        # Build contraction path
        self.contraction = nn.ModuleList()
        for i in range(1, 5):
            n_in = n_filters if i > 1 else n_input_channels
            self.contraction.append(ContractionBlock(n_in, n_filters, dropout=dropout if i > 1 else None, pooling=i > 1))

        # Build expansion path
        self.expansion = nn.ModuleList()
        for i in range(1, 4):
            self.expansion.append(ExpansionBlock(n_filters, n_filters, dropout=dropout))

        output_layer = nn.Conv3d(in_channels=n_filters, out_channels=n_output_channels, kernel_size=1)
        if sigmoid:
            self.segmentation = nn.Sequential(output_layer, nn.Sigmoid())
        else:
            self.segmentation = output_layer

    def forward(self, image):
        y = image

        # Pass image through contraction path
        cf = []
        for contract in self.contraction:
            y = contract(y)
            cf.append(y)

        # Pass features through expansion path
        for expand, features in zip(self.expansion, reversed(cf[:-1])):
            y = expand(y, features)

        # Collect final output
        segmentation = self.segmentation(y)
        features = cf[-1]

        return segmentation, features

    def nth_activation_map(self, n, image):
        assert n > 0

        y = image
        i = 0

        # Pass image through contraction path
        cf = []
        for contract in self.contraction:
            y = contract(y)
            i += 1
            if i == n:
                break
            cf.append(y)

        if i < n:
            # Pass features through expansion path
            for expand, features in zip(self.expansion, reversed(cf[:-1])):
                y = expand(y, features)
                i += 1
                if i == n:
                    break

        y = F.upsample(y, size=128, mode='trilinear', align_corners=False)
        return self.segmentation(y)


class MemoryUNet(nn.Module):
    def __init__(self, n_input_channels, n_filters, n_output_channels, **kwargs):
        super().__init__()
        self.unet = UNet(n_input_channels, n_filters, n_output_channels, **kwargs)

    def forward(self, image, memory):
        x = torch.cat([image, memory], dim=1)
        segmentation, features = self.unet.forward(x)
        return segmentation

    def nth_activation_map(self, n, image, memory):
        x = torch.cat([image, memory], dim=1)
        return self.unet.nth_activation_map(n, x)


class YLeg(nn.Module):
    def __init__(self, n_input_channels, n_filters, n_output_channels, activation):
        super().__init__()

        if activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation == 'relu':
            activation = nn.ReLU()
        else:
            raise ValueError('Unknown activation function "{}"'.format(activation))

        self.classification = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            *conv3x3(n_input_channels, n_filters),
            nn.MaxPool3d(kernel_size=2),
            *conv3x3(n_filters, n_filters),
            nn.MaxPool3d(kernel_size=2),
            *conv3x3(n_filters, n_filters),
            nn.MaxPool3d(kernel_size=2),
            Flatten(),
            nn.Linear(n_filters, out_features=n_output_channels),
            activation
        )

    def forward(self, features):
        return self.classification(features)


class YNet(nn.Module):
    def __init__(self, n_input_channels, n_filters, n_output_channels):
        super().__init__()
        self.unet = UNet(n_input_channels, n_filters, n_output_channels,
                         dropout=None)  # dropout used to be 0.2 in the MIDL paper
        self.yleg1 = YLeg(n_filters, n_filters // 2, n_output_channels, activation='sigmoid')  # completeness
        self.yleg2 = YLeg(n_filters, n_filters // 2, n_output_channels=1, activation='relu')  # label'sigmoid')#

    def forward(self, image, memory, memory_discs=None):
        if memory_discs is None:
            x = torch.cat([image, memory], dim=1)
        else:
            x = torch.cat([image, memory, memory_discs], dim=1)
        segmentation, features = self.unet(x)
        classification = self.yleg1(features)
        label = self.yleg2(features)

        return segmentation, classification, label.reshape(-1)

    def nth_activation_map(self, n, image, memory, memory_discs=None):
        if memory_discs is None:
            x = torch.cat([image, memory], dim=1)
        else:
            x = torch.cat([image, memory, memory_discs], dim=1)
        return self.unet.nth_activation_map(n, x)


class CNet(nn.Module):
    def __init__(self, n_input_channels, n_filters):
        super().__init__()
        self.classifier = nn.Sequential(
            *conv3x3(n_input_channels, n_filters),
            *conv3x3(n_filters, n_filters),
            *conv3x3(n_filters, n_filters),
            nn.MaxPool3d(kernel_size=2),  # 64
            *conv3x3(n_filters, n_filters),
            *conv3x3(n_filters, n_filters),
            nn.MaxPool3d(kernel_size=2),  # 32
            *conv3x3(n_filters, n_filters),
            *conv3x3(n_filters, n_filters),
            nn.MaxPool3d(kernel_size=2),  # 16
            *conv3x3(n_filters, n_filters),
            nn.MaxPool3d(kernel_size=2),  # 8
            *conv3x3(n_filters, n_filters),
            nn.MaxPool3d(kernel_size=2),  # 4
            nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),  # 2
            Flatten(),
            nn.Linear(n_filters * 2**3, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)
        return self.classifier(x).reshape(-1)


class NeuralNetwork:
    def __init__(self, network, device, dtype, mixed_precision=False):
        self.device = torch.device(device)
        self.dtype = np.dtype(dtype)
        self.network = network.to(device)
        self.optimizer = None
        self.scaler = amp.GradScaler(enabled=mixed_precision)
        self.mixed_precision = mixed_precision

        if device == 'cuda':
            self.n_devices = torch.cuda.device_count()
            if self.n_devices > 1:
                self.network = torch.nn.DataParallel(self.network)
        else:
            self.n_devices = 1

    def __len__(self):
        # Return number of parameters in the network
        n_params = 0
        for param in self.network.parameters():
            n_params += np.prod(param.size())
        return n_params

    def __str__(self):
        return repr(self.network)

    def snapshot(self, filename):
        # Make sure output directory exists
        dirname = path.dirname(filename)
        if not path.exists(dirname):
            makedirs(dirname)

        # Make sure to never store the parameters with "module." in front of the name even if DataParallel was used
        if self.n_devices == 1:
            network_state_dict = self.network.state_dict()
        else:
            network_state_dict = self.network.module.state_dict()

        torch.save({
            'network': network_state_dict,
            'optimizer': None if self.optimizer is None else self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict()
        }, filename)

    def restore(self, filename):
        state_dicts = torch.load(filename)

        # Backwards compatibly
        if len(state_dicts) > 3:
            network_state_dict = state_dicts
            optimizer_state_dict = None
            scaler_state_dict = None
        else:
            network_state_dict = state_dicts['network']
            optimizer_state_dict = state_dicts.get('optimizer', None)
            scaler_state_dict = state_dicts.get('scaler', None)

        # Remove batch normalization parameters from network state dict
        remove = [key for key in network_state_dict if key.endswith('.running_mean') or key.endswith('.running_var')]
        for key in remove:
            del network_state_dict[key]

        # If multiple devices are used, DataParallel requires all parameter names to begin with "module."
        if self.n_devices > 1:
            self.network.module.load_state_dict(network_state_dict)
        else:
            self.network.load_state_dict(network_state_dict)

        # Set optimizer state dict only if both components are available
        if self.optimizer is not None and optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)

        # Restore scaler state dict
        if scaler_state_dict is not None:
            self.scaler.load_state_dict(scaler_state_dict)

    def _from_numpy(self, data):
        array = np.asarray(data, dtype=self.dtype)
        return torch.from_numpy(array).to(self.device)

    def _to_numpy(self, tensor):
        array = tensor.to('cpu').detach().numpy()
        return array if np.size(array) > 1 else float(array)


class SegmentationNetwork(NeuralNetwork):
    def train(self, images, masks, labels, completenesses, weights, channel=0):
        raise NotImplementedError()

    def predict_loss(self, images, masks, labels, completenesses, weights, channel=0):
        raise NotImplementedError()

    def segment_and_classify(self, image, memory_state, label=None, threshold_segmentation=True, channel=0):
        raise NotImplementedError()

    def nth_activation_map(self, n, *inputs):
        raise NotImplementedError()

    def infer_memory_state_from_mask(self, mask, label):
        return np.zeros_like(mask)


class IterativeSegmentationNetwork(SegmentationNetwork):
    """
    Main network that can be imported from this file.

    The network can be trained on single patches and can segment and classify single patches.
    All methods named _name are for internal use only. All other methods expect numpy arrays
    as input and also return numpy arrays, never pytorch tensors.

    The network is a YNet, which is a UNet for segmentation + a YLeg for classification.
    Implementations of the components can be found below.
    """

    def __init__(self, n_filters=64, n_input_channels=2, n_output_channels=1, traversal_direction='up', device='cuda',
                 dtype='float32', mixed_precision=False):
        assert n_filters > 0
        assert traversal_direction in ('up', 'down')

        super().__init__(
            YNet(n_input_channels=n_input_channels, n_filters=n_filters, n_output_channels=n_output_channels), device,
            dtype, mixed_precision)

        self.traversal_direction = traversal_direction
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001, betas=(0.99, 0.999),
                                          weight_decay=0.00001)
        self.segmentation_error_balance = SigmoidMeanErrorScoreBalance()

    def train(self, images, masks, labels, completenesses, weights, channel=0):
        self.optimizer.zero_grad()
        loss, segmentation_error, classification_error, labeling_error = self._loss(images, masks, labels,
                                                                                    completenesses, weights, channel,
                                                                                    step=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return self._to_numpy(loss), self._to_numpy(segmentation_error), self._to_numpy(
            classification_error), self._to_numpy(labeling_error)

    def predict_loss(self, images, masks, labels, completenesses, weights, channel=0):
        with torch.no_grad():
            loss, segmentation_error, classification_error, labeling_error = self._loss(images, masks, labels,
                                                                                        completenesses, weights,
                                                                                        channel)
        return self._to_numpy(loss), self._to_numpy(segmentation_error), self._to_numpy(
            classification_error), self._to_numpy(labeling_error)

    def segment_and_classify(self, image, memory_state, label=None, threshold_segmentation=True, channel=0, *, memory_state_discs=None):
        # If there is a label, we are supposed to infer the memory state
        if label is None:
            memory_state = self._from_numpy(memory_state > 0)
        else:
            memory_state, _ = self._infer_memory_state_from_mask(memory_state, label)

        # Pass image through the network
        with torch.no_grad(), amp.autocast(enabled=self.mixed_precision):
            segmentation, classification, labeling = self._forward([image], memory_state[None, :, :, :], deterministic=True, channel=channel)
            segmentation = segmentation[0, :, :, :]

        # Threshold or return probabilities?
        if threshold_segmentation:
            segmentation = segmentation.round()
            if label:
                segmentation *= float(label)
            labeling = labeling.round()

        return self._to_numpy(segmentation), self._to_numpy(classification), self._to_numpy(labeling)

    def nth_activation_map(self, n, *inputs):
        image = self._from_numpy(inputs[0])[None, None, :, :, :]
        memory_state = self._from_numpy(inputs[1] > 0)[None, None, :, :, :]

        self.network.train(False)
        with torch.no_grad(), amp.autocast(enabled=self.mixed_precision):
            activation = self.network.nth_activation_map(n, image, memory_state)

        return self._to_numpy(torch.sum(activation[0, :, :, :, :], dim=0))

    def infer_memory_state_from_mask(self, mask, label):
        memory_state, _ = self._infer_memory_state_from_mask(mask, label, return_tensors=False)
        return memory_state

    def _loss(self, images, masks, labels, completenesses, weights, channel, step=False):
        memory_states, ground_truths = self._infer_memory_states_from_masks(masks, labels)

        with amp.autocast(enabled=self.mixed_precision):
            # Forward pass
            pred_segm, pred_cmpl, pred_labl = self._forward(images, memory_states, channel=channel)

            # Calculate loss terms
            segmentation_error = mean_error_score(pred_segm, ground_truths, self._from_numpy(weights),
                                                  self.segmentation_error_balance.weight) + \
                                 binary_cross_entropy(pred_segm, ground_truths)
            classification_error = binary_cross_entropy(pred_cmpl, self._from_numpy(completenesses))
            labeling_error = mean_absolute_error(pred_labl, self._from_numpy(labels)) * 10 + \
                             mean_squared_error(pred_labl, self._from_numpy(labels))

            loss = segmentation_error + classification_error + labeling_error

        if step:
            self.segmentation_error_balance.step()

        # print(pred_labl.cpu().detach().numpy())
        # print(self._from_numpy(labels))
        # print(segmentation_error)
        # print(labeling_error)

        return loss, segmentation_error, classification_error, labeling_error

    def _forward(self, images, memory_states, memory_states_discs=None, deterministic=False, channel=0):
        """ Passes images (list of numpy array) with corresponding memory state (tensor) through the network """
        images = self._from_numpy(images)[:, None, :, :, :]
        memory_states = memory_states[:, None, :, :, :]
        self.network.train(not deterministic)
        segmentation, classification, labeling = self.network(images, memory_states)
        return segmentation[:, channel, :, :, :], classification[:, channel], labeling

    def _infer_memory_state_from_mask(self, mask, label, return_tensors=True):
        """ Input mask should be a numpy array, output masks are pytorch tensors if return_tensor is True """
        if label == 0:
            # Background label, everything is foreground
            memory_state = mask > 0
            ground_truth = np.zeros_like(mask, dtype=self.dtype)
        else:
            # Memory state depends on the direction of traversal
            if self.traversal_direction == 'up':
                memory_state = mask > label
            else:
                memory_state = np.logical_and(mask < label, mask > 0)

            ground_truth = mask == label

        # Convert from numpy to pytorch
        if return_tensors:
            memory_state = self._from_numpy(memory_state)
            ground_truth = self._from_numpy(ground_truth)

        return memory_state, ground_truth

    def _infer_memory_states_from_masks(self, masks, labels, return_tensors=True):
        """ Input masks are a list of numpy arrays, output masks are pytorch tensors if return_tensor is True """
        states_list = [self._infer_memory_state_from_mask(mask, label, return_tensors=False) for mask, label in
                       zip(masks, labels)]
        states = np.asarray(states_list)

        memory_states = states.take(0, axis=1)
        ground_truths = states.take(1, axis=1)

        # Convert from numpy to pytorch
        if return_tensors:
            memory_states = self._from_numpy(memory_states)
            ground_truths = self._from_numpy(ground_truths)

        return memory_states, ground_truths


class IterativeSegmentationNetworkDoubleMemoryState(IterativeSegmentationNetwork):
    """
    Main network that can be imported from this file.

    The network can be trained on single patches and can segment and classify single patches.
    All methods named _name are for internal use only. All other methods expect numpy arrays
    as input and also return numpy arrays, never pytorch tensors.

    The network is a YNet, which is a UNet for segmentation + a YLeg for classification.
    Implementations of the components can be found below.
    """

    def __init__(self, n_filters=64, n_input_channels=3, n_output_channels=3, traversal_direction='up', device='cuda',
                 dtype='float32', mixed_precision=False):
        assert n_filters > 0
        assert traversal_direction in ('up', 'down')

        super(IterativeSegmentationNetwork, self).__init__(
            YNet(n_input_channels=n_input_channels, n_filters=n_filters, n_output_channels=n_output_channels), device,
            dtype, mixed_precision)

        self.n_input_channels = n_input_channels
        self.traversal_direction = traversal_direction
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001, betas=(0.99, 0.999),
                                          weight_decay=0.00001)
        self.segmentation_error_balance = SigmoidMeanErrorScoreBalance()

    def segment_and_classify(self, image, memory_state, label=None, threshold_segmentation=True, channel=0, *, memory_state_discs=None):
        # When used during training the mask with all structures is used as input
        # When used in the Test script the previously segmented memory state is used as input with all structures separately
        if memory_state_discs is None:
            memory_state_discs = memory_state

        # If there is a label, we are supposed to infer the memory state
        if label is None:
            memory_state, _ = self._infer_memory_state_from_mask(memory_state, 0)
            memory_state_discs, _ = self._infer_memory_state_discs_from_mask(memory_state_discs, 0)
        else:
            memory_state, _ = self._infer_memory_state_from_mask(memory_state, label)
            memory_state_discs, _ = self._infer_memory_state_discs_from_mask(memory_state_discs, label)

        # Pass image through the network
        with torch.no_grad(), amp.autocast(enabled=self.mixed_precision):
            segmentation, classification, labeling = self._forward([image], memory_state[None, :, :, :],
                                                                   memory_state_discs[None, :, :, :],
                                                                   deterministic=True, channel=channel)

        # Threshold or return probabilities?
        if threshold_segmentation:
            segmentation = segmentation.round()
            if label:
                segmentation *= float(label)
            labeling = labeling.round()

        return self._to_numpy(segmentation[0]), self._to_numpy(classification), self._to_numpy(labeling)

    def train(self, images, masks, labels, completenesses, weights, channel=0):
        self.optimizer.zero_grad()
        loss, segmentation_error, classification_error, labeling_error, segmentation_error_vert, segmentation_error_disc, segmentation_error_SC = self._loss(
            images, masks, labels,
            completenesses, weights, channel,
            step=True)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return self._to_numpy(loss), self._to_numpy(segmentation_error), self._to_numpy(
            classification_error), self._to_numpy(labeling_error), self._to_numpy(
            segmentation_error_vert), self._to_numpy(segmentation_error_disc), self._to_numpy(segmentation_error_SC)

    def predict_loss(self, images, masks, labels, completenesses, weights, channel=0):
        with torch.no_grad():
            loss, segmentation_error, classification_error, labeling_error, segmentation_error_vert, segmentation_error_disc, segmentation_error_SC = self._loss(
                images, masks, labels,
                completenesses, weights,
                channel)
        return self._to_numpy(loss), self._to_numpy(segmentation_error), self._to_numpy(
            classification_error), self._to_numpy(labeling_error), self._to_numpy(
            segmentation_error_vert), self._to_numpy(segmentation_error_disc), self._to_numpy(segmentation_error_SC)

    def infer_memory_state_from_mask(self, mask, label):
        memory_state, ground_truth = self._infer_memory_state_from_mask(mask, label, return_tensors=False)
        memory_states_disc, _ = self._infer_memory_state_discs_from_mask(mask, label, return_tensors=False)
        return memory_state, memory_states_disc, ground_truth

    def _segmentation_error(self, weights, pred_segm, ground_truths):
        W = self._from_numpy(weights)
        S = self.segmentation_error_balance.weight
        BCE = binary_cross_entropy(pred_segm, ground_truths)

        segmentation_error = mean_error_score(pred_segm, ground_truths, W, S) + BCE

        return segmentation_error

    def _loss(self, images, masks, labels, completenesses, weights, channel, step=False):
        memory_states, ground_truths = self._infer_memory_states_from_masks(masks, labels)
        memory_states_discs, _ = self._infer_memory_states_from_masks(masks, labels, discs=True)

        with amp.autocast(enabled=self.mixed_precision):
            # Forward pass
            pred_segm, pred_cmpl, pred_labl = self._forward(images, memory_states, memory_states_discs, channel=channel)

            weights_vert = [np.copy(weight[0]) for weight in weights]
            weights_disc = [np.copy(weight[1]) for weight in weights]
            weights_SC = [np.copy(weight[2]) for weight in weights]

            segmentation_error_vert = self._segmentation_error(weights_vert, pred_segm[:, 0, :, :, :],
                                                               ground_truths[:, 0, :, :, :])
            segmentation_error_disc = self._segmentation_error(weights_disc, pred_segm[:, 1, :, :, :],
                                                               ground_truths[:, 1, :, :, :])
            segmentation_error_SC = self._segmentation_error(weights_SC, pred_segm[:, 2, :, :, :],
                                                             ground_truths[:, 2, :, :, :])

            segmentation_error = segmentation_error_vert + segmentation_error_disc + segmentation_error_SC
            classification_error = binary_cross_entropy(pred_cmpl, self._from_numpy(completenesses))
            labeling_error = mean_absolute_error(pred_labl, self._from_numpy(labels)) * 10 + \
                             mean_squared_error(pred_labl, self._from_numpy(labels))

            loss = segmentation_error + classification_error + labeling_error

        if step:
            self.segmentation_error_balance.step()

        # print(pred_labl.cpu().detach().numpy())
        # print(self._from_numpy(labels).cpu().detach().numpy())
        print(segmentation_error.cpu().detach().numpy())
        # print(labeling_error.cpu().detach().numpy())

        return loss, segmentation_error, classification_error, labeling_error, segmentation_error_vert, segmentation_error_disc, segmentation_error_SC

    def _forward(self, images, memory_states, memory_states_discs=None, deterministic=False, channel=0):
        """ Passes images (list of numpy array) with corresponding memory state (tensor) through the network """
        images = self._from_numpy(images)[:, None, :, :, :]
        # if len(memory_states.shape) == 4:
        memory_states = memory_states[:, None, :, :, :]
        # if len(memory_states_discs.shape) == 4:
        memory_states_discs = memory_states_discs[:, None, :, :, :]
        self.network.train(not deterministic)
        segmentation, classification, labeling = self.network(images, memory_states, memory_discs=memory_states_discs)
        return segmentation, classification, labeling

    def _infer_memory_state_discs_from_mask(self, mask, label, return_tensors=True):
        """ Input mask should be a numpy array, output masks are pytorch tensors if return_tensor is True """
        if label == 0:
            # Background label, everything is foreground
            memory_state = mask > 200  # All discs
            ground_truth = np.stack(
                [np.zeros_like(mask, dtype=self.dtype), np.zeros_like(mask, dtype=self.dtype), mask == 100])
        else:
            ground_truth = np.stack([mask == label, mask == (label + 200), mask == 100])
            # Memory state depends on the direction of traversal
            if self.traversal_direction == 'up':
                memory_state = mask > (label + 200)
            else:
                memory_state = np.logical_and(mask < (label + 200), mask > 100)

        # Convert from numpy to pytorch
        if return_tensors:
            memory_state = self._from_numpy(memory_state)
            ground_truth = self._from_numpy(ground_truth)

        return memory_state, ground_truth

    def _infer_memory_state_from_mask(self, mask, label, return_tensors=True):
        """ Input mask should be a numpy array, output masks are pytorch tensors if return_tensor is True """
        # mask[np.where(mask > 25)] = 0 # WHACH OUT! 25 is the max vertebra.
        if label == 0:
            # Background label, everything is foreground
            memory_state = np.logical_and(mask > 0, mask < 99)  # All vertebrae
            ground_truth = np.stack(
                [np.zeros_like(mask, dtype=self.dtype), np.zeros_like(mask, dtype=self.dtype), mask == 100])
        else:
            # compute ground truth, three channels. First vertebrae, than IVDs, than SC
            ground_truth = np.stack([mask == label, mask == (label + 200), mask == 100])

            # Memory state depends on the direction of traversal
            if self.traversal_direction == 'up':
                memory_state = np.logical_and(mask > label, mask < 100)
            else:
                memory_state = np.logical_and(mask < label, mask > 0)

        # Convert from numpy to pytorch
        if return_tensors:
            memory_state = self._from_numpy(memory_state)
            ground_truth = self._from_numpy(ground_truth)

        return memory_state, ground_truth

    def _infer_memory_states_from_masks(self, masks, labels, return_tensors=True, discs=False):
        """ Input masks are a list of numpy arrays, output masks are pytorch tensors if return_tensor is True """
        memory_states = []
        ground_truths = []

        if discs:
            for mask, label in zip(masks, labels):
                memory_state, ground_truth = self._infer_memory_state_discs_from_mask(mask, label, return_tensors=False)
                memory_states.append(memory_state)
                ground_truths.append(ground_truth)
        else:
            for mask, label in zip(masks, labels):
                memory_state, ground_truth = self._infer_memory_state_from_mask(mask, label, return_tensors=False)
                memory_states.append(memory_state)
                ground_truths.append(ground_truth)

        memory_states = np.asarray(memory_states)
        ground_truths = np.asarray(ground_truths)

        # Convert from numpy to pytorch
        if return_tensors:
            memory_states = self._from_numpy(memory_states)
            ground_truths = self._from_numpy(ground_truths)

        return memory_states, ground_truths


class DiscsSpinalCanalSegmentationNetwork(SegmentationNetwork):
    """
    Main network that can be imported from this file.

    The network can be trained on single patches and can segment and classify single patches.
    All methods named _name are for internal use only. All other methods expect numpy arrays
    as input and also return numpy arrays, never pytorch tensors.

    The network is a UNet for segmentation of intervertebral discs and the spinal canal.
    Implementations of the components can be found below.
    """

    def __init__(self, n_filters=64, n_output_channels=2, traversal_direction='up', device='cuda', dtype='float32'):
        assert n_filters > 0
        assert traversal_direction in ('up', 'down')

        # WATCH OUT: change the n_input_channels for with or without SC
        super().__init__(MemoryUNet(n_input_channels=2, n_filters=n_filters, n_output_channels=n_output_channels),
                         device, dtype)

        self.traversal_direction = traversal_direction
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001, betas=(0.99, 0.999),
                                          weight_decay=0.00001)
        self.segmentation_error_balance = SigmoidMeanErrorScoreBalance()

    def train(self, images, masks, labels, completenesses, weights, channel=0):
        self.optimizer.zero_grad()
        loss, segmentation_error = self._loss(images, masks, labels, weights, step=True)
        loss.backward()
        self.optimizer.step()
        return self._to_numpy(loss), self._to_numpy(segmentation_error)

    def predict_loss(self, images, masks, labels, completenesses, weights, channel=0):
        with torch.no_grad():
            loss, segmentation_error = self._loss(images, masks, labels, weights)
        return self._to_numpy(loss), self._to_numpy(segmentation_error)

    def segment_and_classify(self, image, memory_state, label=None, threshold_segmentation=True, channel=0):
        # If there is a label, we are supposed to infer the memory state
        if label is None:
            memory_state = self._from_numpy(memory_state > 0)
        else:
            memory_state, _ = self._infer_memory_state_from_mask(memory_state, label)

        # Pass image through the network
        with torch.no_grad():
            segmentation = self._forward([image], memory_state[None, :, :, :], deterministic=True)
            segmentation = segmentation[0, :, :, :]

        # Threshold or return probabilities?
        if threshold_segmentation:
            segmentation = segmentation.round()

        return self._to_numpy(segmentation)

    def nth_activation_map(self, n, *inputs):
        image = self._from_numpy(inputs[0])[None, None, :, :, :]
        memory_state = self._from_numpy(inputs[1] > 0)[None, None, :, :, :]

        self.network.train(False)
        with torch.no_grad():
            activation = self.network.nth_activation_map(n, image, memory_state)

        return self._to_numpy(torch.sum(activation[0, :, :, :, :], dim=0))

    def infer_memory_state_from_mask(self, mask, label):
        memory_state, _ = self._infer_memory_state_from_mask(mask, label, return_tensors=False)
        return memory_state

    def _loss(self, images, masks, labels, weights, step=False):
        memory_states, ground_truths = self._infer_memory_states_from_masks(masks, labels)

        # Forward pass
        pred_segm = self._forward(images, memory_states)

        # Calculate loss terms
        segmentation_error = mean_error_score(pred_segm, ground_truths, self._from_numpy(weights),
                                              self.segmentation_error_balance.weight) + \
                             binary_cross_entropy(pred_segm, ground_truths)

        loss = segmentation_error

        if step:
            self.segmentation_error_balance.step()

        return loss, segmentation_error

    def _forward(self, images, memory_states, deterministic=False):
        """ Passes images (list of numpy array) with corresponding memory state (tensor) through the network """
        images = self._from_numpy(images)[:, None, :, :, :]
        memory_states = memory_states[:, None, :, :, :]
        self.network.train(not deterministic)
        segmentation = self.network(images, memory_states)
        return segmentation

    def _infer_memory_state_from_mask(self, mask, label, return_tensors=True):
        """ Input mask should be a numpy array, output masks are pytorch tensors if return_tensor is True """
        if label == 0:
            # Background label, everything is foreground
            memory_state = np.logical_and(mask > 0, mask != 100)  # everything without spinal canal
            ground_truth = np.stack([np.zeros_like(mask, dtype=self.dtype), mask == 100])
        else:
            # Memory state depends on the direction of traversal
            if self.traversal_direction == 'up':
                memory_state = np.logical_or(mask > label, np.logical_and(mask < 100, mask > (label - 201)))
            else:
                memory_state = np.logical_or(np.logical_and(mask < label, mask > 100),
                                             np.logical_and(mask < (label - 199), mask > 0))

            ground_truth = np.stack([mask == label, mask == 100])

        # Convert from numpy to pytorch
        if return_tensors:
            memory_state = self._from_numpy(memory_state)
            ground_truth = self._from_numpy(ground_truth)

        return memory_state, ground_truth

    def _infer_memory_states_from_masks(self, masks, labels, return_tensors=True):
        """ Input masks are a list of numpy arrays, output masks are pytorch tensors if return_tensor is True """
        states = [self._infer_memory_state_from_mask(mask, label, return_tensors=False) for mask, label in
                  zip(masks, labels)]

        memory_states = np.asarray([state[0] for state in states])
        ground_truths = np.asarray([state[1] for state in states])

        # Convert from numpy to pytorch
        if return_tensors:
            memory_states = self._from_numpy(memory_states)
            ground_truths = self._from_numpy(ground_truths)

        return memory_states, ground_truths


class MulticlassSegmentationNetwork(SegmentationNetwork):
    def __init__(self, n_filters=64, n_classes=25, device='cuda', dtype='float32'):
        assert n_filters > 0
        assert n_classes > 1

        network = UNet(n_input_channels=1, n_filters=n_filters, n_output_channels=n_classes, sigmoid=False)
        super().__init__(network, device, dtype)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=0.001, betas=(0.99, 0.999), weight_decay=0.00001
        )
        self.loss_function = SoftmaxCrossEntropy()

    def train(self, images, masks, labels, completenesses, weights, channel=0):
        self.optimizer.zero_grad()
        loss = self._loss(images, masks, weights)
        loss.backward()
        self.optimizer.step()

        segmentation_error = self._to_numpy(loss)
        return segmentation_error, segmentation_error, 0, 0

    def predict_loss(self, images, masks, labels, completenesses, weights, channel=0):
        with torch.no_grad():
            loss = self._loss(images, masks, weights)

        segmentation_error = self._to_numpy(loss)
        return segmentation_error, segmentation_error, 0, 0

    def segment_and_classify(self, image, memory_state, label=None, threshold_segmentation=True, channel=0):
        # Pass image through the network
        with torch.no_grad():
            segmentation = self._forward([image], deterministic=True)
            segmentation = segmentation[0, :, :, :, :]

        # Threshold or return probabilities?
        if threshold_segmentation:
            segmentation = segmentation.argmax(dim=0)
            if label:
                segmentation *= float(label)
        else:
            segmentation = F.softmax(segmentation, dim=0)

        return self._to_numpy(segmentation), 1, label if label else 1

    def nth_activation_map(self, n, *inputs):
        image = self._from_numpy(inputs[0])[None, None, :, :, :]

        self.network.train(False)
        with torch.no_grad():
            activation = self.network.nth_activation_map(n, image)

        return self._to_numpy(torch.sum(activation[0, :, :, :, :], dim=0))

    def _loss(self, images, masks, weights):
        return self.loss_function(self._forward(images), self._from_numpy(masks), self._from_numpy(weights))

    def _forward(self, images, deterministic=False):
        """
        Passes images (list of numpy array) through the network
        and returns the segmentation masks (5D pytorch tensor)
        """
        images = self._from_numpy(images)[:, None, :, :, :]
        self.network.train(not deterministic)
        segmentation, features = self.network(images)
        return segmentation


class RibDetectionNetwork(NeuralNetwork):
    def __init__(self, n_filters=24, device='cuda', dtype='float32'):
        assert n_filters > 0
        super().__init__(CNet(n_input_channels=2, n_filters=n_filters), device, dtype)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001, betas=(0.99, 0.999),
                                          weight_decay=0.00001)
        self.optimizer.zero_grad()

    def train(self, images, masks, labels, step=True):
        loss = self._loss(images, masks, labels)
        loss.backward()
        if step:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return self._to_numpy(loss)

    def predict_loss(self, images, masks, labels):
        with torch.no_grad():
            loss = self._loss(images, masks, labels)
        return self._to_numpy(loss)

    def predict(self, image, mask):
        # Pass image through the network
        with torch.no_grad():
            classification = self._forward([image], mask[None, :, :, :], deterministic=True)
        return self._to_numpy(classification)

    def _loss(self, images, masks, labels):
        predictions = self._forward(images, masks)
        return binary_cross_entropy(predictions, self._from_numpy(labels))

    def _forward(self, images, masks, deterministic=False):
        """ Passes images (list of numpy array) with corresponding memory state (tensor) through the network """
        images = self._from_numpy(images)[:, None, :, :, :]
        masks = self._from_numpy(masks)[:, None, :, :, :]
        self.network.train(not deterministic)
        return self.network(images, masks)
