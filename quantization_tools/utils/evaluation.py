import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.autograd import Variable
from scipy import linalg
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
# for clip_score
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from tqdm import tqdm

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in 
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output 
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.upsample(x, size=(299, 299), mode='bilinear', align_corners=True)

        if self.normalize_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, path, transform=None):
        'Initialization'
        self.file_names = self.get_filenames(path)
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_names)

    def __getitem__(self, index):
        'Generates one sample of data'
        img = Image.open(self.file_names[index]).convert('RGB')
        # Convert image and label to torch tensors
        if self.transform is not None:
            img = self.transform(img)
        return img

    def get_filenames(self, data_path):
        images = []
        for path, subdirs, files in os.walk(data_path):
            for name in files:
                if name.rfind('jpg') != -1 or name.rfind('png') != -1:
                    filename = os.path.join(path, name)
                    if os.path.isfile(filename):
                        images.append(filename)
        return images


def get_activations(images, model, batch_size=64, dims=2048, cuda=False, verbose=True):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    #d0 = images.shape[0]

    d0 = images.__len__() * batch_size
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))
    #for i in range(n_batches):
    for i, batch in enumerate(images):
        #batch = batch[0]
        #if verbose:
            #print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
        #import ipdb
        #ipdb.set_trace()
        start = i * batch_size
        end = start + batch_size

        #batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
        #batch = Variable(batch, volatile=True)

        if cuda:
            batch = batch.cuda()
        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)
    if verbose:
        print(' done')

    return pred_arr


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        dataset = Dataset(path, transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ]))
        print(dataset.__len__())
        dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
        m, s = calculate_activation_statistics(dataloader, model, batch_size, dims, cuda)
    return m, s


def calculate_activation_statistics(images, model, batch_size=64,
                                    dims=2048, cuda=False, verbose=True):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(images, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an 
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an 
               representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_fid_given_paths(paths, batch_size, dims, cuda=True):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size, dims, cuda)
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size, dims, cuda)
    # np.savez("/hd6/dongbeibei/quant/dataset/distribution.npz", mu=m2, sigma=s2)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    print(f"fid: {fid_value}")
    return fid_value



# CLIP score
@torch.no_grad()
def calculate_clip_score(clip_score_fn, images, prompts, device):

    images_int = (images).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2).to(device), prompts).detach()
    sd_clip_score = round(float(clip_score), 4)
    return sd_clip_score

# calculate CLIP score given a image directory        
def calculate_clip_score_dir(dir_path, bs = 50, max_num = None, device = "cuda"):

    texts = []
    imgs = []
    # clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-large-patch14")
    for i, img_file in enumerate( tqdm(os.listdir(dir_path), desc="reading images") ):
        # text = img_file.replace("_"," ").split('.')[0]
        # text = text.split(" ")[1:]
        # text = " ".join(text) + "."
        text = img_file.replace("_"," ").replace(".png",'')
        text = text.split(" ")[1:]
        text = " ".join(text)
        texts.append(text)
        img = Image.open(os.path.join(dir_path,img_file))
        img_array = np.array(img)
        imgs.append(img_array)
        if max_num is not None and i >= max_num -1:
            break

    total_score = 0
    total_num = 0
    for start in tqdm(range(0,len(texts), bs), desc="calculating score"):
        curr_texts = texts[start:start+bs]
        curr_imgs = np.stack(imgs[start:start+bs], axis=0)
        score = calculate_clip_score(clip_score_fn, curr_imgs, curr_texts, device)
        curr_num = len(curr_texts)
        total_score = total_score + score*curr_num
        total_num += curr_num
    
    avg_clip_score = total_score/total_num
    print(f"dir path:{dir_path}")
    print(f"Average CLIP score of {total_num} paris: {avg_clip_score:.2f}")
    return avg_clip_score
