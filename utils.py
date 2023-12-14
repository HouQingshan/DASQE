from torch.utils.data import DataLoader
from networks import vgg_19
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from data import ImageFilelist, ImageFolder
import torch
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import time


def get_all_data_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    if 'new_size' in conf:
        new_size_a = new_size_b = new_size_c = conf['new_size']
    else:
        new_size_a = conf['new_size_a']
        new_size_b = conf['new_size_b']
        new_size_c = conf['new_size_c']
    height = conf['crop_image_height']
    width = conf['crop_image_width']

    if 'data_root' in conf:
        train_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], 'data_a'), batch_size, True,
                                                new_size_a, height, width, num_workers, True)
        train_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], 'data_b'), batch_size, True,
                                                new_size_b, height, width, num_workers, True)
        train_loader_c = get_data_loader_folder(os.path.join(conf['data_root'], 'data_c'), batch_size, True,
                                                new_size_c, height, width, num_workers, True)

        test_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], 'data_a'), batch_size, False,
                                               new_size_a, new_size_a, new_size_a, num_workers, True)
        test_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], 'data_b'), batch_size, False,
                                               new_size_b, new_size_b, new_size_b, num_workers, True)
        test_loader_c = get_data_loader_folder(os.path.join(conf['data_root'], 'data_c'), batch_size, False,
                                               new_size_c, new_size_c, new_size_c, num_workers, True)
    else:
        train_loader_a = get_data_loader_list(conf['data_folder_train_a'], conf['data_list_train_a'], batch_size, True,
                                                new_size_a, height, width, num_workers, True)
        test_loader_a = get_data_loader_list(conf['data_folder_test_a'], conf['data_list_test_a'], batch_size, False,
                                                new_size_a, new_size_a, new_size_a, num_workers, True)
        train_loader_b = get_data_loader_list(conf['data_folder_train_b'], conf['data_list_train_b'], batch_size, True,
                                                new_size_b, height, width, num_workers, True)                                    
        test_loader_b = get_data_loader_list(conf['data_folder_test_b'], conf['data_list_test_b'], batch_size, False,
                                                new_size_b, new_size_b, new_size_b, num_workers, True)
        train_loader_c = get_data_loader_list(conf['data_folder_train_c'], conf['data_list_train_c'], batch_size, True,
                                              new_size_c, height, width, num_workers, True)
        test_loader_c = get_data_loader_list(conf['data_folder_test_c'], conf['data_list_test_c'], batch_size, False,
                                             new_size_c, new_size_c, new_size_c, num_workers, True)
    return train_loader_a, train_loader_b, train_loader_c, test_loader_a, test_loader_b, test_loader_c


def get_data_loader_list(root, file_list, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFilelist(root, file_list, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader

def get_data_loader_folder(input_folder, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor()] 
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list 

    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader
def get_data_loader_folder_Colorjit(input_folder, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True, color_jit = True):
    transform_list = [transforms.ToTensor()]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform_list = [transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5),
                                    hue=(-0.4, 0.5))] + transform_list if color_jit else transform_list 
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input
    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return 
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def eformat(f, prec):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    return "%se%d"%(mantissa, int(exp))


def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] 
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=False)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)
    __write_images(image_outputs[0:n // 3], display_image_num, '%s/gen_a_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[n // 3:2 * n // 3], display_image_num, '%s/gen_b_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[2 * n // 3:n], display_image_num, '%s/gen_c_%s.jpg' % (image_directory, postfix))


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations,img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>/home/ubuntu/IQA_EYE/division/result/last_patch/
        <p>
        """ % (img_filename, img_filename, all_size))
    return


def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations, '%s/gen_a2b_train_current.jpg' % (image_directory), all_size)
    write_one_row_html(html_file, iterations, '%s/gen_b2a_train_current.jpg' % (image_directory), all_size)
    for j in range(iterations, image_save_iterations-1, -1):
        if j % image_save_iterations == 0:
            write_one_row_html(html_file, j, '%s/gen_a2b_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_a2b_train_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_train_%08d.jpg' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def slerp(val, low, high):
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_slerp_interp(nb_latents, nb_interp, z_dim):

    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def load_vgg19():

    vgg = vgg_19()
    return vgg


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) 
    batch = batch * 255   
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean)) 
    return batch


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None 
    elif hyperparameters['lr_policy'] == 'CosineAnnealingWarm':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=5e-5)
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'], gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


class BMTD_algorithm():
    def __init__(self):
        self.initial_task1_loss_list = 0
        self.initial_task2_loss_list = 0
        self.weights_c1_save = []
        self.weights_c2_save = []

    def __call__(self, iteration, c1_loss, c2_loss, alpha=0.5):

        if iteration == 1:
            self.initial_task1_loss_list = c1_loss.item()
            self.initial_task2_loss_list = c2_loss.item()

        c1_loss_ratio = c1_loss.item() / self.initial_task1_loss_list
        c2_loss_ratio = c2_loss.item() / self.initial_task2_loss_list

        c1_loss_weight = pow(c1_loss_ratio,alpha)
        c2_loss_weight = pow(c2_loss_ratio,alpha)

        weights_sum = c1_loss_weight + c2_loss_weight

        c1_loss_weight = c1_loss_weight / weights_sum * 2
        c2_loss_weight = c2_loss_weight / weights_sum * 2

        self.weights_c1_save.append(c1_loss_weight)
        self.weights_c2_save.append(c2_loss_weight)

        losses = c1_loss * c1_loss_weight + c2_loss * c2_loss_weight
        return losses / 2.0