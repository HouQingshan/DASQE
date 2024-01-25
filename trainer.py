from networks import AdaINGen, MsImageDis, Dis_content, VAEGen
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg19, get_scheduler, BMTD_algorithm
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import functional as F
# from GaussianSmoothLayer import GaussionSmoothLayer, GradientLoss
import os


class UNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  
        self.gen_c = VAEGen(hyperparameters['input_dim_c'], hyperparameters['gen'])  

        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  
        self.dis_c = MsImageDis(hyperparameters['input_dim_c'], hyperparameters['dis']) 
       
        self.dis_content_ab = Dis_content()
        self.dis_content_bc = Dis_content()
        self.gpuid = hyperparameters['gpuID']
        self.device = hyperparameters['device']
       
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
       
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters()) + list(self.dis_c.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters()) + list(self.gen_c.parameters())
        content_params = list(self.dis_content_ab.parameters()) + list(self.dis_content_bc.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.content_opt = torch.optim.Adam([p for p in content_params if p.requires_grad],
                                        lr=lr / 2, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.content_scheduler = get_scheduler(self.content_opt, hyperparameters)

       
        self.gen_a.apply(weights_init(hyperparameters['init']))
        self.gen_b.apply(weights_init(hyperparameters['init']))
        self.gen_c.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))
        self.dis_c.apply(weights_init('gaussian'))
        self.dis_content_ab.apply(weights_init('gaussian'))
        self.dis_content_bc.apply(weights_init('gaussian'))

    
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg19()
            if torch.cuda.is_available():
                self.vgg.cuda(self.gpuid)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

   
    def forward(self, x_a, x_b):
        self.eval()
        h_a = self.gen_a.encode_cont(x_a)
        
        x_ab = self.gen_b.decode_cont(h_a)
    
        return x_ab 

    def __compute_kl(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def content_update(self, x_a, x_b, x_c, hyperparameters): 
        self.content_opt.zero_grad()
        enc_a = self.gen_a.encode_cont(x_a)
        enc_b = self.gen_b.encode_cont(x_b)
        enc_c = self.gen_c.encode_cont(x_c)

        pred_fake_ab = self.dis_content_ab.forward(enc_a)
        pred_real_ab = self.dis_content_ab.forward(enc_b)
        pred_fake_bc = self.dis_content_bc.forward(enc_c)
        pred_real_bc = self.dis_content_bc.forward(enc_b)
        loss_D = 0

        if hyperparameters['gan_type'] == 'lsgan':
            loss_D += torch.mean((pred_fake_ab - 0)**2) + torch.mean((pred_real_ab - 1)**2) + \
                      torch.mean((pred_fake_bc - 0)**2) + torch.mean((pred_real_bc - 1)**2)
        elif hyperparameters['gan_type'] == 'nsgan':
            all0_ab = Variable(torch.zeros_like(pred_fake_ab.data).cuda(self.gpuid), requires_grad=False)
            all1_ab = Variable(torch.ones_like(pred_real_ab.data).cuda(self.gpuid), requires_grad=False)
            all0_bc = Variable(torch.zeros_like(pred_fake_bc.data).cuda(self.gpuid), requires_grad=False)
            all1_bc = Variable(torch.ones_like(pred_real_bc.data).cuda(self.gpuid), requires_grad=False)
            loss_D += torch.mean(F.binary_cross_entropy(F.sigmoid(pred_fake_ab), all0_ab) +
                                   F.binary_cross_entropy(F.sigmoid(pred_real_ab), all1_ab) +
                                 F.binary_cross_entropy(F.sigmoid(pred_fake_bc), all0_bc) +
                                   F.binary_cross_entropy(F.sigmoid(pred_real_bc), all1_bc))
        else:
            assert 0, "Unsupported GAN type: {}".format(hyperparameters['gan_type'])

        loss_D.backward()
        nn.utils.clip_grad_norm_(self.dis_content_ab.parameters(), 5)
        nn.utils.clip_grad_norm_(self.dis_content_bc.parameters(), 5)

        self.content_opt.step()

    def gen_update(self, x_a, x_b, x_c, hyperparameters):
        self.gen_opt.zero_grad()
        self.content_opt.zero_grad()
        h_a = self.gen_a.encode_cont(x_a)
        h_b = self.gen_b.encode_cont(x_b)
        h_c = self.gen_c.encode_cont(x_c)
        h_a_sty = self.gen_a.encode_sty(x_a)
        h_c_sty = self.gen_c.encode_sty(x_c)
       
        out_a = self.dis_content_ab(h_a)
        out_b_ab = self.dis_content_ab(h_b)
        out_c = self.dis_content_bc(h_c)
        out_b_bc = self.dis_content_bc(h_b)
        self.loss_ContentD = 0

        if hyperparameters['gan_type'] == 'lsgan':
            self.loss_ContentD += torch.mean((out_a - 0.5) ** 2) + torch.mean((out_b_ab - 0.5) ** 2) + \
                                  torch.mean((out_a - 0.5) ** 2) + torch.mean((out_b_ab - 0.5) ** 2)
        elif hyperparameters['gan_type'] == 'nsgan':
            all1_ab = Variable(0.5 * torch.ones_like(out_b_ab.data).cuda(self.gpuid), requires_grad=False)
            all1_bc = Variable(0.5 * torch.ones_like(out_b_bc.data).cuda(self.gpuid), requires_grad=False)
            self.loss_ContentD += torch.mean(F.binary_cross_entropy(F.sigmoid(out_a), all1_ab) +
                                   F.binary_cross_entropy(F.sigmoid(out_b_ab), all1_ab) +
                                             F.binary_cross_entropy(F.sigmoid(out_c), all1_bc) +
                                   F.binary_cross_entropy(F.sigmoid(out_b_bc), all1_bc))
        else:
            assert 0, "Unsupported GAN type: {}".format(hyperparameters['gan_type'])

        h_a_cont = torch.cat((h_a, h_a_sty), 1)
        noise_a = torch.randn(h_a_cont.size()).cuda(h_a_cont.data.get_device())
        x_a_recon = self.gen_a.decode_recs(h_a_cont + noise_a)
        noise_b = torch.randn(h_b.size()).cuda(h_b.data.get_device())
        x_b_recon = self.gen_b.decode_cont(h_b + noise_b)
        h_c_cont = torch.cat((h_c, h_c_sty), 1)
        noise_c = torch.randn(h_c_cont.size()).cuda(h_c_cont.data.get_device())
        x_c_recon = self.gen_c.decode_recs(h_c_cont + noise_c)
        
        h_ba_cont = torch.cat((h_b, h_a_sty), 1)
        x_ba = self.gen_a.decode_recs(h_ba_cont + noise_a)
        x_ab = self.gen_b.decode_cont(h_a + noise_b)
        h_bc_cont = torch.cat((h_b, h_c_sty), 1)
        x_bc = self.gen_c.decode_recs(h_bc_cont + noise_c)
        x_cb = self.gen_b.decode_cont(h_c + noise_b)
    
        h_b_recon_a = self.gen_a.encode_cont(x_ba)
        h_b_sty_recon_a = self.gen_a.encode_sty(x_ba)
        h_b_recon_c = self.gen_c.encode_cont(x_bc)
        h_b_sty_recon_c = self.gen_c.encode_sty(x_bc)


        h_a_recon = self.gen_b.encode_cont(x_ab)
        h_c_recon = self.gen_b.encode_cont(x_cb)

        h_a_cat_recs = torch.cat((h_a_recon, h_b_sty_recon_a), 1)
        h_c_cat_recs = torch.cat((h_c_recon, h_b_sty_recon_c), 1)

        x_aba = self.gen_a.decode_recs(h_a_cat_recs + noise_a) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode_cont(h_b_recon_a + noise_b) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_cbc = self.gen_c.decode_recs(h_c_cat_recs + noise_c) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bcb = self.gen_b.decode_cont(h_b_recon_c + noise_b) if hyperparameters['recon_x_cyc_w'] > 0 else None

    
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_x_c = self.recon_criterion(x_c_recon, x_c)

        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_recon_kl_c = self.__compute_kl(h_c)
        self.loss_gen_recon_kl_sty = self.__compute_kl(h_a_sty) + self.__compute_kl(h_c_sty)


        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a) if x_aba is not None else 0
        self.loss_gen_cyc_x_bab = self.recon_criterion(x_bab, x_b) if x_aba is not None else 0
        self.loss_gen_cyc_x_c = self.recon_criterion(x_cbc, x_c) if x_cbc is not None else 0
        self.loss_gen_cyc_x_bcb = self.recon_criterion(x_bcb, x_b) if x_cbc is not None else 0

        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon_a)
        self.loss_gen_recon_kl_cyc_cbc = self.__compute_kl(h_c_recon)
        self.loss_gen_recon_kl_cyc_bcb = self.__compute_kl(h_b_recon_c)
        self.loss_gen_recon_kl_cyc_sty = self.__compute_kl(h_b_sty_recon_a) + self.__compute_kl(h_b_sty_recon_c)

        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab) + self.dis_b.calc_gen_loss(x_cb)
        self.loss_gen_adv_c = self.dis_c.calc_gen_loss(x_bc)
    
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) + self.compute_vgg_loss(self.vgg, x_cb, x_c)\
            if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_c = self.compute_vgg_loss(self.vgg, x_bc, x_b) if hyperparameters['vgg_w'] > 0 else 0

        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_c + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_c + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_c + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_sty + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_bab + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_c + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_cbc + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_bcb + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bcb + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_sty + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_c + \
                              hyperparameters['gan_w'] * self.loss_ContentD
        self.loss_gen_total.backward()
        self.gen_opt.step()
        self.content_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)


    def sample(self, x_a, x_b, x_c):
        if x_a is None or x_b is None or x_c is None:
            return None
        self.eval()
        x_a_recon, x_b_recon, x_c_recon = [], [], []
        x_ab, x_ac, x_ba, x_bc, x_cb, x_ca = [], [], [], [], [], []

        for i in range(x_a.size(0)):
            h_a = self.gen_a.encode_cont(x_a[i].unsqueeze(0))
            h_a_sty = self.gen_a.encode_sty(x_a[i].unsqueeze(0))
            h_b = self.gen_b.encode_cont(x_b[i].unsqueeze(0))
            h_c = self.gen_c.encode_cont(x_c[i].unsqueeze(0))
            h_c_sty = self.gen_c.encode_sty(x_c[i].unsqueeze(0))

            h_ba_cont = torch.cat((h_b, h_a_sty), 1)
            h_bc_cont = torch.cat((h_b, h_c_sty), 1)

            h_aa_cont = torch.cat((h_a, h_a_sty), 1)
            h_cc_cont = torch.cat((h_c, h_c_sty), 1)

            x_ab_img = self.gen_b.decode_cont(h_a)
            x_cb_img = self.gen_b.decode_cont(h_c)
            h_ab = self.gen_b.encode_cont(x_ab_img)
            h_cb = self.gen_b.encode_cont(x_cb_img)
            h_ac_cont = torch.cat((h_ab, h_c_sty), 1)
            h_ca_cont = torch.cat((h_cb, h_a_sty), 1)

            x_a_recon.append(self.gen_a.decode_recs(h_aa_cont))
            x_b_recon.append(self.gen_b.decode_cont(h_b))
            x_c_recon.append(self.gen_c.decode_recs(h_cc_cont))

            x_ab.append(x_ab_img)
            x_cb.append(x_cb_img)
            x_ba.append(self.gen_a.decode_recs(h_ba_cont))
            x_bc.append(self.gen_c.decode_recs(h_bc_cont))
            x_ac.append(self.gen_c.decode_recs(h_ac_cont))
            x_ca.append(self.gen_a.decode_recs(h_ca_cont))

            
        x_a_recon, x_b_recon, x_c_recon = torch.cat(x_a_recon), torch.cat(x_b_recon), torch.cat(x_c_recon)
        x_ba, x_bc = torch.cat(x_ba), torch.cat(x_bc)
        x_ab, x_ac = torch.cat(x_ab), torch.cat(x_ac)
        x_ca, x_cb = torch.cat(x_ca), torch.cat(x_cb)
        self.train()

        return x_a, x_a_recon, x_ab, x_ac, x_b, x_b_recon, x_ba, x_bc, x_c, x_c_recon, x_cb, x_ca

    def dis_update(self, x_a, x_b, x_c, hyperparameters, iteration=0):
        self.dis_opt.zero_grad()
        self.content_opt.zero_grad()
    
        h_a = self.gen_a.encode_cont(x_a)
        h_a_sty = self.gen_a.encode_sty(x_a)
        h_b = self.gen_b.encode_cont(x_b)
        h_c = self.gen_c.encode_cont(x_c)
        h_c_sty = self.gen_c.encode_sty(x_c)

        out_a = self.dis_content_ab(h_a)
        out_b_ab = self.dis_content_ab(h_b)
        out_c = self.dis_content_bc(h_c)
        out_b_bc = self.dis_content_bc(h_b)
        self.loss_ContentD = 0
        if hyperparameters['gan_type'] == 'lsgan':
            self.loss_ContentD += torch.mean((out_a - 0)**2) + torch.mean((out_b_ab - 1)**2 +
                                             (out_c - 0)**2) + torch.mean((out_b_bc - 1)**2)
        elif hyperparameters['gan_type'] == 'nsgan':
            all0_ab = Variable(torch.zeros_like(out_a.data).cuda(self.gpuid), requires_grad=False)
            all1_ab = Variable(torch.ones_like(out_b_ab.data).cuda(self.gpuid), requires_grad=False)
            all0_bc = Variable(torch.zeros_like(out_c.data).cuda(self.gpuid), requires_grad=False)
            all1_bc = Variable(torch.ones_like(out_b_bc.data).cuda(self.gpuid), requires_grad=False)
            self.loss_ContentD += torch.mean(F.binary_cross_entropy(F.sigmoid(out_a), all0_ab) +
                                   F.binary_cross_entropy(F.sigmoid(out_b_ab), all1_ab) +
                                             F.binary_cross_entropy(F.sigmoid(out_c), all0_bc) +
                                   F.binary_cross_entropy(F.sigmoid(out_b_bc), all1_bc))
        else:
            assert 0, "Unsupported GAN type: {}".format(hyperparameters['gan_type'])
        
        h_cat_ab = torch.cat((h_b, h_a_sty), 1)
        noise_b_ab = torch.randn(h_cat_ab.size()).cuda(h_cat_ab.data.get_device())
        x_ba = self.gen_a.decode_recs(h_cat_ab + noise_b_ab)
        noise_a = torch.randn(h_a.size()).cuda(h_a.data.get_device())
        x_ab = self.gen_b.decode_cont(h_a + noise_a)

        h_cat_cb = torch.cat((h_b, h_c_sty), 1)
        noise_b_bc = torch.randn(h_cat_cb.size()).cuda(h_cat_cb.data.get_device())
        x_bc = self.gen_c.decode_recs(h_cat_cb + noise_b_bc)
        noise_c = torch.randn(h_c.size()).cuda(h_c.data.get_device())
        x_cb = self.gen_b.decode_cont(h_c + noise_c)

        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b_ab = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_c = self.dis_c.calc_dis_loss(x_bc.detach(), x_c)
        self.loss_dis_b_bc = self.dis_b.calc_dis_loss(x_cb.detach(), x_b)

        self.loss_dis_total = hyperparameters['gan_w'] * (self.loss_dis_a + self.loss_dis_b_ab +
                                                          self.loss_dis_c + self.loss_dis_b_bc + self.loss_ContentD)

        # Dynamically adjusting loss term weights with BMTD algorithm
        # loss_adv_e = self.loss_ContentD
        # loss_adv_i = self.loss_dis_a + self.loss_dis_b_ab + self.loss_dis_c + self.loss_dis_b_bc
        # self.loss_dis_total = hyperparameters['gan_w'] * BMTD_algorithm(iteration, loss_adv_e, loss_adv_i)

        self.loss_dis_total.backward()        
        nn.utils.clip_grad_norm_(self.dis_content_ab.parameters(), 5) 
        nn.utils.clip_grad_norm_(self.dis_content_bc.parameters(), 5)
        self.dis_opt.step()
        self.content_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.content_scheduler is not None:
            self.content_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        self.gen_c.load_state_dict(state_dict['c'])
        iterations = int(last_model_name[-11:-3])
        last_model_name = get_model_list(checkpoint_dir, "dis_00188000")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        self.dis_c.load_state_dict(state_dict['c'])

        last_model_name = get_model_list(checkpoint_dir, "dis_Content")
        state_dict = torch.load(last_model_name)
        self.dis_content_ab.load_state_dict(state_dict['dis_c_ab'])
        self.dis_content_bc.load_state_dict(state_dict['dis_c_bc'])

        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        self.content_opt.load_state_dict(state_dict['dis_content'])
       
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        self.content_scheduler = get_scheduler(self.content_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):

        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        dis_Con_name = os.path.join(snapshot_dir, 'dis_Content_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict(), 'c': self.gen_c.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict(), 'c': self.dis_c.state_dict()}, dis_name)
        torch.save({'dis_c_ab':self.dis_content_ab.state_dict(), 'dis_c_bc':self.dis_content_bc.state_dict()}, dis_Con_name)

      
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict(), \
                                                    'dis_content':self.content_opt.state_dict()}, opt_name)