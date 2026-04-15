import os
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .networks import SEM, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, TVLoss
import math


class NoOpGradScaler:
    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return None


def _cuda_grad_scaler():
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler(device='cuda')
    return torch.cuda.amp.GradScaler()


def _cuda_autocast_context(enabled=True):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast(device_type='cuda', enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.checkpoints_dir = getattr(config, 'CHECKPOINTS_DIR', None) or os.path.join(config.PATH, 'checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.pretrain_from = os.path.abspath(config.PRETRAIN_FROM) if getattr(config, 'PRETRAIN_FROM', None) else None
        self.resume_from = os.path.abspath(config.RESUME_FROM) if getattr(config, 'RESUME_FROM', None) else None
        self.last_checkpoint_path = os.path.join(self.checkpoints_dir, 'last.pth')
        self.final_checkpoint_path = os.path.join(self.checkpoints_dir, 'final.pth')
        self.best_checkpoint_path = os.path.join(self.checkpoints_dir, 'best.pth')
        self.gen_weights_path = os.path.join(self.checkpoints_dir, name + '_gen.pth')
        self.dis_weights_path = os.path.join(self.checkpoints_dir, name + '_dis.pth')
        self.save_history = bool(int(getattr(config, 'SAVE_HISTORY', 0) or 0))
        self.history_dir = os.path.join(self.checkpoints_dir, 'history')
        if self.save_history:
            os.makedirs(self.history_dir, exist_ok=True)

        self.legacy_gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.legacy_dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')
        self.legacy_last_checkpoint_path = os.path.join(config.PATH, 'last.pth')

    def _torch_load(self, path):
        if self.config.DEVICE.type == 'cuda':
            return torch.load(path)
        return torch.load(path, map_location=lambda storage, loc: storage)

    def _build_checkpoint(self):
        checkpoint = {
            'iteration': self.iteration,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
        }
        if hasattr(self, 'gen_optimizer'):
            checkpoint['gen_optimizer'] = self.gen_optimizer.state_dict()
        if hasattr(self, 'dis_optimizer'):
            checkpoint['dis_optimizer'] = self.dis_optimizer.state_dict()
        if hasattr(self, 'gen_scheduler'):
            checkpoint['gen_scheduler'] = self.gen_scheduler.state_dict()
        if hasattr(self, 'dis_scheduler'):
            checkpoint['dis_scheduler'] = self.dis_scheduler.state_dict()
        if hasattr(self, 'scaler'):
            checkpoint['scaler'] = self.scaler.state_dict()
        return checkpoint

    def _extract_state_dict(self, data, preferred_key):
        if isinstance(data, dict):
            state_dict = data.get(preferred_key)
            if isinstance(state_dict, dict):
                return state_dict
            if data and all(isinstance(key, str) for key in data.keys()) and all(torch.is_tensor(value) for value in data.values()):
                return data
        raise KeyError(f'checkpoint does not contain a valid "{preferred_key}" state_dict')

    def _load_matching_state_dict(self, module, incoming_state_dict, module_name):
        current_state = module.state_dict()
        matched_state = {}
        skipped = []

        for key, value in incoming_state_dict.items():
            target = current_state.get(key)
            if target is None:
                skipped.append((key, 'missing_in_model'))
                continue
            if target.shape != value.shape:
                skipped.append((key, f'shape_mismatch checkpoint={tuple(value.shape)} model={tuple(target.shape)}'))
                continue
            matched_state[key] = value

        current_state.update(matched_state)
        module.load_state_dict(current_state, strict=False)

        if skipped:
            print(f'Partially loaded {module_name}: matched={len(matched_state)}, skipped={len(skipped)}')
            for key, reason in skipped[:20]:
                print(f'  skip {key}: {reason}')
            if len(skipped) > 20:
                print(f'  ... {len(skipped) - 20} more skipped keys')
        else:
            print(f'Fully loaded {module_name}: matched={len(matched_state)}')

    def _resolve_pretrain_paths(self, source):
        source = os.path.abspath(source)
        if os.path.isdir(source):
            return (
                os.path.join(source, self.name + '_gen.pth'),
                os.path.join(source, self.name + '_dis.pth'),
            )

        if source.endswith('_gen.pth'):
            return source, source.replace('_gen.pth', '_dis.pth')

        if source.endswith('_dis.pth'):
            return source.replace('_dis.pth', '_gen.pth'), source

        return source, None

    def _load_pretrain_weights(self, checkpoint_path, discriminator_path=None):
        data = self._torch_load(checkpoint_path)
        self._load_matching_state_dict(
            self.generator,
            self._extract_state_dict(data, 'generator'),
            f'{self.name} generator',
        )
        self.iteration = 0

        if self.config.MODE == 1:
            dis_data = None
            if discriminator_path is not None and os.path.exists(discriminator_path):
                dis_data = self._torch_load(discriminator_path)
            elif isinstance(data, dict) and isinstance(data.get('discriminator'), dict):
                dis_data = data

            if dis_data is not None:
                print('Loading %s discriminator...' % self.name)
                self._load_matching_state_dict(
                    self.discriminator,
                    self._extract_state_dict(dis_data, 'discriminator'),
                    f'{self.name} discriminator',
                )

    def _load_legacy_checkpoint(self):
        data = self._torch_load(self.legacy_gen_weights_path)
        self.generator.load_state_dict(self._extract_state_dict(data, 'generator'), strict=False)
        if isinstance(data, dict) and 'iteration' in data:
            self.iteration = data['iteration']
        else:
            self.iteration = 0

        if self.config.MODE == 1 and os.path.exists(self.legacy_dis_weights_path):
            print('Loading %s discriminator...' % self.name)
            dis_data = self._torch_load(self.legacy_dis_weights_path)
            self.discriminator.load_state_dict(self._extract_state_dict(dis_data, 'discriminator'))

    def _load_resume_checkpoint(self, checkpoint_path):
        data = self._torch_load(checkpoint_path)
        self.generator.load_state_dict(data['generator'], strict=False)
        self.iteration = data.get('iteration', 0)

        if self.config.MODE == 1:
            if data.get('discriminator') is not None:
                print('Loading %s discriminator...' % self.name)
                self.discriminator.load_state_dict(data['discriminator'])
            if data.get('gen_optimizer') is not None:
                self.gen_optimizer.load_state_dict(data['gen_optimizer'])
            if data.get('dis_optimizer') is not None:
                self.dis_optimizer.load_state_dict(data['dis_optimizer'])
            if data.get('gen_scheduler') is not None:
                self.gen_scheduler.load_state_dict(data['gen_scheduler'])
            if data.get('dis_scheduler') is not None:
                self.dis_scheduler.load_state_dict(data['dis_scheduler'])
            if data.get('scaler') is not None and hasattr(self.scaler, 'load_state_dict'):
                self.scaler.load_state_dict(data['scaler'])

    def _save_history_checkpoint(self, checkpoint, suffix=''):
        if not self.save_history:
            return

        filename = f'iter_{int(self.iteration):07d}{suffix}.pth'
        torch.save(checkpoint, os.path.join(self.history_dir, filename))

    def _auto_resume_candidates(self):
        candidates = []
        for path in [self.last_checkpoint_path, self.legacy_last_checkpoint_path]:
            if path is None:
                continue
            normalized = os.path.abspath(path)
            if normalized not in candidates:
                candidates.append(normalized)
        return candidates

    def load(self):
        if self.resume_from is not None:
            if not os.path.exists(self.resume_from):
                raise FileNotFoundError(f"resume checkpoint not found: {self.resume_from}")
            print('Loading %s generator...' % self.name)
            print(f"Resume checkpoint: {self.resume_from}")
            self._load_resume_checkpoint(self.resume_from)
            return

        if self.pretrain_from is not None:
            gen_weights_path, dis_weights_path = self._resolve_pretrain_paths(self.pretrain_from)
            if not os.path.exists(gen_weights_path):
                raise FileNotFoundError(f"pretrain checkpoint not found: {gen_weights_path}")
            print('Loading %s generator...' % self.name)
            print(f"Pretrain checkpoint: {gen_weights_path}")
            print('Initializing weights only; iteration/optimizer state will be reset.')
            self._load_pretrain_weights(gen_weights_path, dis_weights_path)
            return

        for candidate in self._auto_resume_candidates():
            if not os.path.exists(candidate):
                continue
            print('Loading %s generator...' % self.name)
            print(f"Auto-resume checkpoint: {candidate}")
            self._load_resume_checkpoint(candidate)
            return

        if os.path.exists(self.legacy_gen_weights_path):
            print('Loading %s generator...' % self.name)
            print(f"Legacy checkpoint: {self.legacy_gen_weights_path}")
            if self.config.MODE == 1:
                print('Legacy split weights restore model weights and iteration only; optimizer/scheduler/scaler state will not be resumed.')
            self._load_legacy_checkpoint()

    def save(self):
        print('\nsaving %s...\n' % self.name)
        checkpoint = self._build_checkpoint()
        torch.save(checkpoint, self.last_checkpoint_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)
        self._save_history_checkpoint(checkpoint)

    def save_final(self):
        print('\nsaving final %s...\n' % self.name)
        checkpoint = self._build_checkpoint()
        torch.save(checkpoint, self.final_checkpoint_path)
        self._save_history_checkpoint(checkpoint, '_final')

    def save_best(self):
        print('\nsaving best %s...\n' % self.name)
        checkpoint = self._build_checkpoint()
        torch.save(checkpoint, self.best_checkpoint_path)
        self._save_history_checkpoint(checkpoint, '_best')



class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)


        generator = SEM(config=config)
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

        #### learning rate decay
        self.gen_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.gen_optimizer, last_epoch=-1, milestones=[20000, 40000,60000,80000,120000], gamma=self.config.LR_Decay)
        self.dis_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.dis_optimizer, last_epoch=-1,
                                                                  milestones=[20000, 40000,60000,80000,120000], gamma=self.config.LR_Decay)
        self.enable_lr_scheduler = bool(int(getattr(config, 'ENABLE_LR_SCHEDULER', 0) or 0))
                                                                  
                                                                  
        if self.config.DEVICE.type == 'cuda':
            self.scaler = _cuda_grad_scaler()
        else:
            self.scaler = NoOpGradScaler()

    def _autocast_context(self):
        if self.config.DEVICE.type == 'cuda':
            return _cuda_autocast_context()
        return nullcontext()

    def process(self, images, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs

        outputs_img = self(images, masks)

        
        gen_loss = 0
        dis_loss = 0
        
        
        with self._autocast_context():


        # discriminator loss
            dis_input_real = images
            dis_input_fake = outputs_img.detach()


            dis_real, _ = self.discriminator(dis_input_real)                   
            dis_fake, _ = self.discriminator(dis_input_fake)                   

            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
            gen_input_fake = outputs_img
            gen_fake, _ = self.discriminator(gen_input_fake)
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_loss += gen_gan_loss


        # generator l1 loss
            gen_l1_loss = self.l1_loss(outputs_img, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        #gen_l1_loss = self.l1_loss(outputs_img, images) * self.config.L1_LOSS_WEIGHT
            gen_loss += gen_l1_loss


        # generator perceptual loss
            gen_content_loss = self.perceptual_loss(outputs_img, images)
            gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
            gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs_img * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss
        #############################

        logs = [
            ("gLoss", gen_loss.item()),
            ("dLoss", dis_loss.item())
        ]

        return outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss

    # def forward(self, images, landmarks, masks):
    def forward(self, images, masks):
        images_masked = (images * (1 - masks).float()) + masks
        inputs = images_masked
        scaled_masks_tiny = F.interpolate(masks, size=[int(masks.shape[2] / 8), int(masks.shape[3] / 8)],
                                     mode='nearest')        
        
        scaled_masks_quarter = F.interpolate(masks, size=[int(masks.shape[2] / 4), int(masks.shape[3] / 4)],
                                     mode='nearest')
        scaled_masks_half = F.interpolate(masks, size=[int(masks.shape[2] / 2), int(masks.shape[3] / 2)],
                                     mode='nearest')
                                     
        seq_len_level1 = masks.shape[2] * masks.shape[3]
        seq_len_level2 = scaled_masks_half.shape[2] * scaled_masks_half.shape[3]
        seq_len_level3 = scaled_masks_quarter.shape[2] * scaled_masks_quarter.shape[3]
        seq_len_level4 = scaled_masks_tiny.shape[2] * scaled_masks_tiny.shape[3]

        pos1 = PositionalEncoding(48, seq_len_level1, device=images.device)
        pos2 = PositionalEncoding(96, seq_len_level2, device=images.device)
        pos3 = PositionalEncoding(192, seq_len_level3, device=images.device)
        pos4 = PositionalEncoding(384, seq_len_level4, device=images.device)
        pos1_dec = PositionalEncoding(96, seq_len_level1, device=images.device)
        
        

        outputs_img = self.generator(inputs,masks,scaled_masks_half,scaled_masks_quarter,scaled_masks_tiny, pos1, pos2, pos3, pos4, pos1_dec)             
        return outputs_img

    def backward(self, gen_loss = None, dis_loss = None):
        self.scaler.scale(dis_loss).backward(retain_graph=True)
        self.scaler.scale(gen_loss).backward()

        self.scaler.step(self.dis_optimizer)
        self.scaler.step(self.gen_optimizer)
        self.scaler.update()

        if self.enable_lr_scheduler:
            self.gen_scheduler.step()
            self.dis_scheduler.step()

        print(self.gen_optimizer.param_groups[0]['lr'])

    def backward_joint(self, gen_loss = None, dis_loss = None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()



def abs_smooth(x):
    absx = torch.abs(x)
    minx = torch.min(absx,other=torch.ones(absx.shape).cuda())
    r = 0.5 *((absx-1)*minx + absx)
    return r
    
def PositionalEncoding(d_model, max_len=5000, device=None):
    max_len = int(max_len)
    if device is None:
        device = torch.device("cpu")

    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe
