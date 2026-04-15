import os
import yaml


def _serialize_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(v) for v in value]
    return str(value)


class Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.safe_load(self._yaml)
            self._dict['PATH'] = os.path.dirname(config_path)
            self._dict['CONFIG_PATH'] = os.path.abspath(config_path)

    def __setattr__(self, name, value):
        if name.startswith('_') or '_dict' not in self.__dict__:
            super().__setattr__(name, value)
            return

        self._dict[name] = value
        super().__setattr__(name, value)
 
    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        if DEFAULT_CONFIG.get(name) is not None:
            return DEFAULT_CONFIG[name]

        return None

    def to_dict(self):
        data = dict(self._dict)
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                data[key] = value
        return _serialize_value(data)

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True))
        print('')
        print('---------------------------------')
        print('')

DEFAULT_CONFIG = {
    'MODE': 1,                      # 1: train, 2: test, 3: eval
    'MODEL': 1,                     # 1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model
    'MASK': 3,                      # 1: random block, 2: half, 3: external, 4: (external, random block), 5: (external, random block, half)
    'NMS': 1,                       # 0: no non-max-suppression, 1: applies non-max-suppression on the external edges by multiplying by Canny
    'SEED': 10,                     # random seed
    'GPU': [0],                     # list of gpu ids
    'AUGMENTATION_TRAIN': 0,        # 1: train 0: false use augmentation to train landmark predictor

    'LR': 0.0001,                   # learning rate
    'D2G_LR': 0.1,                  # discriminator/generator learning rate ratio
    'BETA1': 0.0,                   # adam optimizer beta1
    'BETA2': 0.9,                   # adam optimizer beta2
    'BATCH_SIZE': 4,                # input batch size for training
    'INPUT_SIZE': 256,              # input image size for training 0 for original size
    'MAX_ITERS': 30000,             # maximum number of iterations to train the model

    'L1_LOSS_WEIGHT': 1,            # l1 loss weight
    'STYLE_LOSS_WEIGHT': 1,         # style loss weight
    'CONTENT_LOSS_WEIGHT': 1,       # perceptual loss weight
    'INPAINT_ADV_LOSS_WEIGHT': 0.01,# adversarial loss weight
    'TV_LOSS_WEIGHT': 0.1,          # total variation loss weight

    'GAN_LOSS': 'lsgan',            # nsgan | lsgan | hinge
    'GAN_POOL_SIZE': 0,             # fake images pool size

    'SAVE_INTERVAL': 500,           # how many iterations to wait before saving model (0: never)
    'VISUALIZE_INTERVAL': 1000,     # how many iterations to wait before saving train visualizations (0: never)
    'SAMPLE_INTERVAL': 1000,        # how many iterations to wait before sampling (0: never)
    'SAMPLE_SIZE': 12,              # number of images to sample
    'EVAL_INTERVAL': 500,           # how many iterations to wait before model evaluation (0: never)
    'LOG_INTERVAL': 50,             # how many iterations to wait before logging training status (0: never)
    'SAVE_HISTORY': 0,              # 1: keep per-save full checkpoint snapshots under checkpoints/history
    'AUTO_TEST_AFTER_TRAIN': 0,     # 1: run MODE=2 metrics once after training ends and save a dedicated summary
    'VAL_INPAINT_IMAGE_FLIST': None,# optional validation image flist for post-train auto evaluation
    'VAL_MASK_FLIST': None,         # optional validation mask flist for post-train auto evaluation
    'VAL_MASK_BUCKETS': None,       # optional bucketed validation mask configs
    'ENABLE_LR_SCHEDULER': 1,       # 1: enable stepping MultiStepLR during training
    'TRAIN_NUM_WORKERS': 8,         # training DataLoader worker count
    'TRAIN_PIN_MEMORY': 1,          # pin host memory before transfer to GPU
    'TRAIN_PERSISTENT_WORKERS': 1,  # keep DataLoader workers alive across epochs
    'TRAIN_PREFETCH_FACTOR': 4,     # number of prefetched batches per worker
    'BEST_MONITOR': 'masked_l1_ave',# validation metric used to select checkpoints
    'BEST_MONITOR_MODE': 'min',     # min|max for BEST_MONITOR
    'FID_MODE': 'clean',            # cleanfid mode for bucketed evaluation
    'LCBC_LATENT_ENABLE': 1,        # 1: enable LCBC after the latent SEM stack
    'LCBC_LATENT_EMBED_DIM': 96,    # token embedding width for LCBC matching
    'LCBC_SOFTMAX_SCALE': 10.0,     # matching sharpness for LCBC
    'LCBC_LATENT_RES_SCALE_INIT': 0.03, # PATINA default: keep LCBC active but friendly to official SEM pretrain
    'MRDA_USE_BIAS': 0,             # 1: enable bias in embedded MRDA feature projection
    'MRDA_STAGE1_ENABLE': 1,        # 1: couple corrective MRDA residual into down1_2
    'MRDA_STAGE2_ENABLE': 1,        # 1: couple corrective MRDA residual into down2_3
    'MRDA_STAGE3_ENABLE': 1,        # 1: couple corrective MRDA residual into down3_4
    'MRDA_STAGE1_RES_SCALE_INIT': 0.00, # start stage1 at zero and let training decide whether shallow repair should grow
    'MRDA_STAGE2_RES_SCALE_INIT': 0.06, # moderate mid-scale corrective MRDA strength
    'MRDA_STAGE3_RES_SCALE_INIT': 0.10, # deep-stage corrective MRDA strength for large holes
    'DFCC_GROUPS': 1,               # DFCC frequency branch groups
    'DFCC_FFT_NORM': 'ortho',       # DFCC FFT normalization mode
    'DFCC_LATENT_ENABLE': 1,        # 1: enable latent DFCC residual branch
    'DFCC_DECODER3_ENABLE': 1,      # 1: enable decoder level3 DFCC residual branch
    'DFCC_LATENT_INPUT_SHAPE': 32,  # latent feature map size at 256 input
    'DFCC_DECODER3_INPUT_SHAPE': 64,# decoder level3 feature map size at 256 input
    'DFCC_LATENT_RES_SCALE_INIT': 0.04, # PATINA latent DFCC scale
    'DFCC_DECODER3_RES_SCALE_INIT': 0.02, # PATINA decoder DFCC scale
    'DFCC_MASK_CONTEXT_KERNEL': 5,  # expand hole context for frequency refinement near missing boundaries
    'LATENT_BRANCH_MIXER_ENABLE': 1,# 1: adaptively mix LCBC/DFCC latent corrections instead of serial hard stacking
    'LATENT_BRANCH_MIXER_TEMPERATURE': 1.0, # initial softmax temperature for latent branch competition
    'PATINA_PRECONDITION_ENABLE': 1, # PATINA: keep deep preconditioning available with zero-init residual scale
    'PATINA_MASK_ROUTE_ENABLE': 1,   # PATINA: keep deep route enabled with micro initialization on 4090
    'PATINA_SKIP_FUSION_ENABLE': 1,  # PATINA: reinterpret as pretrained-aware residual skip adapter
    'PATINA_REFINEMENT_ENABLE': 1,   # PATINA: enable residual refinement head backed by pretrained refinement blocks
    'PATINA_PRECONDITION_SCALE_INIT': 0.00, # zero-init residual precondition keeps initial behavior close to baseline
    'PATINA_MASK_ROUTE_SCALE_INIT': 0.005,  # micro routed-Mamba initialization won the stabilized 4090 smoke test
    'PATINA_SKIP_ADAPTER_RES_SCALE_INIT': 0.02, # slightly stronger skip residual won the 4090 smoke test
    'PATINA_REFINEMENT_RES_SCALE_INIT': 0.00,   # zero-init output refinement residual
}
