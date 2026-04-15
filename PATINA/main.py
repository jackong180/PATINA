import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.experiment import ExperimentLogger
from src.sem import sem


def infer_run_dir_from_resume(resume_from):
    checkpoint_path = os.path.abspath(resume_from)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if os.path.basename(checkpoint_dir) == 'checkpoints':
        return os.path.dirname(checkpoint_dir)
    return None


def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, reads from config file if not specified
    """
    config = load_config(mode)
    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # init device
    if config.GPU is not None and len(config.GPU) == 0:
        print('GPU list is empty, use cpu')
        config.DEVICE = torch.device("cpu")
    elif torch.cuda.is_available():
        print('Cuda is available')
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        print('Cuda is unavailable, use cpu')
        config.DEVICE = torch.device("cpu")



    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)


    # initialize random seed
    torch.manual_seed(config.SEED)
    if config.DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)



    # build the model and initialize
    model = sem(config)
    model.load()


    # model training
    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()



def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, reads from config file if not specified
    """

    project_root = os.path.dirname(os.path.abspath(__file__))
    default_checkpoints_path = os.path.join(project_root, 'checkpoints')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints_irr_45000_tv0_beta0.5_wopixelshuffle_inputwithmask', help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--path', '--checkpoints', type=str, default=default_checkpoints_path,
                        help='model checkpoints path (default: project_root/checkpoints)')
    parser.add_argument('--exp_name', type=str, default='PATINA',
                        help='experiment group name under the outputs root (default: PATINA)')
    parser.add_argument('--outputs_dir', type=str, default='../outputs',
                        help='root directory for all experiment outputs, resolved from project root (default: ../outputs)')
    parser.add_argument('--run_dir', type=str, default=None,
                        help='reuse an existing run directory for resume/test')
    parser.add_argument('--pretrain_from', type=str, default=None,
                        help='official pretrain source used only for weight initialization; ignores iteration/optimizer state')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='explicit user-run checkpoint path for full resume/evaluation (e.g. outputs/<exp>/<run>/checkpoints/last.pth)')
    parser.add_argument('--skip_src_backup', action='store_true',
                        help='skip backing up main.py/src/requirements.txt into the run directory')
    parser.add_argument('--mode', type=int, choices=[1, 2], default=None,
                        help='override run mode: 1=train, 2=test')
    # parser.add_argument(('--save_path', type=str, default='./checkpoints1' ,help='model save path (default: ./checkpoints1)'))


    # parser.add_argument('--model', type=int, default='3',choices=[1, 2, 3], help='1: landmark prediction model, 2: inpaint model, 3: joint model')
    parser.add_argument('--model', type=int, default='2', choices=[1, 2, 3],
                        help='1: landmark prediction model, 2: inpaint model, 3: joint model')

    parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
    parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
    parser.add_argument('--landmark', type=str, help='path to the landmarks directory or a landmark file')
    parser.add_argument('--output', type=str, help='path to the output directory')

    args = parser.parse_args()
    if not os.path.isabs(args.path):
        args.path = os.path.abspath(os.path.join(project_root, args.path))
    if args.pretrain_from is not None:
        args.pretrain_from = os.path.abspath(args.pretrain_from)
    if args.resume_from is not None:
        args.resume_from = os.path.abspath(args.resume_from)
        if args.run_dir is None:
            inferred_run_dir = infer_run_dir_from_resume(args.resume_from)
            if inferred_run_dir is not None:
                args.run_dir = inferred_run_dir
    if args.pretrain_from is not None and args.resume_from is not None:
        raise ValueError('--pretrain_from and --resume_from are mutually exclusive')

    effective_mode = mode if mode is not None else args.mode
    if effective_mode == 2 and args.resume_from is None:
        raise ValueError(
            '--resume_from is required in --mode 2 so evaluation always targets an explicit run checkpoint '
            '(for example outputs/<exp>/<run>/checkpoints/best.pth or final.pth).'
        )

    config_path = os.path.join(args.path, 'config.yml')


    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        config_template_path = os.path.join(project_root, 'config.yml.example')
        if not os.path.exists(config_template_path):
            raise FileNotFoundError(
                'config.yml not found at {} and template is missing at {}. '
                'Please pass --path to an existing checkpoints directory.'.format(
                    config_path,
                    config_template_path,
                )
            )
        copyfile(config_template_path, config_path)

    # load config file
    config = Config(config_path)
    print(config_path)
    config.PRETRAIN_FROM = args.pretrain_from
    config.RESUME_FROM = args.resume_from

    # train mode
    if effective_mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model

    # test mode
    elif effective_mode == 2:
        config.MODE = 2
        config.MODEL = args.model if args.model is not None else 3

        if args.input is not None:
            config.TEST_INPAINT_IMAGE_FLIST = args.input

        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask


        if args.output is not None:
            config.RESULTS = args.output

    ExperimentLogger(config, args, project_root).prepare()

    return config


if __name__ == "__main__":
    main()
