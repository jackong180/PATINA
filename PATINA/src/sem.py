import os
import json
import warnings
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from torch.utils.data import DataLoader
from imageio.v2 import imread
from .dataset import Dataset
from .models import InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR
from cv2 import circle
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
import torchvision
import time
try:
    from cleanfid import fid as clean_fid
except ImportError:
    clean_fid = None

try:
    import wandb
except ImportError:
    wandb = None

'''
This repo is modified basing on Edge-Connect
https://github.com/knazeri/edge-connect
'''

class sem():
    def __init__(self, config):
        self.config = config


        if config.MODEL == 2:
            model_name = 'inpaint'

        self.debug = False
        self.model_name = model_name

        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)
        self.transf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.loss_fn_vgg = self._build_lpips_loss().to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.cal_mae = nn.L1Loss(reduction='sum')
        self.test_dataset = None
        self.results_path = getattr(config, 'VISUALIZATIONS_DIR', None) or os.path.join(config.PATH, 'results')
        self._mask_ratio_cache = {}
        self._test_image_paths_cache = None
        self._default_test_dataset_meta = None

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        #train mode
        if self.config.MODE == 1:

            if self.config.MODEL == 2:
                self.train_dataset = Dataset(config, config.TRAIN_INPAINT_IMAGE_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)

        # test mode
        if self.config.MODE == 2:
            if self.config.MODEL == 2:
                print('model = inpaint model')
                self._ensure_test_dataset()

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        logs_dir = getattr(config, 'LOGS_DIR', None) or config.PATH
        self.log_file = os.path.join(logs_dir, 'log_' + model_name + '.dat')
        self.test_summary_path = self._default_test_summary_path(logs_dir)
        self.best_metric_path = os.path.join(logs_dir, 'best_metric.json')
        self._eval_save_visualizations = True
        self._eval_log_per_sample = True
        self._eval_compute_fid = True
        self._eval_bucket_configs_override = None
        self.fid_mode = getattr(config, 'FID_MODE', None) or 'clean'
        self._best_metric_name, self._best_metric_mode = self._get_best_metric_spec()
        self._best_metric_value = None
        self._best_metric_iteration = None
        self._load_best_metric_state()

    def _build_lpips_loss(self):
        # lpips currently constructs torchvision backbones with the legacy
        # pretrained=True API. Suppress only those dependency warnings locally.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r"The parameter 'pretrained' is deprecated since 0\.13.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                'ignore',
                message=r"Arguments other than a weight enum or `None` for 'weights' are deprecated since 0\.13.*",
                category=UserWarning,
            )
            return lpips.LPIPS(net='vgg')

    def _default_test_summary_path(self, logs_dir):
        if self.config.MODE == 2 and getattr(self.config, 'RESUME_FROM', None):
            eval_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            return os.path.join(logs_dir, f'test_metrics_summary_{eval_stamp}.json')
        return os.path.join(logs_dir, 'test_metrics_summary.json')

    def load(self):


        if self.config.MODEL == 2:
            self.inpaint_model.load()


    def save(self):
 
        if self.config.MODEL == 2:
            self.inpaint_model.save()

    def save_final(self):
 
        if self.config.MODEL == 2:
            self.inpaint_model.save_final()

    def _ensure_test_dataset(self):
        if self.config.MODEL != 2:
            return None

        if self.test_dataset is None:
            plan = self._build_test_dataset_plan(
                name='all',
                mask_source=getattr(self.config, 'TEST_MASK_FLIST', None),
                ratio_range=None,
                results_root=self.results_path,
            )
            self.test_dataset = plan['dataset']
            self._default_test_dataset_meta = {
                'mask_source': plan['mask_source'],
                'mask_paths': list(plan['mask_paths']),
                'effective_mask_paths': list(plan['effective_mask_paths']),
                'source_mask_count': plan['source_mask_count'],
                'effective_mask_count': plan['effective_mask_count'],
                'unique_effective_mask_count': plan['unique_effective_mask_count'],
                'effective_mask_repeats': plan['effective_mask_repeats'],
                'mask_schedule_mode': plan['mask_schedule_mode'],
                'ratio_range': plan['ratio_range'],
                'test_image_count': plan['test_image_count'],
            }

        return self.test_dataset

    def _should_auto_test_after_train(self):
        return bool(int(getattr(self.config, 'AUTO_TEST_AFTER_TRAIN', 0) or 0))

    def _get_best_metric_spec(self):
        metric_name = getattr(self.config, 'BEST_MONITOR', None) or 'masked_l1_ave'
        metric_mode = str(getattr(self.config, 'BEST_MONITOR_MODE', None) or 'min').lower()
        if metric_mode not in {'min', 'max'}:
            raise ValueError('BEST_MONITOR_MODE must be "min" or "max"')
        return metric_name, metric_mode

    def _load_best_metric_state(self):
        if self.config.MODE != 1 or not os.path.exists(self.best_metric_path):
            return
        try:
            payload = json.loads(open(self.best_metric_path, 'r', encoding='utf-8').read())
        except (OSError, json.JSONDecodeError):
            return

        if payload.get('metric_name') != self._best_metric_name or payload.get('metric_mode') != self._best_metric_mode:
            return

        self._best_metric_value = payload.get('metric_value')
        self._best_metric_iteration = payload.get('iteration')

    def _save_best_metric_state(self, metric_value, iteration, summary_path=None):
        payload = {
            'metric_name': self._best_metric_name,
            'metric_mode': self._best_metric_mode,
            'metric_value': float(metric_value),
            'iteration': int(iteration),
            'checkpoint_path': getattr(self.inpaint_model, 'best_checkpoint_path', None),
            'summary_path': summary_path,
        }
        with open(self.best_metric_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _summaries_metric_value(self, summaries):
        values = []
        for summary in summaries:
            if summary.get('sample_count', 0) <= 0:
                continue
            value = summary.get(self._best_metric_name)
            if value is None:
                continue
            values.append(float(value))
        if not values:
            return None
        return float(np.mean(values))

    def _is_better_metric(self, metric_value):
        if metric_value is None:
            return False
        if self._best_metric_value is None:
            return True
        if self._best_metric_mode == 'min':
            return metric_value < self._best_metric_value
        return metric_value > self._best_metric_value

    def _maybe_update_best_checkpoint(self, iteration, summaries, summary_path=None):
        metric_value = self._summaries_metric_value(summaries)
        if metric_value is None:
            print('Skipping best-checkpoint update because the monitored validation metric is unavailable.')
            return

        if not self._is_better_metric(metric_value):
            print(
                'Validation monitor {}={} did not beat current best {} (mode={}).'.format(
                    self._best_metric_name,
                    metric_value,
                    self._best_metric_value,
                    self._best_metric_mode,
                )
            )
            return

        self._best_metric_value = metric_value
        self._best_metric_iteration = int(iteration)
        self.inpaint_model.save_best()
        self._save_best_metric_state(metric_value, iteration, summary_path=summary_path)
        print(
            'Updated best checkpoint with {}={} at iteration {}.'.format(
                self._best_metric_name,
                metric_value,
                iteration,
            )
        )

    def _run_test_with_targets(
        self,
        results_path=None,
        summary_path=None,
        test_image_flist=None,
        test_mask_flist=None,
        test_mask_buckets=None,
        save_visualizations=True,
        log_per_sample=True,
        compute_fid=True,
    ):
        original_results_path = self.results_path
        original_summary_path = self.test_summary_path
        original_test_image_flist = getattr(self.config, 'TEST_INPAINT_IMAGE_FLIST', None)
        original_test_mask_flist = getattr(self.config, 'TEST_MASK_FLIST', None)
        original_test_dataset = self.test_dataset
        original_test_image_paths_cache = self._test_image_paths_cache
        original_default_test_dataset_meta = self._default_test_dataset_meta
        original_mask_ratio_cache = dict(self._mask_ratio_cache)
        original_eval_save_visualizations = self._eval_save_visualizations
        original_eval_log_per_sample = self._eval_log_per_sample
        original_eval_compute_fid = self._eval_compute_fid
        original_bucket_configs_override = self._eval_bucket_configs_override

        if results_path is not None:
            self.results_path = results_path
        if summary_path is not None:
            self.test_summary_path = summary_path
        if test_image_flist is not None:
            self.config.TEST_INPAINT_IMAGE_FLIST = test_image_flist
        if test_mask_flist is not None:
            self.config.TEST_MASK_FLIST = test_mask_flist
        self._eval_save_visualizations = save_visualizations
        self._eval_log_per_sample = log_per_sample
        self._eval_compute_fid = compute_fid
        self._eval_bucket_configs_override = test_mask_buckets

        self.test_dataset = None
        self._test_image_paths_cache = None
        self._default_test_dataset_meta = None
        self._mask_ratio_cache = {}

        try:
            summaries = self.test()
        finally:
            self.results_path = original_results_path
            self.test_summary_path = original_summary_path
            self.config.TEST_INPAINT_IMAGE_FLIST = original_test_image_flist
            self.config.TEST_MASK_FLIST = original_test_mask_flist
            self.test_dataset = original_test_dataset
            self._test_image_paths_cache = original_test_image_paths_cache
            self._default_test_dataset_meta = original_default_test_dataset_meta
            self._mask_ratio_cache = original_mask_ratio_cache
            self._eval_save_visualizations = original_eval_save_visualizations
            self._eval_log_per_sample = original_eval_log_per_sample
            self._eval_compute_fid = original_eval_compute_fid
            self._eval_bucket_configs_override = original_bucket_configs_override

        return summaries

    def _get_validation_targets(self):
        val_image_flist = getattr(self.config, 'VAL_INPAINT_IMAGE_FLIST', None)
        val_mask_flist = getattr(self.config, 'VAL_MASK_FLIST', None)
        if val_image_flist and val_mask_flist:
            return {
                'label': 'val',
                'image_flist': val_image_flist,
                'mask_flist': val_mask_flist,
                'mask_buckets': getattr(self.config, 'VAL_MASK_BUCKETS', None),
            }
        return None

    def run_validation_evaluation(self, iteration, save_visualizations=False, log_per_sample=False):
        if self.config.MODEL != 2:
            print('Skipping validation evaluation because MODEL!=2.')
            return None, None

        eval_targets = self._get_validation_targets()
        if eval_targets is None:
            print('Skipping validation evaluation because VAL_INPAINT_IMAGE_FLIST/VAL_MASK_FLIST are not configured.')
            return None, None

        iteration_tag = 'iter_{:07d}'.format(int(iteration))
        if save_visualizations:
            eval_results_path = os.path.join(self.results_path, 'post_train_eval', eval_targets['label'], iteration_tag)
        else:
            eval_results_path = os.path.join(self.results_path, 'periodic_eval', eval_targets['label'], iteration_tag)
        eval_summary_path = os.path.join(
            getattr(self.config, 'LOGS_DIR', None) or self.config.PATH,
            '{}_metrics_summary_{}.json'.format(eval_targets['label'], iteration_tag)
        )

        print('\nstart {} evaluation at iteration {}...\n'.format(eval_targets['label'], iteration))
        summaries = self._run_test_with_targets(
            results_path=eval_results_path,
            summary_path=eval_summary_path,
            test_image_flist=eval_targets['image_flist'],
            test_mask_flist=eval_targets['mask_flist'],
            test_mask_buckets=eval_targets.get('mask_buckets'),
            save_visualizations=save_visualizations,
            log_per_sample=log_per_sample,
            compute_fid=save_visualizations,
        )
        return summaries, eval_summary_path

    def run_post_train_evaluation(self):
        return self.run_validation_evaluation(
            iteration=self.inpaint_model.iteration,
            save_visualizations=True,
            log_per_sample=True,
        )


    def train(self):
        
        train_num_workers = int(getattr(self.config, 'TRAIN_NUM_WORKERS', 8) or 0)
        train_pin_memory = bool(int(getattr(self.config, 'TRAIN_PIN_MEMORY', 1) or 0)) and self.config.DEVICE.type == 'cuda'
        train_persistent_workers = bool(int(getattr(self.config, 'TRAIN_PERSISTENT_WORKERS', 1) or 0)) and train_num_workers > 0
        train_prefetch_factor = int(getattr(self.config, 'TRAIN_PREFETCH_FACTOR', 4) or 2)
        train_loader_kwargs = dict(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=train_num_workers,
            drop_last=True,
            shuffle=True,
            pin_memory=train_pin_memory,
        )
        if train_num_workers > 0:
            train_loader_kwargs['persistent_workers'] = train_persistent_workers
            train_loader_kwargs['prefetch_factor'] = train_prefetch_factor

        train_loader = DataLoader(**train_loader_kwargs)


        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        eval_interval = int(getattr(self.config, 'EVAL_INTERVAL', 0) or 0)
        visualize_interval = int(getattr(self.config, 'VISUALIZE_INTERVAL', 40) or 0)
        total = len(self.train_dataset)
        
        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)


            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])
            

            for items in train_loader:
                

            
                self.inpaint_model.train()

                if model == 2:
                    images, masks = self.cuda(*items)

                    outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss = self.inpaint_model.process(images,masks)
                    outputs_merged = (outputs_img * masks) + (images * (1-masks))

                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()

                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration

                stop_training = iteration >= max_iteration

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])
                if iteration % 10 == 0 and wandb is not None and wandb.run is not None:
                        wandb.log({'gen_loss': gen_loss, 'l1_loss': gen_l1_loss, 'style_loss': gen_style_loss,
                                   'perceptual loss': gen_content_loss, 'gen_gan_loss': gen_gan_loss,
                                   'dis_loss': dis_loss}, step=iteration)
			 
                if visualize_interval and iteration % visualize_interval == 0:
                    create_dir(self.results_path)
                    inputs = (images * (1 - masks))
                    images_joint = stitch_images(
                        self.postprocess(images),
                        self.postprocess(inputs),
                        self.postprocess(outputs_img),
                        self.postprocess(outputs_merged),
                        img_per_row=1
                    )
                                                        

                    path_masked = os.path.join(self.results_path,self.model_name,'masked')
                    path_result = os.path.join(self.results_path, self.model_name,'result')
                    path_joint = os.path.join(self.results_path,self.model_name,'joint')
                    
                    name = 'epoch_{:04d}_iter_{:07d}.png'.format(epoch, iteration)

                    create_dir(path_masked)
                    create_dir(path_result)
                    create_dir(path_joint)
                    

                    masked_images = self.postprocess(images*(1-masks)+masks)[0]
                    images_result = self.postprocess(outputs_merged)[0]

                    print(os.path.join(path_joint,name[:-4]+'.png'))
                   
                    images_joint.save(os.path.join(path_joint,name[:-4]+'.png'))
                    imsave(masked_images,os.path.join(path_masked,name))
                    imsave(images_result,os.path.join(path_result,name))

                    print(name + ' complete!')
                    
                ##############


                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)



                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()
                if eval_interval and iteration % eval_interval == 0:
                    summaries, summary_path = self.run_validation_evaluation(
                        iteration=iteration,
                        save_visualizations=False,
                        log_per_sample=False,
                    )
                    if summaries is not None:
                        self._maybe_update_best_checkpoint(iteration, summaries, summary_path=summary_path)
                if stop_training:
                    keep_training = False
                    break
        if self.inpaint_model.iteration > 0:
            self.save()
            self.save_final()
            if self._should_auto_test_after_train():
                summaries, summary_path = self.run_post_train_evaluation()
                if summaries is not None:
                    self._maybe_update_best_checkpoint(self.inpaint_model.iteration, summaries, summary_path=summary_path)
        print('\nEnd training....')


    def test(self):

        self.inpaint_model.eval()
        self._ensure_test_dataset()
        model = self.config.MODEL
        create_dir(self.results_path)
        if model != 2:
            print('Only MODEL=2 test is currently supported.')
            return

        summaries = []
        for plan in self._build_test_plans():
            summary = self._run_test_plan(plan)
            summaries.append(summary)

        self._write_test_summary(summaries)

        print('\nEnd Testing')
        if len(summaries) == 1 and summaries[0]['name'] == 'all':
            summary = summaries[0]
            if summary['sample_count'] == 0:
                print('full_psnr_ave:N/A full_ssim_ave:N/A full_l1_ave:N/A full_lpips:N/A masked_psnr_ave:N/A masked_ssim_ave:N/A masked_l1_ave:N/A masked_lpips:N/A fid_clean:N/A sample_count:0 mask_schedule_mode:{}'.format(
                    summary.get('mask_schedule_mode', 'unknown'),
                ))
            else:
                print('full_psnr_ave:{} full_ssim_ave:{} full_l1_ave:{} full_lpips:{} masked_psnr_ave:{} masked_ssim_ave:{} masked_l1_ave:{} masked_lpips:{} fid_clean:{} sample_count:{} source_mask_count:{} effective_mask_count:{} mask_schedule_mode:{}'.format(
                    summary['full_image_psnr_ave'],
                    summary['full_image_ssim_ave'],
                    summary['full_image_l1_ave'],
                    summary['full_image_lpips'],
                    summary['masked_psnr_ave'],
                    summary['masked_ssim_ave'],
                    summary['masked_l1_ave'],
                    summary['masked_lpips'],
                    summary.get('fid_clean'),
                    summary['sample_count'],
                    summary.get('source_mask_count', summary.get('mask_count', 0)),
                    summary.get('effective_mask_count', summary['sample_count']),
                    summary.get('mask_schedule_mode', 'unknown'),
                ))
            return summaries

        for summary in summaries:
            if summary['sample_count'] == 0:
                print('[{}] full_psnr_ave:N/A full_ssim_ave:N/A full_l1_ave:N/A full_lpips:N/A masked_psnr_ave:N/A masked_ssim_ave:N/A masked_l1_ave:N/A masked_lpips:N/A fid_clean:N/A source_mask_count:{} effective_mask_count:{} mask_schedule_mode:{}'.format(
                    summary['name'],
                    summary.get('source_mask_count', 0),
                    summary.get('effective_mask_count', 0),
                    summary.get('mask_schedule_mode', 'unknown'),
                ))
                continue

            print('[{}] full_psnr_ave:{} full_ssim_ave:{} full_l1_ave:{} full_lpips:{} masked_psnr_ave:{} masked_ssim_ave:{} masked_l1_ave:{} masked_lpips:{} fid_clean:{} sample_count:{} mask_ratio_mean:{} source_mask_count:{} effective_mask_count:{} mask_schedule_mode:{}'.format(
                summary['name'],
                summary['full_image_psnr_ave'],
                summary['full_image_ssim_ave'],
                summary['full_image_l1_ave'],
                summary['full_image_lpips'],
                summary['masked_psnr_ave'],
                summary['masked_ssim_ave'],
                summary['masked_l1_ave'],
                summary['masked_lpips'],
                summary.get('fid_clean'),
                summary['sample_count'],
                summary['mask_ratio_mean'],
                summary.get('source_mask_count', summary.get('mask_count', 0)),
                summary.get('effective_mask_count', summary['sample_count']),
                summary.get('mask_schedule_mode', 'unknown'),
            ))

        return summaries

    def _get_test_mask_mode(self):
        configured_mode = getattr(self.config, 'TEST_MASK_MODE', None)
        if configured_mode is not None:
            return int(configured_mode)

        if self.config.MASK == 3:
            # Default external test masks should be deterministic for reproducible evaluation.
            return 6

        return int(self.config.MASK)

    def _create_test_dataset(self, mask_flist):
        dataset_config = SimpleNamespace(
            INPUT_SIZE=self.config.INPUT_SIZE,
            MASK=self._get_test_mask_mode(),
        )
        return Dataset(
            dataset_config,
            self.config.TEST_INPAINT_IMAGE_FLIST,
            mask_flist,
            augment=False,
            training=False,
        )

    def _resolve_paths(self, source):
        paths = Dataset.resolve_flist(source)

        if isinstance(paths, np.ndarray):
            paths = paths.tolist()

        if isinstance(paths, str):
            paths = [paths]

        return [str(path) for path in paths]

    def _resolve_mask_paths(self, mask_source):
        return self._resolve_paths(mask_source)

    def _get_test_image_paths(self):
        if self._test_image_paths_cache is None:
            self._test_image_paths_cache = self._resolve_paths(self.config.TEST_INPAINT_IMAGE_FLIST)
        return list(self._test_image_paths_cache)

    def _get_mask_ratio(self, mask_path):
        mask_path = str(mask_path)
        if mask_path in self._mask_ratio_cache:
            return self._mask_ratio_cache[mask_path]

        mask = imread(mask_path)
        ratio = float((mask > 0).mean())
        self._mask_ratio_cache[mask_path] = ratio
        return ratio

    def _filter_mask_paths_by_ratio(self, mask_source, min_ratio=None, max_ratio=None):
        mask_paths = self._resolve_mask_paths(mask_source)
        filtered_paths = []

        for mask_path in mask_paths:
            ratio = self._get_mask_ratio(mask_path)
            if min_ratio is not None and ratio < float(min_ratio):
                continue
            if max_ratio is not None and ratio >= float(max_ratio):
                continue
            filtered_paths.append(mask_path)

        return filtered_paths

    def _align_mask_paths_to_test_images(self, mask_paths):
        source_mask_paths = [str(mask_path) for mask_path in mask_paths]
        test_image_count = len(self._get_test_image_paths())
        source_mask_count = len(source_mask_paths)

        if test_image_count == 0:
            effective_mask_paths = []
            mask_schedule_mode = 'no_test_images'
        elif source_mask_count == 0:
            effective_mask_paths = []
            mask_schedule_mode = 'no_masks'
        elif source_mask_count == test_image_count:
            effective_mask_paths = list(source_mask_paths)
            mask_schedule_mode = 'one_to_one'
        elif source_mask_count > test_image_count:
            effective_mask_paths = list(source_mask_paths[:test_image_count])
            mask_schedule_mode = 'truncated'
        else:
            repeats, remainder = divmod(test_image_count, source_mask_count)
            effective_mask_paths = (source_mask_paths * repeats) + source_mask_paths[:remainder]
            mask_schedule_mode = 'cycled'

        unique_effective_mask_count = len(set(effective_mask_paths))
        effective_mask_count = len(effective_mask_paths)

        return {
            'source_mask_paths': source_mask_paths,
            'effective_mask_paths': effective_mask_paths,
            'source_mask_count': source_mask_count,
            'effective_mask_count': effective_mask_count,
            'unique_effective_mask_count': unique_effective_mask_count,
            'effective_mask_repeats': max(0, effective_mask_count - unique_effective_mask_count),
            'mask_schedule_mode': mask_schedule_mode,
            'test_image_count': test_image_count,
        }

    def _sanitize_bucket_name(self, name):
        sanitized = str(name)
        sanitized = sanitized.replace('%', 'pct')
        sanitized = sanitized.replace(' ', '')
        sanitized = sanitized.replace('-', '_')
        sanitized = sanitized.replace('/', '_')
        sanitized = sanitized.replace('.', 'p')
        return sanitized

    def _get_active_bucket_configs(self):
        if self._eval_bucket_configs_override is not None:
            return self._eval_bucket_configs_override
        return getattr(self.config, 'TEST_MASK_BUCKETS', None)

    def _build_test_dataset_plan(self, name, mask_source, ratio_range, results_root):
        min_ratio, max_ratio = (ratio_range or [None, None])
        mask_paths = self._filter_mask_paths_by_ratio(mask_source, min_ratio=min_ratio, max_ratio=max_ratio)
        alignment = self._align_mask_paths_to_test_images(mask_paths)
        dataset = None
        if alignment['effective_mask_count'] > 0:
            dataset = self._create_test_dataset(alignment['effective_mask_paths'])

        return {
            'name': name,
            'dataset': dataset,
            'mask_paths': alignment['source_mask_paths'],
            'effective_mask_paths': alignment['effective_mask_paths'],
            'mask_source': mask_source,
            'ratio_range': ratio_range,
            'results_root': results_root,
            'source_mask_count': alignment['source_mask_count'],
            'effective_mask_count': alignment['effective_mask_count'],
            'unique_effective_mask_count': alignment['unique_effective_mask_count'],
            'effective_mask_repeats': alignment['effective_mask_repeats'],
            'mask_schedule_mode': alignment['mask_schedule_mode'],
            'test_image_count': alignment['test_image_count'],
        }

    def _build_test_plans(self):
        bucket_configs = self._get_active_bucket_configs()
        if not bucket_configs:
            self._ensure_test_dataset()
            default_meta = self._default_test_dataset_meta or {}
            return [{
                'name': 'all',
                'dataset': self.test_dataset,
                'mask_paths': list(default_meta.get('mask_paths', [])),
                'effective_mask_paths': list(default_meta.get('effective_mask_paths', [])),
                'mask_source': default_meta.get('mask_source', getattr(self.config, 'TEST_MASK_FLIST', None)),
                'ratio_range': default_meta.get('ratio_range'),
                'results_root': self.results_path,
                'source_mask_count': default_meta.get('source_mask_count', 0),
                'effective_mask_count': default_meta.get('effective_mask_count', 0),
                'unique_effective_mask_count': default_meta.get('unique_effective_mask_count', 0),
                'effective_mask_repeats': default_meta.get('effective_mask_repeats', 0),
                'mask_schedule_mode': default_meta.get('mask_schedule_mode', 'unknown'),
                'test_image_count': default_meta.get('test_image_count', len(self._get_test_image_paths())),
            }]

        plans = []
        for index, bucket in enumerate(bucket_configs):
            if not isinstance(bucket, dict):
                raise ValueError('TEST_MASK_BUCKETS entries must be dicts')

            name = bucket.get('name', f'bucket_{index}')
            min_ratio = bucket.get('min_ratio')
            max_ratio = bucket.get('max_ratio')
            mask_source = bucket.get('mask_flist', getattr(self.config, 'TEST_MASK_FLIST', None))
            plans.append(self._build_test_dataset_plan(
                name=name,
                mask_source=mask_source,
                ratio_range=[min_ratio, max_ratio],
                results_root=os.path.join(self.results_path, 'bucketed', self._sanitize_bucket_name(name)),
            ))

        return plans

    def _save_test_visualizations(self, dataset, index, images, masks, outputs_img, outputs_merged, results_root):
        images_joint = stitch_images(
            self.postprocess(images),
            self.postprocess(images * (1 - masks)),
            self.postprocess(outputs_img),
            self.postprocess(outputs_merged),
            img_per_row=1
        )

        path_masked = os.path.join(results_root, self.model_name, 'masked_lama')
        path_result = os.path.join(results_root, self.model_name, 'result_lama')
        path_joint = os.path.join(results_root, self.model_name, 'joint_lama')
        path_gt = os.path.join(results_root, self.model_name, 'gt_lama')

        name = dataset.load_name(index - 1)[:-4] + '.png'

        create_dir(path_masked)
        create_dir(path_result)
        create_dir(path_joint)
        create_dir(path_gt)

        masked_images = self.postprocess(images * (1 - masks) + masks)[0]
        images_result = self.postprocess(outputs_merged)[0]
        gt_images = self.postprocess(images)[0]

        print(os.path.join(path_joint, name[:-4] + '.png'))

        images_joint.save(os.path.join(path_joint, name[:-4] + '.png'))
        imsave(masked_images, os.path.join(path_masked, name))
        imsave(images_result, os.path.join(path_result, name))
        imsave(gt_images, os.path.join(path_gt, name))

        print(name + ' complete!')

    def _compute_fid(self, gt_dir, gen_dir):
        if clean_fid is None:
            print('Skipping FID because cleanfid is not installed.')
            return None

        gt_images = [name for name in os.listdir(gt_dir) if os.path.isfile(os.path.join(gt_dir, name))]
        gen_images = [name for name in os.listdir(gen_dir) if os.path.isfile(os.path.join(gen_dir, name))]
        if len(gt_images) < 2 or len(gen_images) < 2:
            print('Skipping FID because there are fewer than 2 images in gt/gen directories.')
            return None

        return float(clean_fid.compute_fid(gt_dir, gen_dir, mode=self.fid_mode))

    def _tensor_to_uint8_image(self, tensor):
        array = tensor.detach().clamp(0, 1) * 255.0
        array = array.permute(0, 2, 3, 1)
        return array.cpu().numpy().astype(np.uint8)[0]

    def _mask_to_bool(self, masks):
        return masks.detach().cpu().numpy()[0, 0] > 0.5

    def _mask_bbox(self, mask_bool, height, width, pad=8, min_size=64):
        ys, xs = np.where(mask_bool)
        if len(xs) == 0 or len(ys) == 0:
            return 0, 0, width, height

        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(width, int(xs.max()) + 1 + pad)
        y1 = min(height, int(ys.max()) + 1 + pad)

        if (x1 - x0) < min_size:
            need = min_size - (x1 - x0)
            left = need // 2
            right = need - left
            x0 = max(0, x0 - left)
            x1 = min(width, x1 + right)
        if (y1 - y0) < min_size:
            need = min_size - (y1 - y0)
            top = need // 2
            bottom = need - top
            y0 = max(0, y0 - top)
            y1 = min(height, y1 + bottom)

        return x0, y0, x1, y1

    def _lpips_distance(self, pred, gt):
        pred_input = self.transf(pred[0].detach().cpu()).unsqueeze(0)
        gt_input = self.transf(gt[0].detach().cpu()).unsqueeze(0)
        if self.config.DEVICE.type == 'cuda':
            pred_input = pred_input.to(self.config.DEVICE)
            gt_input = gt_input.to(self.config.DEVICE)
        return self.loss_fn_vgg(pred_input, gt_input).item()

    def metric(self, gt, pre):
        gt_np = self._tensor_to_uint8_image(gt)
        pre_np = self._tensor_to_uint8_image(pre)

        psnr = min(100, compare_psnr(gt_np, pre_np))
        ssim = compare_ssim(gt_np, pre_np, channel_axis=-1, data_range=255)

        return psnr, ssim

    def masked_metrics(self, gt, pre, masks):
        gt_np = self._tensor_to_uint8_image(gt)
        pre_np = self._tensor_to_uint8_image(pre)
        mask_bool = self._mask_to_bool(masks)

        if not np.any(mask_bool):
            psnr, ssim = self.metric(gt, pre)
            l1_loss = torch.nn.functional.l1_loss(pre, gt, reduction='mean').item()
            lpips_score = self._lpips_distance(pre, gt)
            return psnr, ssim, l1_loss, lpips_score

        masked_diff = (pre_np.astype(np.float32) - gt_np.astype(np.float32))[mask_bool]
        masked_mse = float(np.mean(masked_diff ** 2))
        masked_psnr = 100.0 if masked_mse <= 0 else min(100.0, float(compare_psnr(gt_np[mask_bool], pre_np[mask_bool], data_range=255)))

        _, ssim_map = compare_ssim(gt_np, pre_np, channel_axis=-1, data_range=255, full=True)
        if ssim_map.ndim == 3:
            ssim_map = ssim_map.mean(axis=2)
        masked_ssim = float(np.mean(ssim_map[mask_bool]))

        mask_tensor = masks.expand_as(pre)
        denom = float(mask_tensor.sum().item())
        masked_l1 = 0.0 if denom <= 0 else float(torch.abs(pre - gt).mul(mask_tensor).sum().item() / denom)

        height, width = mask_bool.shape
        x0, y0, x1, y1 = self._mask_bbox(mask_bool, height, width)
        pred_crop = pre[:, :, y0:y1, x0:x1]
        gt_crop = gt[:, :, y0:y1, x0:x1]
        mask_crop = masks[:, :, y0:y1, x0:x1]
        pred_crop = pred_crop * mask_crop
        gt_crop = gt_crop * mask_crop
        masked_lpips = self._lpips_distance(pred_crop, gt_crop)

        return masked_psnr, masked_ssim, masked_l1, masked_lpips

    def _run_test_plan(self, plan):
        dataset = plan['dataset']
        if dataset is None:
            print('[{}] No masks matched the configured ratio range.'.format(plan['name']))
            return {
                'name': plan['name'],
                'mask_source': plan['mask_source'],
                'mask_count': plan.get('source_mask_count', 0),
                'sample_count': 0,
                'ratio_range': plan['ratio_range'],
                'results_root': plan['results_root'],
                'source_mask_count': plan.get('source_mask_count', 0),
                'effective_mask_count': plan.get('effective_mask_count', 0),
                'unique_effective_mask_count': plan.get('unique_effective_mask_count', 0),
                'effective_mask_repeats': plan.get('effective_mask_repeats', 0),
                'mask_schedule_mode': plan.get('mask_schedule_mode'),
                'test_image_count': plan.get('test_image_count', 0),
            }

        create_dir(plan['results_root'])
        test_loader = DataLoader(dataset=dataset, batch_size=1)

        psnr_list = []
        ssim_list = []
        l1_list = []
        lpips_list = []
        masked_psnr_list = []
        masked_ssim_list = []
        masked_l1_list = []
        masked_lpips_list = []
        mask_ratio_list = []

        print('\nTesting bucket: {}'.format(plan['name']))
        if plan['mask_schedule_mode'] != 'one_to_one':
            print(
                '[{}] mask schedule mode={} source_mask_count={} test_image_count={} effective_mask_count={} unique_effective_mask_count={}'.format(
                    plan['name'],
                    plan['mask_schedule_mode'],
                    plan['source_mask_count'],
                    plan['test_image_count'],
                    plan['effective_mask_count'],
                    plan['unique_effective_mask_count'],
                )
            )
        index = 0
        for items in test_loader:
            images, masks = self.cuda(*items)
            index += 1

            inputs = (images * (1 - masks))
            with torch.no_grad():
                outputs_img = self.inpaint_model(images, masks)

            outputs_merged = (outputs_img * masks) + (images * (1 - masks))

            if self._eval_log_per_sample:
                print('outpus_size', outputs_merged.size())
                print('images', images.size())

            psnr, ssim = self.metric(images, outputs_merged)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            pl = self._lpips_distance(outputs_merged, images)
            lpips_list.append(pl)

            l1_loss = torch.nn.functional.l1_loss(outputs_merged, images, reduction='mean').item()
            l1_list.append(l1_loss)
            masked_psnr, masked_ssim, masked_l1, masked_lpips = self.masked_metrics(images, outputs_merged, masks)
            masked_psnr_list.append(masked_psnr)
            masked_ssim_list.append(masked_ssim)
            masked_l1_list.append(masked_l1)
            masked_lpips_list.append(masked_lpips)

            mask_ratio = float(masks.mean().item())
            mask_ratio_list.append(mask_ratio)

            if self._eval_log_per_sample:
                print('[{}] full_psnr:{}/{} full_ssim:{}/{} full_l1:{}/{} full_lpips:{}/{} masked_psnr:{}/{} masked_ssim:{}/{} masked_l1:{}/{} masked_lpips:{}/{} mask_ratio:{}/{}  {}'.format(
                    plan['name'],
                    psnr,
                    np.average(psnr_list),
                    ssim,
                    np.average(ssim_list),
                    l1_loss,
                    np.average(l1_list),
                    pl,
                    np.average(lpips_list),
                    masked_psnr,
                    np.average(masked_psnr_list),
                    masked_ssim,
                    np.average(masked_ssim_list),
                    masked_l1,
                    np.average(masked_l1_list),
                    masked_lpips,
                    np.average(masked_lpips_list),
                    mask_ratio,
                    np.average(mask_ratio_list),
                    len(ssim_list),
                ))

            if self._eval_save_visualizations:
                self._save_test_visualizations(
                    dataset,
                    index,
                    images,
                    masks,
                    outputs_img,
                    outputs_merged,
                    plan['results_root'],
                )

        summary = {
            'name': plan['name'],
            'mask_source': plan['mask_source'],
            'mask_count': len(plan['mask_paths']),
            'sample_count': len(psnr_list),
            'ratio_range': plan['ratio_range'],
            'source_mask_count': plan['source_mask_count'],
            'effective_mask_count': plan['effective_mask_count'],
            'unique_effective_mask_count': plan['unique_effective_mask_count'],
            'effective_mask_repeats': plan['effective_mask_repeats'],
            'mask_schedule_mode': plan['mask_schedule_mode'],
            'test_image_count': plan['test_image_count'],
            'mask_ratio_mean': float(np.average(mask_ratio_list)),
            'mask_ratio_min': float(np.min(mask_ratio_list)),
            'mask_ratio_max': float(np.max(mask_ratio_list)),
            'full_image_psnr_ave': float(np.average(psnr_list)),
            'full_image_ssim_ave': float(np.average(ssim_list)),
            'full_image_l1_ave': float(np.average(l1_list)),
            'full_image_lpips': float(np.average(lpips_list)),
            'masked_psnr_ave': float(np.average(masked_psnr_list)),
            'masked_ssim_ave': float(np.average(masked_ssim_list)),
            'masked_l1_ave': float(np.average(masked_l1_list)),
            'masked_lpips': float(np.average(masked_lpips_list)),
            'edge_psnr_ave': float(np.average(psnr_list)),
            'edge_ssim_ave': float(np.average(ssim_list)),
            'l1_ave': float(np.average(l1_list)),
            'lpips': float(np.average(lpips_list)),
            'results_root': plan['results_root'],
        }

        fid_score = None
        if self._eval_compute_fid and self._eval_save_visualizations:
            gt_dir = os.path.join(plan['results_root'], self.model_name, 'gt_lama')
            gen_dir = os.path.join(plan['results_root'], self.model_name, 'result_lama')
            fid_score = self._compute_fid(gt_dir, gen_dir)
        summary['fid_clean'] = fid_score

        return summary

    def _write_test_summary(self, summaries):
        payload = {
            'eval_image_flist': str(self.config.TEST_INPAINT_IMAGE_FLIST),
            'eval_mask_flist': str(self.config.TEST_MASK_FLIST),
            'test_image_flist': str(self.config.TEST_INPAINT_IMAGE_FLIST),
            'test_mask_flist': str(self.config.TEST_MASK_FLIST),
            'test_image_count': len(self._get_test_image_paths()),
            'test_mask_mode': self._get_test_mask_mode(),
            'summaries': summaries,
        }

        os.makedirs(os.path.dirname(self.test_summary_path), exist_ok=True)
        with open(self.test_summary_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print('Test summary saved to: {}'.format(self.test_summary_path))



    def log(self, logs):
        with open(self.log_file, 'a') as f:
            print('load the generator:')
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))
            print('finish load')

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
