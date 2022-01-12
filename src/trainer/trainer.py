import copy
import numpy as np
import torch

from ..base import BaseTrainer
import src.data_loader as module_data
import src.model.model as module_arch
import src.model.loss as module_loss
import src.model.metric as module_metric
import src.utils.custom_lr_scheduler as custom_lr_scheduler
from ..utils import inf_loop, MetricTracker, compute_sim_mat, \
                        prepare_device, Tokenizer, read_json



class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, config):
        logger = config.get_logger('train')
        self.config = config

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(config['n_gpu'])
        self.device = device

        self.train_data_loader = getattr(module_data, config['data_loader']['type'])(
                                   config['data_loader']['train_path'], shuffle=True, 
                                   **config['data_loader']['args'])
        test_loader_args = copy.deepcopy(config['data_loader']['args'])
        test_loader_args['video_setting']['sample'] = 'uniform'
        test_loader_args['text_setting']['sample'] = 'all'
        self.val_data_loader = getattr(module_data, config['data_loader']['type'])(
                                        config['data_loader']['val_path'], 
                                        shuffle=False, drop_last=False, 
                                        **test_loader_args)
        self.test_data_loader = getattr(module_data, config['data_loader']['type'])(
                                        config['data_loader']['test_path'], 
                                        shuffle=False, drop_last=False, 
                                        **test_loader_args)
        self.do_validation = self.val_data_loader is not None

        self.tokenizer = Tokenizer(config['arch']['args']['base_setting']['type'])
        model = getattr(module_arch, config['arch']['type'])(
                            device=device, **config['arch']['args'])
        logger.info(model)
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        criterion = getattr(module_loss, config['loss']['type'])(**config['loss']['args'])
        metric_ftns = [getattr(module_metric, met) for met in config['metrics']]

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        try:
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
        except:
            lr_scheduler = getattr(custom_lr_scheduler, config['lr_scheduler']['type'])(
                                        optimizer, **config['lr_scheduler']['args'])
        self.lr_scheduler = lr_scheduler
        # avoid 0 learning rate for first epoch
        if config['lr_scheduler']['type'].startswith('LinearWarmup'):
            self.lr_scheduler.step()

        self.log_step = config.config.get('log_step', 10)
        self.train_metrics = MetricTracker('loss')
        self.len_epoch = len(self.train_data_loader)

        # amp
        self.use_amp = self.config.config.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # five query evaluation sample
        if self.do_validation:
            self.five_query_sample = read_json(self.config['val_eval_sample'])['5']

        super().__init__(model, criterion, metric_ftns, optimizer, config)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (frames, captions) in enumerate(self.train_data_loader):
            frames = frames.to(self.device)
            captions = self.tokenizer.tokenize(captions)
            if isinstance(captions, torch.Tensor):
                captions = captions.to(self.device)
            else:
                captions = {k: v.to(self.device) for k, v in captions.items()}

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                video_features, text_features = self.model(frames, captions)
                sim_mat = compute_sim_mat(text_features, video_features)
                loss = self.criterion(sim_mat)
                            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.train_metrics.update('loss', loss.item(), frames.size(0))
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

        log = self.train_metrics.result()
        log = {'train/'+k : v for k, v in log.items()}

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val/'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        self.model.eval()
        all_video_features, all_text_features = [], []
        with torch.no_grad():
            for batch_idx, (frames, captions) in enumerate(self.val_data_loader):
                frames = frames.to(self.device)
                captions = self.tokenizer.tokenize(captions)
                if isinstance(captions, torch.Tensor):
                    captions = captions.to(self.device)
                else:
                    captions = {k: v.to(self.device) for k, v in captions.items()}
        
                if self.config['n_gpu'] > 1:
                    video_features, text_features = self.model.module.base(frames, captions)
                else:
                    video_features, text_features = self.model.base(frames, captions)
                all_video_features.append(video_features)
                all_text_features.append(text_features)

            all_video_features = torch.cat(all_video_features, dim=0)
            all_text_features = torch.cat(all_text_features, dim=0)
            
            valid_metrics = {}
            for idx in self.five_query_sample:
                idx = torch.tensor(idx, device=self.device).unsqueeze(2).expand(-1, -1, all_text_features.shape[2])
                query_text_features = torch.gather(all_text_features, 1, idx)
                if self.config['n_gpu'] > 1:
                    query_text_features = self.model.module.text_head(query_text_features)
                else:
                    query_text_features = self.model.text_head(query_text_features)

                sim_mat = compute_sim_mat(query_text_features, all_video_features)
                metrics = module_metric.retrieval_metric(sim_mat)
                for key, item in metrics.items():
                    valid_metrics.setdefault(key, []).append(item)

        valid_metrics = {key: np.mean(item) for key, item in valid_metrics.items()}
        return valid_metrics

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
