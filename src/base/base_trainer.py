import torch
from abc import abstractmethod
from numpy import inf
from datetime import datetime

from src.utils import write_json, write_pkl



class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """

        if self.do_validation:
            val_log = self._valid_epoch(0)
            log = {'epoch': 0}
            log.update(**{'val/'+k : v for k, v in val_log.items()})
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

        for epoch in range(self.start_epoch, self.epochs + 1):
            start_time = datetime.now()
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            log['epoch_time'] = abs(datetime.now() - start_time).total_seconds()
            self.log = log

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))
            
            self.logger.info('    {:15s}: {}'.format('time used:', datetime.now() - start_time))

            if self.save_period == 0:
                self._save_checkpoint(epoch, save_best=False, only_latest=True)
            elif epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=False, only_latest=True)

            if epoch == self.epochs:
                self._save_checkpoint(epoch, save_best=False, only_latest=True)

    def _save_checkpoint(self, epoch, save_best=False, only_latest=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        if hasattr(self.model, 'module'):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        if hasattr(self, 'scaler'):
            state['scaler'] = self.scaler.state_dict()

        if only_latest:
            filename = str(self.checkpoint_dir / 'checkpoint-latest.pth'.format(epoch))
        else:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        write_json(self.log, str(self.checkpoint_dir / 'checkpoint.json'))

        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")
            write_json(self.log, str(self.checkpoint_dir / 'best.json'))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

