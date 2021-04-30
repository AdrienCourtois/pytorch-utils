from inspect import isclass
import re
import os
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision

import metrics 


class Trainer:
    def __init__(self, 
        model, 
        criterion=None, metrics=None, scheduler=None,
        lr=1e-3, weight_decay=0, optimizer=None, 
        warmup=0,
        use_cuda=True, 
        writer=None, session_name=None, show_imgs=False,
        verbose=True, resume=False
    ):
        self.model = model

        self.use_cuda = use_cuda
        self.verbose = verbose
        self.resume = resume

        self.session_name = session_name

        self.writer = SummaryWriter(f"runs/{self.session_name}") if writer is None else writer
        self.show_imgs = show_imgs

        if self.show_imgs:
            self.show_train_idx = np.random.rand(4)
            self.show_test_idx = np.random.rand(4)

        self.criterion = criterion
        self.metrics = metrics

        self._lr = lr
        self.lr_max = lr
        self._weight_decay = weight_decay
        self.optimizer = torch.optim.Adam if optimizer is None else optimizer

        self.warmup = warmup
        self.scheduler = scheduler
        
        if not self.resume:
            self.save_model("best")
        else:
            self.model = self.load_best_model()


    @property
    def session_name(self):
        return self._session_name


    @session_name.setter
    def session_name(self, v):
        tmp_name = "" if v is None else v+"_"
        regex = re.compile(f"model_(.*?)_{tmp_name}[0-9]+.pth")

        other_models = os.listdir('./')
        other_models = filter(lambda x: regex.search(x), other_models)
        other_idx = list(map(lambda x: int(x[:-4].split('_')[-1]), other_models))

        if len(other_idx) == 0:
            if self.resume:
                raise Exception("The last checkpoint hasn't been found")

            self._session_name = tmp_name+str(0)
        else:
            if not self.resume:
                self._session_name = tmp_name+str(max(other_idx)+1)
            else:
                self._session_name = tmp_name+str(max(other_idx))


    @property
    def nb_params(self):
        return sum(np.prod(p.size()) for p in self.model.parameters() if p.requires_grad)
    

    @torch.no_grad()
    def save_imgs(self, loader, alphas, name, epoch):
        dataset = loader.dataset

        x = torch.cat([
            dataset[int(np.round(len(dataset)*x))][0][None]
            for x in alphas
        ], 0)
        y = torch.cat([
            dataset[int(np.round(len(dataset)*x))][1][None]
            for x in alphas
        ], 0)

        if self.use_cuda:
            x, y = map(lambda x: x.cuda(), [x,y])
        
        y_hat = self.model(x)

        self.writer.add_image(f'{name} input', torchvision.utils.make_grid(x.cpu()), epoch)
        self.writer.add_image(f'{name} output', torchvision.utils.make_grid(y_hat.cpu()), epoch)
        self.writer.add_image(f'{name} ground truth', torchvision.utils.make_grid(y.cpu()), epoch)


    @property
    def scheduler(self):
        return self._scheduler
    
    @scheduler.setter 
    def scheduler(self, v):
        if isclass(v):
            self._scheduler = v(self.optimizer)

        else:
            self._scheduler = v


    @property
    def metrics(self):
        return self._metric_names

    @metrics.setter
    def metrics(self, v):
        self._metric_names = [
            x if isinstance(x, str) else x.__name__ 
            for x in v
        ]

        self._metrics_fn = []
        for x in v:
            if not isinstance(x, str):
                self._metrics_fn.append(x)
            else:
                n = f"compute_batch_{x.lower()}"

                if hasattr(metrics, n):
                    self._metrics_fn.append(getattr(metrics, n))
                else:
                    raise Exception(f'Not implemented metric: {x}')


    @property
    def lr(self):
        return self._lr
    
    @lr.setter
    def lr(self, v):
        self._lr = v

        if self.optimizer is not None:
            for g in self.optimizer.param_groups:
                g['lr'] = v


    @property
    def weight_decay(self):
        return self._weight_decay
    
    @weight_decay.setter
    def weight_decay(self, v):
        self._weight_decay = v

        if self.optimizer is not None:
            for g in self.optimizer.param_groups:
                g['weight_decay'] = v


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, v):
        if isclass(v):
            self._optimizer = v(
                self.model.parameters(), 
                lr=self.lr,
                weight_decay=self.weight_decay
            )

        else:
            self._optimizer = v


    def nan_check(self, loss, epoch, idx):
        if loss != loss:
            raise Exception(f'nan loss at epoch {epoch} iteration {idx}')
        
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None and torch.any(p.grad != p.grad):
                raise Exception(f'nan grad at epoch {epoch} iteration {idx}')
            if p is not None and torch.any(p != p):
                raise Exception(f'nan parameter at epoch {epoch} iteration {idx}')


    def train_iter(self, train_loader, epoch):
        if self.criterion is None:
            raise Exception('Please define a training criterion first')
        
        losses = []
        metrics = {
            x: 0 for x in self.metrics
        }

        for idx, (x, y) in enumerate(train_loader):
            if self.use_cuda:
                x, y = map(lambda x: x.cuda(), [x, y])
            
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)

            self.nan_check(loss, epoch, idx)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            losses.append(loss.item())

            # Compute metrics
            for metric_name, metric_fn in zip(self.metrics, self._metrics_fn):
                metrics[metric_name] += metric_fn(y_hat, y).sum().item()

        # Normalize metrics
        for metric_name in self.metrics:
            metrics[metric_name] /= len(train_loader.dataset)
        
        return np.mean(losses), metrics
    
    @torch.no_grad()
    def test_iter(self, test_loader, epoch):
        losses = []
        metrics = {
            x: 0 for x in self.metrics
        }

        for x, y in test_loader:
            if self.use_cuda:
                x, y = map(lambda x: x.cuda(), [x, y])
            
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)

            losses.append(loss.item())
            
            # Compute metrics
            for metric_name, metric_fn in zip(self.metrics, self._metrics_fn):
                metrics[metric_name] += metric_fn(y_hat, y).sum().item()
        
        # Normalize metrics
        for metric_name in self.metrics:
            metrics[metric_name] /= len(test_loader.dataset)
        
        return np.mean(losses), metrics            
    
    def train(self, epochs, train_loader, test_loader=None):
        best_loss = np.inf

        generator = range(1, epochs+1)
        generator = generator if not self.verbose else tqdm(generator)

        for epoch in generator:
            # Warmup
            if self.warmup > 0:
                warmup_epochs = int(np.round(self.warmup * epochs))

                if epoch <= warmup_epochs:
                    self.lr = self.lr_max * epoch / warmup_epochs

            # Training
            self.model.train()
            train_loss, train_metrics = self.train_iter(train_loader, epoch)

            # Scheduler
            if self.scheduler is not None:
                self.scheduler.step(train_loss)

            # Write in TensorBoard
            self.writer.add_scalar('Training loss', train_loss, epoch)

            for metric_name in self.metrics:
                self.writer.add_scalar(f'Training {metric_name}', train_metrics[metric_name], epoch)


            # Evaluation
            if test_loader is not None:
                self.model.eval()
                test_loss, test_metrics = self.test_iter(test_loader, epoch)

                # Write in TensorBoard
                self.writer.add_scalar('Testing loss', test_loss, epoch)

                for metric_name in self.metrics:
                    self.writer.add_scalar(f'Testing {metric_name}', test_metrics[metric_name], epoch)
                
                # Saving model if there's a test set
                if best_loss > test_loss:
                    best_loss = test_loss
                    self.save_model("best")
            
            else:
                # Saving model if there's no test set
                if best_loss > train_loss:
                    best_loss = train_loss
                    self.save_model("best")
            
            # Plot images in TensorBoard
            if self.show_imgs:
                self.save_imgs(train_loader, self.show_train_idx, "Training", epoch)
                self.save_imgs(test_loader, self.show_test_idx, "Testing", epoch)
    

    def save_model(self, name):
        torch.save(self.model.state_dict(), f"./model_{name}_{self.session_name}.pth")

    def load_best_model(self):
        best_model = deepcopy(self.model)
        best_model.load_state_dict(torch.load(f"./model_best_{self.session_name}.pth"))

        return best_model