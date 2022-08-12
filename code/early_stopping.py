import numpy as np
import torch
import copy

class EarlyStopping:
    def __init__(self, patience=6, mode="max", max_epoch=1e6, min_epoch=0, at_last_score=None):
        self.patience = patience
        self.mode = mode
        self.max_epoch = max_epoch
        self.min_epoch = min_epoch
        self.at_last_score = at_last_score if at_last_score is not None else -np.Inf 
        self.epoch = 0
        self.early_stop = False
        self.best_model = None
        self.best_epoch = 0
        self.model_path = None
        self.best_score = -np.Inf if self.mode == "max" else np.Inf

    def __call__(self, epoch_score, model=None, model_path=None):
        self.model_path = model_path
        self.epoch += 1

        score = -epoch_score if self.mode == "min" else epoch_score
        
        if score <= self.best_score: 
            counter = self.epoch - self.best_epoch
            print('EarlyStopping counter: {} out of {}'.format(counter, self.patience))
            if (counter >= self.patience) and (self.best_score > self.at_last_score) and (self.epoch >= self.min_epoch):
                self.early_stop = True 
                self._save_checkpoint()
        else:                    
            self.best_score = score 
            self.best_epoch = self.epoch
            self.best_model = copy.deepcopy(model).cpu()
        
        if self.max_epoch <= self.epoch:
            self.early_stop = True 
            self._save_checkpoint()

    def _save_checkpoint(self):
        if self.model_path is not None and self.best_model is not None:
            torch.save(self.best_model.state_dict(), self.model_path.replace('model_es','model_es_epoch_'+str(self.epoch)))
            print('model saved at: ',self.model_path.replace('model_es','model_es_epoch_'+str(self.epoch)))
            
