# trainer.py
import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model, device, optimizer, criterion, scaler, status_bar, model_type, epochs):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.status_bar = status_bar
        self.model_type = model_type
        self.epochs = epochs
        self.loss_history = []
        self.val_loss_history = []

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        
        # Initialize the progress bar outside the loop
        progress_text = "Training in progress"
        the_bar = self.status_bar.progress(0, text=progress_text)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device, dtype=torch.float32)
            target = target.to(self.device, dtype=torch.long)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.device.type == 'cuda'):
                output = self.model(data)
                loss = self.criterion(output, target)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            
            # Update the progress bar within the loop
            if batch_idx % 5 == 0:
                progress_percent = (batch_idx + 1) / len(train_loader) * 100
                the_bar.progress(int(progress_percent), text=progress_text)
                torch.cuda.empty_cache()
        
        epoch_loss = running_loss / len(train_loader)
        self.loss_history.append(epoch_loss)
        return epoch_loss

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device, dtype=torch.float32)
                target = target.to(self.device, dtype=torch.long)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.device.type == 'cuda'):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                val_loss += loss.item()

        return val_loss / len(val_loader)