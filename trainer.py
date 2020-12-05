from torch import nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from tqdm import tqdm

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_loader = train_dataset
        self.test_loader = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model 
        if config.stage == 'stage1':
            criterion = nn.BCEWithLogitsLoss()
        elif config.stage == 'stage2':
            criterion = nn.MSELoss()
        else:
            criterion1 = nn.MSELoss()
            criterion2 = nn.BCEWithLogitsLoss()
            
        optimizer = optim.Adam(lr = config.learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, config.max_epochs, eta_min = config.learning_rate/10)
        scaler = GradScaler()

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            loader = self.train_loader if is_train else self.test_loader

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(is_train):
                        if config.stage == 'stage3':
                            x1, x2 = model(x)
                        else:
                            logits = model(x)

                        if config.stage == 'stage3':
                            loss = 0.5*(criterion1(x1, x) + criterion2(x2, y))
                        else:
                            loss = criterion(logits, y)

                        losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}")
    
            scheduler.step()

            if is_train:
                return float(np.mean(losses))

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        for epoch in range(config.max_epochs):

            train_loss = run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            wandb.log({'epoch_valid_loss': test_loss, 'epoch_train_loss': train_loss, 'epoch': epoch + 1})

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_loader is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                print(f'Saving at epoch {epoch + 1}')
                self.save_checkpoint()