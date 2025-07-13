import torch
from models.contrastive import LS_HATCL_LOSS, HATCL_LOSS
from tqdm import tqdm
from src.models.attention_model import *
from pytorch_lightning.loggers import WandbLogger
import wandb
from models import TSEncoder

class CaTT:
    '''The CaTT model'''
    
    def __init__(
        self,
        args,
        config,
        device='cuda',
    ):
        '''
          Initialize a DynaCL model.

        '''
        
        self.args = args
        self.config = config
        super().__init__()
        
        self.device = device
        self.net = FeatureProjector(input_size=args['feature_dim'], output_size=args['out_features']).to(self.device)

        # self.net = TSEncoder(input_dims=args['feature_dim'], output_dims=args['out_features']).to(self.device)
        self.n_iters = 0
        
        

       
    
    def fit(self, train_dataset, ds_name, verbose=False):
        ''' Training the CaTT model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        
        train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size= self.args['batch_size'],
                shuffle = True,
                num_workers=self.config.NUM_WORKERS,
                drop_last = True,
            )
        
        # Wandb setup
        if self.config.WANDB:    
            proj_name = 'Dynamic_CL' + ds_name + str(self.config.SEED)
            run_name = 'DynaCL'

            wandb_logger = WandbLogger(project=proj_name)
            
            # Initialize Wandb
            wandb.init(project=proj_name, name=run_name)
            wandb.watch(self.net, log='all', log_freq=100)

            # Update Wandb config
        
            wandb.config.update(self.args)
            wandb.config.update({
                'Algorithm': f'{run_name}',
                'Dataset': f'{ds_name}',
                'Train_DS_size': len(train_dataset),
                'Batch_Size': self.args["batch_size"],
                'Epochs': self.args["epochs"],
                'Patience': self.config.PATIENCE,
                'Seed': self.config.SEED

            })
            wandb.run.name = run_name
            wandb.run.save()
        
        # Define loss function and optimizer

        if self.args['loss'] == 'HATCL_LOSS':
            cl_loss = HATCL_LOSS(temperature=self.args['temperature'])

        elif self.args['loss'] == 'LS_HATCL_LOSS':
            cl_loss = LS_HATCL_LOSS(temperature=self.args['temperature'])

        else:
            raise ValueError(f"Unsupported loss function: {self.args['loss']}")

        self.args['lr'] = float(self.args['lr'])

        if self.args['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args['lr'])  # Example optimizer

        elif self.args['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args['lr'])  # Example optimizer

        elif self.args['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.args['lr'])  # Example optimizer

        elif self.args['optimizer'] == 'Adadelta':
            optimizer = torch.optim.Adadelta(self.net.parameters(), lr=self.args['lr'])  # Example optimizer

        else:
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args['lr'])  # Example optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=self.config.PATIENCE)
        
        
        n_iters = self.args['iterations']
        pbar = tqdm(total=n_iters, desc="Training")
        epoch = 0

        while True:

            # Training phase
            self.net.train()  # Set the model to training mode
            train_running_loss = 0.0
            n_epoch_iters = 0

            for x, _ in train_loader:

                interrupted = False
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                x = x.to(self.device)
               
                out = self.net(x)

                features = out.reshape(-1, out.size(-1))
                
                loss = cl_loss(features)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
                # Update training statistics
                n_epoch_iters += 1
                self.n_iters += 1
                pbar.update(1)

                train_running_loss += loss.item()

            if interrupted:
                break
            train_running_loss /= n_epoch_iters 

            
            epoch += 1
    
            if verbose:
                print(f"Epoch {epoch}, Train Loss: {train_running_loss:.4f}")

            # Log training loss to Wandb
            if self.config.WANDB:
                wandb.log({'Train Loss': train_running_loss, 'Epoch': epoch})
        try:   
            return train_running_loss
        except:
            return 0
    
    def encode(self, x):
        self.net.eval()
        out = self.net(x.to(self.device))

        return out


    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
