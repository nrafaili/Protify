import os
import yaml
import wandb
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from types import SimpleNamespace
from utils import print_message
from probes.trainers import TrainerArguments

@dataclass
class WandbHyperoptArguments:
    """Argument class for Weights & Biases hyperparameter optimization."""
    def __init__(
        self,
        use_wandb_hyperopt: bool = False,
        wandb_project: str = 'protify-hyperopt',
        wandb_entity: str = None,
        sweep_config_path: str = None,
        sweep_count: int = 50,
        **kwargs
    ):
        self.use_wandb_hyperopt = use_wandb_hyperopt
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.sweep_config_path = sweep_config_path
        self.sweep_count = sweep_count


class WandbHyperparameterOptimizer:
    """
    Hyperparameter optimization using W & B.
    Supports YAML-based sweep configurations for easy customization.
    """
    
    def __init__(
        self, 
        wandb_args: WandbHyperoptArguments,
        wandb_api_key: Optional[str] = None
    ):
        self.wandb_args = wandb_args
        self.wandb_api_key = wandb_api_key
        self._init_wandb()
        
    def _init_wandb(self):
        if self.wandb_api_key:
            wandb.login(key=self.wandb_api_key)
        elif os.getenv('WANDB_API_KEY'):
            wandb.login(key=os.getenv('WANDB_API_KEY'))
        else:
            print_message("Warning: No wandb API key found. Make sure to set WANDB_API_KEY environment variable or provide wandb_api_key.")

    def load_sweep_config_from_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """
        Load sweep configuration from a YAML file.
        
        Args:
            yaml_path: Path to YAML file containing sweep configuration
            
        Returns:
            Dictionary containing the sweep configuration
        """
        try:
            with open(yaml_path, 'r') as file:
                sweep_config = yaml.safe_load(file)
            print_message(f"Loaded sweep configuration from {yaml_path}")
            return sweep_config
        except FileNotFoundError:
            print_message(f"Sweep config file not found: {yaml_path}")
            return self.get_default_sweep_config()
        except yaml.YAMLError as e:
            print_message(f"Error parsing YAML file: {e}")
            return self.get_default_sweep_config()

    def get_default_sweep_config(self) -> Dict[str, Any]:
        """
        Get default sweep configuration for protein models.
        
        Returns:
            Default sweep configuration dictionary
        """
        return {
            'method': 'bayes',
            'metric': {
                'name': 'eval_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'lr': {
                    'distribution': 'log_uniform',
                    'min': 1e-5,
                    'max': 1e-2
                },
                'weight_decay': {
                    'distribution': 'log_uniform',
                    'min': 1e-6,
                    'max': 1e-1
                },
                'probe_batch_size': {
                    'values': [16, 32, 64, 128]
                },
                'dropout': {
                    'distribution': 'uniform',
                    'min': 0.0,
                    'max': 0.5
                },
                'hidden_dim': {
                    'values': [512, 1024, 2048, 4096]
                },
                'n_layers': {
                    'values': [1, 2, 3]
                },
                'classifier_dim': {
                    'values': [1024, 2048, 4096]
                }
            }
        }

    def create_sweep(self, sweep_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a wandb sweep.
        
        Args:
            sweep_config: Sweep configuration (If no YAML file is provided, uses default)
            
        Returns:
            Sweep ID
        """
        if sweep_config is None:
            if self.wandb_args.sweep_config_path:
                sweep_config = self.load_sweep_config_from_yaml(self.wandb_args.sweep_config_path)
            else:
                sweep_config = self.get_default_sweep_config()
        
        sweep_id = wandb.sweep(
            sweep_config,
            project=self.wandb_args.wandb_project,
            entity=self.wandb_args.wandb_entity
        )
        
        print_message(f"Created wandb sweep: {sweep_id}")
        print_message(f"View at: https://wandb.ai/{self.wandb_args.wandb_entity or 'your-entity'}/{self.wandb_args.wandb_project}/sweeps/{sweep_id}")
        
        return sweep_id

    def run_sweep(self, train_function: Callable, sweep_id: str = None, sweep_config: Dict[str, Any] = None):
        """
        Run a wandb sweep.
        
        Args:
            train_function: Function to run for each trial (should use wandb.config)
            sweep_id: Existing sweep ID (if None, creates new sweep)
            sweep_config: Sweep configuration (used if creating new sweep)
        """
        if sweep_id is None:
            sweep_id = self.create_sweep(sweep_config)
        
        wandb.agent(
            sweep_id, 
            train_function, 
            count=self.wandb_args.sweep_count,
            project=self.wandb_args.wandb_project,
            entity=self.wandb_args.wandb_entity
        )

    def get_best_hyperparameters(self, sweep_id: str) -> Dict[str, Any]:
        """
        Extract the best hyperparameters from a completed wandb sweep.
        
        Args:
            sweep_id: The sweep ID to extract best parameters from
            
        Returns:
            Dictionary containing the best hyperparameters
        """
        try:
            api = wandb.Api()
            sweep = api.sweep(f"{self.wandb_args.wandb_entity or wandb.api.default_entity()}/{self.wandb_args.wandb_project}/{sweep_id}")
            
            # Get the best run based on the metric specified in sweep config
            best_run = sweep.best_run()
            
            if best_run is None:
                print_message("No best run found in sweep")
                return {}
            
            best_config = best_run.config
            print_message(f"Best run ID: {best_run.id}")
            print_message(f"Best hyperparameters: {best_config}")
            
            return best_config
            
        except Exception as e:
            print_message(f"Error extracting best hyperparameters: {e}")
            return {}

    def run_sweep_and_get_best_config(self, train_function: Callable, sweep_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run a wandb sweep and return the best hyperparameters.
        
        Args:
            train_function: Function to run for each trial (should use wandb.config)
            sweep_config: Sweep configuration (used if creating new sweep)
            
        Returns:
            Dictionary containing the best hyperparameters
        """
        # Create and run sweep
        sweep_id = self.create_sweep(sweep_config)
        print_message(f"Starting hyperparameter optimization with {self.wandb_args.sweep_count} trials...")
        
        self.run_sweep(train_function, sweep_id)
        
        print_message("Hyperparameter optimization completed. Extracting best parameters...")
        
        # Get best hyperparameters
        best_config = self.get_best_hyperparameters(sweep_id)
        
        return best_config

    def save_default_sweep_config(self, output_path: str = "wandb_sweep_config.yaml"):
        """
        Save the default sweep configuration to a YAML file.
        
        Args:
            output_path: Path where to save the YAML file
        """
        default_config = self.get_default_sweep_config()
        
        with open(output_path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False, indent=2)
        
        print_message(f"Default sweep configuration saved to {output_path}")
        print_message("You can edit this file to customize your hyperparameter search space.")


class WandbHyperoptMixin:
    """
    Mixin class to add wandb hyperparameter optimization capabilities to existing classes.
    """
    
    def __init__(self, *args, wandb_hyperopt_args: Optional[WandbHyperoptArguments] = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.wandb_hyperopt_args = wandb_hyperopt_args
        self.wandb_optimizer = None
        
        if wandb_hyperopt_args and wandb_hyperopt_args.use_wandb_hyperopt:
            # Get wandb_api_key from full_args if it exists
            wandb_api_key = None
            if hasattr(self, 'full_args') and hasattr(self.full_args, 'wandb_api_key'):
                wandb_api_key = self.full_args.wandb_api_key
                
            self.wandb_optimizer = WandbHyperparameterOptimizer(
                wandb_args=wandb_hyperopt_args,
                wandb_api_key=wandb_api_key
            )

    def create_wandb_train_function(
        self, 
        model_name: str, 
        data_name: str, 
        train_dataset, 
        valid_dataset, 
        test_dataset,
        tokenizer=None, 
        emb_dict=None, 
        ppi=False
    ):
        """
        Create a training function for wandb sweeps.
        
        Returns:
            Function that can be used with wandb.agent()
        """
        def train():
            # Initialize wandb run
            wandb.init()
            
            # Get hyperparameters from wandb.config
            config = wandb.config
            
            # Update trainer args with wandb config
            original_trainer_args = self.trainer_args
            original_probe_args = getattr(self, 'probe_args', None)
            
            # Create updated args
            updated_trainer_dict = original_trainer_args.__dict__.copy()
            updated_probe_dict = original_probe_args.__dict__.copy() if original_probe_args else {}
            
            # Update with wandb config
            for key, value in config.items():
                if hasattr(original_trainer_args, key):
                    updated_trainer_dict[key] = value
                if original_probe_args and hasattr(original_probe_args, key):
                    updated_probe_dict[key] = value
            
            # Create new args objects
            from protify.probes.trainers import TrainerArguments
            self.trainer_args = TrainerArguments(**updated_trainer_dict)
            if original_probe_args:
                self.probe_args = SimpleNamespace(**updated_probe_dict)
            
            # Enable wandb logging in trainer
            self.trainer_args.report_to = 'wandb'
            
            try:
                # Run training based on configuration
                if self.trainer_args.full_finetuning:
                    model, _, test_metrics = self.trainer_base_model(
                        model=self.get_model_for_training(model_name),
                        tokenizer=tokenizer,
                        model_name=model_name,
                        data_name=data_name,
                        train_dataset=train_dataset,
                        valid_dataset=valid_dataset,
                        test_dataset=test_dataset,
                        ppi=ppi,
                        log_id=f"wandb_{wandb.run.id}"
                    )
                else:
                    probe = self.get_probe_for_training()
                    model, _, test_metrics = self.trainer_probe(
                        model=probe,
                        tokenizer=tokenizer,
                        model_name=model_name,
                        data_name=data_name,
                        train_dataset=train_dataset,
                        valid_dataset=valid_dataset,
                        test_dataset=test_dataset,
                        emb_dict=emb_dict,
                        ppi=ppi,
                        log_id=f"wandb_{wandb.run.id}"
                    )
                
                # Log final metrics
                wandb.log(test_metrics)
                
            finally:
                # Restore original args
                self.trainer_args = original_trainer_args
                if original_probe_args:
                    self.probe_args = original_probe_args
                
                # Finish wandb run
                wandb.finish()
        
        return train

    def run_wandb_hyperparameter_optimization(
        self,
        model_name: str,
        data_name: str,
        train_dataset,
        valid_dataset,
        test_dataset,
        tokenizer=None,
        emb_dict=None,
        ppi=False,
        sweep_config_path: str = None
    ):
        """
        Run hyperparameter optimization using wandb sweeps.
        
        Args:
            model_name: Name of the model to optimize
            data_name: Name of the dataset
            train_dataset, valid_dataset, test_dataset: Datasets
            tokenizer: Tokenizer (optional)
            emb_dict: Embedding dictionary (optional)
            ppi: Whether this is a PPI task
            sweep_config_path: Path to YAML sweep config (optional)
        """
        if not self.wandb_optimizer:
            print_message("Wandb hyperparameter optimization not enabled. Set use_wandb_hyperopt=True.")
            return None
        
        # Create training function
        train_fn = self.create_wandb_train_function(
            model_name=model_name,
            data_name=data_name,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            emb_dict=emb_dict,
            ppi=ppi
        )
        
        # Load sweep config if provided
        sweep_config = None
        if sweep_config_path:
            sweep_config = self.wandb_optimizer.load_sweep_config_from_yaml(sweep_config_path)
        
        # Run sweep
        self.wandb_optimizer.run_sweep(train_fn, sweep_config=sweep_config)

    def run_hyperopt_and_get_best_config(
        self,
        model_name: str,
        data_name: str,
        train_dataset,
        valid_dataset,
        test_dataset,
        tokenizer=None,
        emb_dict=None,
        ppi=False,
        sweep_config_path: str = None
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization and return the best configuration.
        
        Args:
            model_name: Name of the model to optimize
            data_name: Name of the dataset
            train_dataset, valid_dataset, test_dataset: Datasets
            tokenizer: Tokenizer (optional)
            emb_dict: Embedding dictionary (optional)
            ppi: Whether this is a PPI task
            sweep_config_path: Path to YAML sweep config (optional)
            
        Returns:
            Dictionary containing the best hyperparameters
        """
        if not self.wandb_optimizer:
            print_message("Wandb hyperparameter optimization not enabled. Set use_wandb_hyperopt=True.")
            return {}
        
        # Create training function
        train_fn = self.create_wandb_train_function(
            model_name=model_name,
            data_name=data_name,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            emb_dict=emb_dict,
            ppi=ppi
        )
        
        # Load sweep config if provided
        sweep_config = None
        if sweep_config_path:
            sweep_config = self.wandb_optimizer.load_sweep_config_from_yaml(sweep_config_path)
        
        # Run sweep and get best config
        best_config = self.wandb_optimizer.run_sweep_and_get_best_config(train_fn, sweep_config=sweep_config)
        
        return best_config

    def update_args_with_best_config(self, best_config: Dict[str, Any]):
        """
        Update trainer and probe args with the best hyperparameters.
        
        Args:
            best_config: Dictionary containing the best hyperparameters
        """
        print_message("Updating arguments with best hyperparameters...")
        
        # Update trainer args
        for key, value in best_config.items():
            if hasattr(self.trainer_args, key):
                old_value = getattr(self.trainer_args, key)
                setattr(self.trainer_args, key, value)
                print_message(f"Updated trainer_args.{key}: {old_value} -> {value}")
        
        # Update probe args if they exist
        if hasattr(self, 'probe_args'):
            for key, value in best_config.items():
                if hasattr(self.probe_args, key):
                    old_value = getattr(self.probe_args, key)
                    setattr(self.probe_args, key, value)
                    print_message(f"Updated probe_args.{key}: {old_value} -> {value}")

    def create_sweep_config_template(self, output_path: str = "sweep_config.yaml"):
        """Create a template sweep configuration file."""
        if self.wandb_optimizer:
            self.wandb_optimizer.save_default_sweep_config(output_path)
        else:
            print_message("Wandb optimizer not initialized. Enable hyperparameter optimization first.") 