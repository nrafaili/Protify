#!/usr/bin/env python3
"""
Simple test script to verify hyperparameter optimization functionality.
This script tests the basic hyperopt workflow without running actual wandb sweeps.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from types import SimpleNamespace
from protify.hyperopt import WandbHyperoptArguments, WandbHyperparameterOptimizer, WandbHyperoptMixin
from protify.probes.trainers import TrainerArguments
from protify.probes.get_probe import ProbeArguments
from protify.utils import print_message


def test_wandb_hyperopt_arguments():
    """Test WandbHyperoptArguments initialization."""
    print_message("Testing WandbHyperoptArguments...")
    
    # Test default initialization
    args = WandbHyperoptArguments()
    assert args.use_wandb_hyperopt == False
    assert args.wandb_project == 'protify-hyperopt'
    assert args.sweep_count == 50
    
    # Test custom initialization
    args = WandbHyperoptArguments(
        use_wandb_hyperopt=True,
        wandb_project='test-project',
        sweep_count=10
    )
    assert args.use_wandb_hyperopt == True
    assert args.wandb_project == 'test-project'
    assert args.sweep_count == 10
    
    print_message("✓ WandbHyperoptArguments tests passed")


def test_wandb_hyperparameter_optimizer():
    """Test WandbHyperparameterOptimizer basic functionality."""
    print_message("Testing WandbHyperparameterOptimizer...")
    
    args = WandbHyperoptArguments(
        use_wandb_hyperopt=True,
        wandb_project='test-project'
    )
    
    # Don't actually initialize wandb for testing
    try:
        optimizer = WandbHyperparameterOptimizer(args)
    except Exception as e:
        # Expected to fail due to wandb not being configured
        print_message(f"Expected wandb initialization warning: {e}")
    
    # Test default sweep config
    default_config = WandbHyperparameterOptimizer(args).get_default_sweep_config()
    assert 'method' in default_config
    assert 'metric' in default_config
    assert 'parameters' in default_config
    assert 'lr' in default_config['parameters']
    
    print_message("✓ WandbHyperparameterOptimizer tests passed")


def test_wandb_hyperopt_mixin():
    """Test WandbHyperoptMixin functionality."""
    print_message("Testing WandbHyperoptMixin...")
    
    class TestClass(WandbHyperoptMixin):
        def __init__(self):
            self.trainer_args = TrainerArguments(model_save_dir="test")
            self.probe_args = ProbeArguments()
            
            wandb_args = WandbHyperoptArguments(use_wandb_hyperopt=False)
            super().__init__(wandb_hyperopt_args=wandb_args)
    
    test_obj = TestClass()
    assert hasattr(test_obj, 'wandb_hyperopt_args')
    assert hasattr(test_obj, 'wandb_optimizer')
    
    # Test update_args_with_best_config
    best_config = {
        'lr': 0.001,
        'hidden_dim': 2048,
        'dropout': 0.1
    }
    
    test_obj.update_args_with_best_config(best_config)
    assert test_obj.trainer_args.lr == 0.001
    assert test_obj.probe_args.hidden_dim == 2048
    assert test_obj.probe_args.dropout == 0.1
    
    print_message("✓ WandbHyperoptMixin tests passed")


def test_yaml_config_loading():
    """Test YAML configuration loading."""
    print_message("Testing YAML configuration loading...")
    
    args = WandbHyperoptArguments()
    optimizer = WandbHyperparameterOptimizer(args)
    
    # Test loading non-existent file (should return default)
    config = optimizer.load_sweep_config_from_yaml("non_existent_file.yaml")
    assert 'method' in config
    assert 'parameters' in config
    
    # Test saving default config
    output_path = "test_sweep_config.yaml"
    optimizer.save_default_sweep_config(output_path)
    
    # Test loading the saved config
    saved_config = optimizer.load_sweep_config_from_yaml(output_path)
    assert saved_config['method'] == 'bayes'
    assert 'lr' in saved_config['parameters']
    
    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)
    
    print_message("✓ YAML configuration tests passed")


def main():
    """Run all tests."""
    print_message("Starting hyperparameter optimization tests...")
    
    try:
        test_wandb_hyperopt_arguments()
        test_wandb_hyperparameter_optimizer()
        test_wandb_hyperopt_mixin()
        test_yaml_config_loading()
        
        print_message("🎉 All hyperparameter optimization tests passed!")
        
    except Exception as e:
        print_message(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 