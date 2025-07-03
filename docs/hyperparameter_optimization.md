# Hyperparameter Optimization with Weights & Biases

Protify now supports automated hyperparameter optimization using Weights & Biases (wandb) sweeps. This feature allows you to automatically find the best hyperparameters for your protein prediction tasks.

## Quick Start

1. **Enable hyperparameter optimization**:
   ```bash
   python main.py --use_wandb_hyperopt --wandb_project "my-protein-project"
   ```

2. **Use a custom sweep configuration**:
   ```bash
   python main.py --use_wandb_hyperopt --sweep_config_path "path/to/sweep_config.yaml"
   ```

## How It Works

When hyperparameter optimization is enabled, Protify will:

1. **Run hyperparameter search**: Execute multiple training runs with different hyperparameter combinations using wandb sweeps
2. **Find best configuration**: Automatically identify the hyperparameters that achieve the best performance
3. **Final training**: Use the optimal hyperparameters to train the final model on your full dataset

## Configuration Options

### Command Line Arguments

- `--use_wandb_hyperopt`: Enable wandb hyperparameter optimization
- `--wandb_project`: W&B project name (default: "protify-hyperopt")
- `--wandb_entity`: W&B entity/username
- `--sweep_config_path`: Path to custom YAML sweep configuration
- `--sweep_count`: Number of hyperparameter optimization trials (default: 50)

### Environment Variables

Set your W&B API key:
```bash
export WANDB_API_KEY="your_wandb_api_key_here"
```

## Sweep Configuration

### Using Default Configuration

If no sweep config is provided, Protify uses a default configuration optimizing common hyperparameters:

- Learning rate (1e-5 to 1e-2)
- Weight decay (1e-6 to 1e-1)
- Batch size [16, 32, 64, 128]
- Dropout (0.0 to 0.5)
- Hidden dimensions [512, 1024, 2048, 4096]
- Number of layers [1, 2, 3, 4]

### Custom Configuration

Create a YAML file to customize your hyperparameter search space:

```yaml
method: bayes  # or 'grid' or 'random'
metric:
  name: eval_loss  # metric to optimize
  goal: minimize   # or 'maximize'

parameters:
  lr:
    distribution: log_uniform
    min: 1e-5
    max: 1e-2
  
  probe_batch_size:
    values: [32, 64, 128]
  
  hidden_dim:
    values: [1024, 2048, 4096]
```

See `src/protify/yamls/sweep_config_example.yaml` for a complete example.

## Training Modes

Hyperparameter optimization works with all training modes:

### Probe Training (default)
```bash
python main.py --use_wandb_hyperopt --model_names ESM2-8 --data_names DeepLoc-2
```

### Full Fine-tuning
```bash
python main.py --use_wandb_hyperopt --full_finetuning --model_names ESM2-8
```

### Hybrid Probe
```bash
python main.py --use_wandb_hyperopt --hybrid_probe --model_names ESM2-8
```

## Complete Example

```bash
# Set up W&B authentication
export WANDB_API_KEY="your_api_key"

# Run hyperparameter optimization for protein localization
python main.py \
  --use_wandb_hyperopt \
  --wandb_project "protein-localization-optimization" \
  --wandb_entity "your-username" \
  --sweep_config_path "custom_sweep.yaml" \
  --sweep_count 30 \
  --model_names ESM2-8 ESM2-35 \
  --data_names DeepLoc-2 \
  --probe_type transformer \
  --num_epochs 50
```

## Monitoring Progress

1. **View in W&B**: The sweep URL will be printed to the console
2. **Track metrics**: Monitor training progress, loss curves, and hyperparameter relationships
3. **Compare runs**: Analyze which hyperparameters lead to best performance

## Best Practices

1. **Start small**: Begin with fewer trials (10-20) to validate your setup
2. **Choose the right metric**: Optimize for the metric most important to your task
3. **Set reasonable ranges**: Use domain knowledge to set sensible hyperparameter bounds
4. **Consider computational budget**: Balance number of trials with training time per trial

## Troubleshooting

### Common Issues

1. **W&B authentication errors**: Ensure `WANDB_API_KEY` is set correctly
2. **Sweep config errors**: Validate your YAML syntax and parameter ranges
3. **Memory issues**: Reduce batch sizes or model sizes for resource-constrained environments

### Getting Help

- Check W&B documentation: https://docs.wandb.ai/guides/sweeps
- Review sweep configuration examples in `src/protify/yamls/`
- Monitor console output for detailed error messages

## Output

After hyperparameter optimization completes:

1. **Best hyperparameters** are printed to console
2. **Final model** is trained with optimal settings
3. **Results** are saved to your normal output directories
4. **W&B dashboard** contains detailed analysis of all trials 