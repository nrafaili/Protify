model_descriptions = {
    'ESM2-8': {
        'description': 'Small protein language model (8M parameters) from Meta AI that learns evolutionary information from millions of protein sequences.',
        'size': '8M parameters',
        'type': 'Protein language model',
        'citation': 'Lin et al. (2022). Evolutionary-scale prediction of atomic level protein structure with a language model.'
    },
    'ESM2-35': {
        'description': 'Medium-sized protein language model (35M parameters) trained on evolutionary data.',
        'size': '35M parameters',
        'type': 'Protein language model',
        'citation': 'Lin et al. (2022). Evolutionary-scale prediction of atomic level protein structure with a language model.'
    },
    'ESM2-150': {
        'description': 'Large protein language model (150M parameters) with improved protein structure prediction capabilities.',
        'size': '150M parameters',
        'type': 'Protein language model',
        'citation': 'Lin et al. (2022). Evolutionary-scale prediction of atomic level protein structure with a language model.'
    },
    'ESM2-650': {
        'description': 'Very large protein language model (650M parameters) offering state-of-the-art performance on many protein prediction tasks.',
        'size': '650M parameters',
        'type': 'Protein language model',
        'citation': 'Lin et al. (2022). Evolutionary-scale prediction of atomic level protein structure with a language model.'
    },
    'ESM2-3B': {
        'description': 'Largest ESM2 protein language model (3B parameters) with exceptional capability for protein structure and function prediction.',
        'size': '3B parameters',
        'type': 'Protein language model',
        'citation': 'Lin et al. (2022). Evolutionary-scale prediction of atomic level protein structure with a language model.'
    },
    'Random': {
        'description': 'Baseline model with randomly initialized weights, serving as a negative control.',
        'size': 'Varies',
        'type': 'Baseline control',
        'citation': 'N/A'
    },
    'Random-Transformer': {
        'description': 'Randomly initialized transformer model serving as a homology-based control.',
        'size': 'Varies',
        'type': 'Baseline control',
        'citation': 'N/A'
    },
    'ESMC-300': {
        'description': 'Protein language model optimized for classification tasks with 300M parameters.',
        'size': '300M parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'ESMC-600': {
        'description': 'Larger protein language model for classification with 600M parameters.',
        'size': '600M parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'ProtBert': {
        'description': 'BERT-based protein language model trained on protein sequences from UniRef.',
        'size': '420M parameters',
        'type': 'Protein language model',
        'citation': 'Elnaggar et al. (2021). ProtTrans: Towards Cracking the Language of Life\'s Code Through Self-Supervised Learning.'
    },
    'ProtBert-BFD': {
        'description': 'BERT-based protein language model trained on BFD database with improved performance.',
        'size': '420M parameters',
        'type': 'Protein language model',
        'citation': 'Elnaggar et al. (2021). ProtTrans: Towards Cracking the Language of Life\'s Code Through Self-Supervised Learning.'
    },
    'ProtT5': {
        'description': 'T5-based protein language model capable of both encoding and generation tasks.',
        'size': '3B parameters',
        'type': 'Protein language model',
        'citation': 'Elnaggar et al. (2021). ProtTrans: Towards Cracking the Language of Life\'s Code Through Self-Supervised Learning.'
    },
    'ProtT5-XL-UniRef50-full-prec': {
        'description': 'Extra large T5-based protein language model trained on UniRef50 with full precision.',
        'size': '11B parameters',
        'type': 'Protein language model',
        'citation': 'Elnaggar et al. (2021). ProtTrans: Towards Cracking the Language of Life\'s Code Through Self-Supervised Learning.'
    },
    'ANKH-Base': {
        'description': 'Base version of the ANKH protein language model focused on protein structure understanding.',
        'size': '400M parameters',
        'type': 'Protein language model',
        'citation': 'Choromanski et al. (2022). ANKH: Optimized Protein Language Model Unlocks General-Purpose Modelling.'
    },
    'ANKH-Large': {
        'description': 'Large version of the ANKH protein language model with improved structural predictions.',
        'size': '1.2B parameters',
        'type': 'Protein language model',
        'citation': 'Choromanski et al. (2022). ANKH: Optimized Protein Language Model Unlocks General-Purpose Modelling.'
    },
    'ANKH2-Large': {
        'description': 'Improved second generation ANKH protein language model.',
        'size': '1.5B parameters',
        'type': 'Protein language model',
        'citation': 'Choromanski et al. (2022). ANKH: Optimized Protein Language Model Unlocks General-Purpose Modelling.'
    },
    'GLM2-150': {
        'description': 'Medium-sized general language model adapted for protein sequences.',
        'size': '150M parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'GLM2-650': {
        'description': 'Large general language model adapted for protein sequences.',
        'size': '650M parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'GLM2-GAIA': {
        'description': 'Specialized GLM protein language model with GAIA architecture improvements.',
        'size': '1B+ parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'DPLM-150': {
        'description': 'Deep protein language model with 150M parameters focused on protein structure.',
        'size': '150M parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'DPLM-650': {
        'description': 'Larger deep protein language model with 650M parameters.',
        'size': '650M parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'DPLM-3B': {
        'description': 'Largest deep protein language model in the DPLM family with 3B parameters.',
        'size': '3B parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'DLM-150': {
        'description': 'Deep language model for proteins with 150M parameters.',
        'size': '150M parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    },
    'DLM-650': {
        'description': 'Deep language model for proteins with 650M parameters.',
        'size': '650M parameters',
        'type': 'Protein language model',
        'citation': 'N/A'
    }
} 