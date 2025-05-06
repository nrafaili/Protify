dataset_descriptions = {
    'EC': {
        'description': 'Enzyme Commission numbers dataset for predicting enzyme function classification.',
        'type': 'Multi-label classification',
        'task': 'Protein function prediction',
        'citation': 'Gleghorn Lab'
    },
    'GO-CC': {
        'description': 'Gene Ontology Cellular Component dataset for predicting protein localization in cells.',
        'type': 'Multi-label classification',
        'task': 'Protein localization prediction',
        'citation': 'Gleghorn Lab'
    },
    'GO-BP': {
        'description': 'Gene Ontology Biological Process dataset for predicting protein involvement in biological processes.',
        'type': 'Multi-label classification',
        'task': 'Protein function prediction',
        'citation': 'Gleghorn Lab'
    },
    'GO-MF': {
        'description': 'Gene Ontology Molecular Function dataset for predicting protein molecular functions.',
        'type': 'Multi-label classification',
        'task': 'Protein function prediction',
        'citation': 'Gleghorn Lab'
    },
    'MB': {
        'description': 'Metal ion binding dataset for predicting protein-metal interactions.',
        'type': 'Classification',
        'task': 'Protein-metal binding prediction',
        'citation': 'Gleghorn Lab'
    },
    'DeepLoc-2': {
        'description': 'Binary classification dataset for predicting protein localization in 2 categories.',
        'type': 'Binary classification',
        'task': 'Protein localization prediction',
        'citation': 'Gleghorn Lab'
    },
    'DeepLoc-10': {
        'description': 'Multi-class classification dataset for predicting protein localization in 10 categories.',
        'type': 'Multi-class classification',
        'task': 'Protein localization prediction',
        'citation': 'Gleghorn Lab'
    },
    'enzyme-kcat': {
        'description': 'Dataset for predicting enzyme catalytic rate constants (kcat).',
        'type': 'Regression',
        'task': 'Enzyme kinetics prediction',
        'citation': 'Gleghorn Lab'
    },
    'solubility': {
        'description': 'Dataset for predicting protein solubility properties.',
        'type': 'Binary classification',
        'task': 'Protein solubility prediction',
        'citation': 'Gleghorn Lab'
    },
    'localization': {
        'description': 'Dataset for predicting subcellular localization of proteins.',
        'type': 'Multi-class classification',
        'task': 'Protein localization prediction',
        'citation': 'Gleghorn Lab'
    },
    'temperature-stability': {
        'description': 'Dataset for predicting protein stability at different temperatures.',
        'type': 'Binary classification',
        'task': 'Protein stability prediction',
        'citation': 'Gleghorn Lab'
    },
    'peptide-HLA-MHC-affinity': {
        'description': 'Dataset for predicting peptide binding affinity to HLA/MHC complexes.',
        'type': 'Protein-protein interaction',
        'task': 'Binding affinity prediction',
        'citation': 'Gleghorn Lab'
    },
    'optimal-temperature': {
        'description': 'Dataset for predicting the optimal temperature for protein function.',
        'type': 'Regression',
        'task': 'Protein property prediction',
        'citation': 'Gleghorn Lab'
    },
    'optimal-ph': {
        'description': 'Dataset for predicting the optimal pH for protein function.',
        'type': 'Regression',
        'task': 'Protein property prediction',
        'citation': 'Gleghorn Lab'
    },
    'material-production': {
        'description': 'Dataset for predicting protein suitability for material production.',
        'type': 'Classification',
        'task': 'Protein application prediction',
        'citation': 'Gleghorn Lab'
    },
    'fitness-prediction': {
        'description': 'Dataset for predicting protein fitness in various environments.',
        'type': 'Classification',
        'task': 'Protein fitness prediction',
        'citation': 'Gleghorn Lab'
    },
    'number-of-folds': {
        'description': 'Dataset for predicting the number of structural folds in proteins.',
        'type': 'Classification',
        'task': 'Protein structure prediction',
        'citation': 'Gleghorn Lab'
    },
    'cloning-clf': {
        'description': 'Dataset for predicting protein suitability for cloning operations.',
        'type': 'Classification',
        'task': 'Protein engineering prediction',
        'citation': 'Gleghorn Lab'
    },
    'stability-prediction': {
        'description': 'Dataset for predicting overall protein stability.',
        'type': 'Classification',
        'task': 'Protein stability prediction',
        'citation': 'Gleghorn Lab'
    },
    'human-ppi': {
        'description': 'Dataset for predicting human protein-protein interactions.',
        'type': 'Protein-protein interaction',
        'task': 'PPI prediction',
        'citation': 'Gleghorn Lab'
    },
    'SecondaryStructure-3': {
        'description': 'Dataset for predicting protein secondary structure in 3 classes.',
        'type': 'Token-wise classification',
        'task': 'Protein structure prediction',
        'citation': 'Gleghorn Lab'
    },
    'SecondaryStructure-8': {
        'description': 'Dataset for predicting protein secondary structure in 8 classes.',
        'type': 'Token-wise classification',
        'task': 'Protein structure prediction',
        'citation': 'Gleghorn Lab'
    },
    'fluorescence-prediction': {
        'description': 'Dataset for predicting protein fluorescence properties.',
        'type': 'Token-wise regression',
        'task': 'Protein property prediction',
        'citation': 'Gleghorn Lab'
    },
    'plastic': {
        'description': 'Dataset for predicting protein capability for plastic degradation.',
        'type': 'Classification',
        'task': 'Enzyme function prediction',
        'citation': 'Gleghorn Lab'
    },
    'gold-ppi': {
        'description': 'Gold standard dataset for protein-protein interaction prediction.',
        'type': 'Protein-protein interaction',
        'task': 'PPI prediction',
        'citation': 'Synthyra/bernett_gold_ppi'
    },
    'human-ppi-pinui': {
        'description': 'Human protein-protein interaction dataset from PiNUI.',
        'type': 'Protein-protein interaction',
        'task': 'PPI prediction',
        'citation': 'Gleghorn Lab'
    },
    'yeast-ppi-pinui': {
        'description': 'Yeast protein-protein interaction dataset from PiNUI.',
        'type': 'Protein-protein interaction',
        'task': 'PPI prediction',
        'citation': 'Gleghorn Lab'
    },
    'shs27-ppi': {
        'description': 'SHS27k dataset containing 27,000 protein-protein interactions.',
        'type': 'Protein-protein interaction',
        'task': 'PPI prediction',
        'citation': 'Synthyra/SHS27k'
    },
    'shs148-ppi': {
        'description': 'SHS148k dataset containing 148,000 protein-protein interactions.',
        'type': 'Protein-protein interaction',
        'task': 'PPI prediction',
        'citation': 'Synthyra/SHS148k'
    },
    'PPA-ppi': {
        'description': 'Protein-Protein Affinity dataset for quantitative binding predictions.',
        'type': 'Protein-protein interaction',
        'task': 'PPI affinity prediction',
        'citation': 'Synthyra/ProteinProteinAffinity'
    },
    'synthyra-ppi': {
        'description': 'Comprehensive protein-protein interaction dataset curated by Synthyra.',
        'type': 'Protein-protein interaction',
        'task': 'PPI prediction',
        'citation': 'Synthyra/ppi_set_v5'
    },
} 