<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Synthyra/Protify">
    <img src="https://github.com/Synthyra/Protify/blob/main/images/github_banner.png" alt="Logo">
  </a>

  <h3 align="center">Protify</h3>

  <p align="center">
    A low code solution for computationally predicting the properties of chemicals.
    <br />
    <a href="https://github.com/Synthyra/Protify/tree/main/docs"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Synthyra/Protify">View Demo</a>
    &middot;
    <a href="https://github.com/Synthyra/Protify/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/Synthyra/Protify/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#why-protify">Why Protify?</a></li>
        <li><a href="#current-key-features">Current Key Features</a></li>
        <li><a href="#support-protifys-development">Support Protify's Development</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#cite">Cite</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Protify is an open source platform designed to simplify and democratize workflows for chemical language models. With Protify, deep learning models can be trained to predict chemical properties at the click of a button, without requiring extensive coding knowledge or computational resources.

### Why Protify?

- **Benchmark multiple models efficiently**: Need to evaluate 10 different protein language models against 15 diverse datasets with publication-ready figures? Protify makes this possible without writing a single line of code.
- **Flexible for all skill levels**: Build custom pipelines with code or use our no-code interface depending on your needs and expertise.
- **Accessible computing**: No GPU? No problem. Synthyra offers precomputed embeddings for many popular datasets, which Protify can download for analysis with scikit-learn on your laptop.
- **Cost-effective solutions**: The upcoming Synthyra API integration will offer affordable GPU training options, while our Colab notebook provides an accessible entry point for GPU-reliant analysis.

Protify is currently in beta. We're actively working to enhance features and documentation to meet our ambitious goals.

### Current Key Features

- **Multiple interfaces**: Run experiments via an intuitive GUI, CLI, or prepared YAML files
- **Efficient embeddings**: Leverage fast and efficient embeddings from ESM2 and ESMC via [FastPLMs](https://github.com/Synthyra/FastPLMs)
  - Coming soon: Additional protein, SMILES, SELFIES, codon, and nucleotide language models
- **Flexible model probing**: Use efficient MLPs for sequence-wise tasks or transformer probes for token-wise tasks
  - Coming soon: Full model fine-tuning, hybrid probing, and LoRA
- **Automated model selection**: Find optimal scikit-learn models for your data with LazyPredict, enhanced by automatic hyperparameter optimization
  - Coming soon: GPU acceleration
- **Complete reproducibility**: Every session generates a detailed log that can be used to reproduce your entire workflow
- **Publication-ready visualizations**: Generate cross-model and dataset comparisons with radar and bar plots, embedding analysis with PCA, t-SNE, and UMAP, and statistically sound confidence interval plots
- **Extensive dataset support**: Access 25 protein datasets by default, or easily integrate your own local or private datasets
  - Coming soon: Additional protein, SMILES, SELFIES, codon, and nucleotide property datasets
- **Advanced interaction modeling**: Support for protein-protein interaction datasets
  - Coming soon: Protein-small molecule interaction capabilities

### Support Protify's Development

Help us grow by sharing online, starring our repository, or contributing through our [bounty program](https://gleghornlab.notion.site/1de62a314a2e808bb6fdc1e714725900?v=1de62a314a2e80389ed7000c97c1a709&pvs=4).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Installation
From pip

`pip install Protify`

To get started locally
```console
git clone https://@github.com/Synthyra/Protify.git
cd Protify
python -m pip install -r docs/requirements.txt
git submodule update --init --remote --recursive
cd src/protify
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

<details>
  <summary>Toggle </summary><br />
  
  To launch the gui, run
  
  ```console
  python -m gui
  ```
  
  It's recommended to use the user interface alongside an open terminal, as helpful messages and progressbars will show in the terminal while you press the GUI buttons.
  
  ### An example workflow
  
  Here, we will compare various protein models against a random vector baseline (negative control) and random transformer (homology based control).
  
  1.) Start the session
  
  <img src="https://github.com/Synthyra/Protify/blob/main/images/example_workflow/1.PNG">
  
  <img src="https://github.com/Synthyra/Protify/blob/main/images/example_workflow/2.PNG" width="500">
  
  2.) Select the models you would like to benchmark
  
  <img src="https://github.com/Synthyra/Protify/blob/main/images/example_workflow/3.PNG" width="500">
  
  3.) Select the datasets you are interested in. Here we chose Enzynme Comission numbers (multi-label classification), metal-ion binding (binary classificaiton), solubility (deeploc2, binary classification), and catalytic rate (kcat, regression).
  
  <img src="https://github.com/Synthyra/Protify/blob/main/images/example_workflow/4.PNG" width="500">
  
  4.) Embed the proteins in the selected datasets. If your machine does not have a GPU, you can download precomputed embeddings for many common sequences.
    Note: If you download embeddings, it will be faster to use the scikit model tab than the probe tab
  <img src="https://github.com/Synthyra/Protify/blob/main/images/example_workflow/5.PNG" width="500">
  
  5.) Select which probe and configuration you would like. Here, we will use a simple linear probe, a type neural network. It is the **fastest** (by a large margin) but worst performing option (by a small margin usually).
  
  <img src="https://github.com/Synthyra/Protify/blob/main/images/example_workflow/6.PNG" width="500">
  
  6.) Select your settings for training. Like most of the tabs, the defaults are pretty good. If you need information about what setting does what, the `?` button provides a helpful note. The documentations has more extensive information
  
  <img src="https://github.com/Synthyra/Protify/blob/main/images/example_workflow/7.PNG" width="500">
  
  This will train your models!
  
  7.) After training, you can render helpful visualizations by passing the log ID from before. If you forget it, you can look for the file generated in the `logs` folder.
  
  <img src="https://github.com/Synthyra/Protify/blob/main/images/example_workflow/8.PNG" width="500">
  
  Here's a sample of the many plots produced. You can find them all inside `plots/your_log_id/*`
  
  <img src="https://github.com/Synthyra/Protify/blob/main/images/example_workflow/9.png" width="500">
  
  <img src="https://github.com/Synthyra/Protify/blob/main/images/example_workflow/10.png" width="500">
  
  <img src="https://github.com/Synthyra/Protify/blob/main/images/example_workflow/11.png" width="500">
  
  <img src="https://github.com/Synthyra/Protify/blob/main/images/example_workflow/13.png" width="500">

  <img src="https://github.com/Synthyra/Protify/blob/main/images/example_workflow/12.png" width="500">
  
  8.) Need to replicate your findings for a report or paper? Just input the generated log into the replay tab
  
  <img src="https://github.com/Synthyra/Protify/blob/main/images/example_workflow/14.PNG" width="500">

  To run the same session from the command line instead, you would simply execute
  ```
  python -m main --model_names ESM2-8 ESM2-35 ESMC-300 Random Random-Transformer --data_names EC DeepLoc-2 enzyme-kcat --patience 3
  ```
  Or, set up a yaml file with your desired settings (so you don't have to type out everything in the CLI)
  ```
  python -m main --yaml_path yamls/your_custom_yaml_path.yaml
  ```
  Replaying from the CLI is just as simple
  ```
  python -m main --replay_path logs/your_log_id.txt
  ```

</details>



<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

We work with a [bounty system](https://gleghornlab.notion.site/1de62a314a2e808bb6fdc1e714725900?v=1de62a314a2e80389ed7000c97c1a709&pvs=4). You can find bounties on this page. Contributing bounties will get you listed on the Protify consortium and potentially coauthorship on published papers involving the framework.

Simply open a pull request with the bounty ID in the title to claim one. For additional features not on the bounty list simply use a descriptive title.

For bugs and general suggestions please use [GitHub issues](https://github.com/Synthyra/Protify/issues).

### Top contributors:

<a href="https://github.com/Synthyra/Protify/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Synthyra/Protify" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With
* [![PyTorch][PyTorch-badge]][PyTorch-url]
* [![Transformers][Transformers-badge]][Transformers-url]
* [![Datasets][Datasets-badge]][Datasets-url]
* [![PEFT][PEFT-badge]][PEFT-url]
* [![scikit-learn][Scikit-learn-badge]][Scikit-learn-url]
* [![NumPy][NumPy-badge]][NumPy-url]
* [![SciPy][SciPy-badge]][SciPy-url]
* [![Einops][Einops-badge]][Einops-url]
* [![PAUC][PAUC-badge]][PAUC-url]
* [![LazyPredict][LazyPredict-badge]][LazyPredict-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the Protify License. See `LICENSE.md` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Email: info@synthyra.com  
Website: [https://synthyra.com](https://synthyra.com)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Cite

If you use this package, please cite the following papers. (Coming soon)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Synthyra/Protify.svg?style=for-the-badge
[contributors-url]: https://github.com/Synthyra/Protify/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Synthyra/Protify.svg?style=for-the-badge
[forks-url]: https://github.com/Synthyra/Protify/network/members
[stars-shield]: https://img.shields.io/github/stars/Synthyra/Protify.svg?style=for-the-badge
[stars-url]: https://github.com/Synthyra/Protify/stargazers
[issues-shield]: https://img.shields.io/github/issues/Synthyra/Protify.svg?style=for-the-badge
[issues-url]: https://github.com/Synthyra/Protify/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/company/synthyra
[product-screenshot]: images/screenshot.png

[Transformers-badge]: https://img.shields.io/badge/Hugging%20Face-Transformers-FF6C44?style=for-the-badge&logo=Huggingface&logoColor=white  
[Transformers-url]: https://github.com/huggingface/transformers

[PyTorch-badge]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white  
[PyTorch-url]: https://github.com/pytorch/pytorch

[Datasets-badge]: https://img.shields.io/badge/Hugging%20Face-Datasets-0078D4?style=for-the-badge&logo=Huggingface&logoColor=white  
[Datasets-url]: https://github.com/huggingface/datasets

[Scikit-learn-badge]: https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white  
[Scikit-learn-url]: https://github.com/scikit-learn/scikit-learn

[NumPy-badge]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white  
[NumPy-url]: https://github.com/numpy/numpy

[SciPy-badge]: https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white  
[SciPy-url]: https://github.com/scipy/scipy

[PAUC-badge]: https://img.shields.io/badge/PAUC-Package-4B8BBE?style=for-the-badge&logo=python&logoColor=white  
[PAUC-url]: https://pypi.org/project/pauc

[LazyPredict-badge]: https://img.shields.io/badge/LazyPredict-Modeling-4B8BBE?style=for-the-badge&logo=python&logoColor=white  
[LazyPredict-url]: https://github.com/shankarpandala/lazypredict

[PEFT-badge]: https://img.shields.io/badge/PEFT-HuggingFace-713196?style=for-the-badge&logo=Huggingface&logoColor=white  
[PEFT-url]: https://github.com/huggingface/peft

[Einops-badge]: https://img.shields.io/badge/Einops-Transform-4B8BBE?style=for-the-badge&logo=python&logoColor=white  
[Einops-url]: https://github.com/arogozhnikov/einops
