<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
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
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Protify is an open source effort to simplify and democratize workflows for chemical language models, where deep learning models can be trained to predict chemical properties at the click of a button. Need to benchmark 10 different protein language models against 15 diverse datasets, and want publication ready figures when the models are done? With Protify you can do this without writing any code. Looking to build more custom pipelines and ready to write some code yourself? That is also easy with the Protify ecosystem. No GPU? No problem. Synthyra offers precomputed embeddings for many popular datasets, and Protify can download them for analysis with scikit-learn on your laptop. Soon, the Synthyra API will be integrated, offering low cost solutions for GPU training. Our colab notebook is also an affordable way to dive into GPU reliant analysis. 

Protify is a new project in beta. Bear with us as we get the features and documentation on par with our ambitions. 

### Current Key features
- Offers fast and efficient embeddings from ESM2 and ESMC from [FastPLMs](https://github.com/Synthyra/FastPLMs)
  - More protein, smiles, selfies, codon, and nucleotide language models to come
- Probe base models with efficient MLPs for sequence-wise tasks, or use transformer probes for token-wise tasks
  - Full model fine tuning, hybrid probing, and Lora coming soon
- Find the best scikit-learn model for your model / dataset with LazyPredict, and then use our automatic hyperparameter search to improve the model further
  - GPU enhancement here in the future
- Every session gets an informative log that is completely reproducible. Simple pass the log file as the `replay_path` and have your entire session repeat infront of your eyes
- Publication ready figures including radar plots, bar plots, custom CI regression plots, and utilizing [pAUC](https://github.com/srijitseal/pauc) for statistically sound classification CI plots
- 25 protein datasets offered by default, utilize your own local or private datasets with ease
 - More protein, smiles, selfies, codon, and nucleotide property datasets to come
- Supports protein-protein interaction datasets
 - Protein small molecule to come


### Want to support Protify?
Share online everywhere and anywhere! Contribute or suggest features with our [bounty program](https://gleghornlab.notion.site/1de62a314a2e808bb6fdc1e714725900?v=1de62a314a2e80389ed7000c97c1a709&pvs=4) (mentioned below).

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

Examples coming soon.

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

info@synthyra.com

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
