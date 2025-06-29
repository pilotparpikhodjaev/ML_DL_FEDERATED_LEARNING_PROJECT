# Federated Learning Under the Lens of Model Editing

This repository contains the implementation and experimental results for our research on federated learning with model editing techniques. The project investigates how different masking strategies affect model performance across various data distribution scenarios.

## 🎯 Project Overview

We explore federated learning scenarios with model editing capabilities, focusing on:

- **Centralized Learning**: Traditional centralized training approach
- **IID Federated Learning**: Federated learning with independently and identically distributed data
- **Non-IID Federated Learning**: Federated learning with non-uniform data distribution across clients

The project implements masking strategies for model editing using Vision Transformer (ViT) models pre-trained with DINO.

## 📊 Key Findings

Based on our experimental results:

### Non-IID Scenarios

- **Baseline (no mask)**: 6.54% test accuracy
- **Random Masking (70%)**: 10.11% test accuracy
- **Least-Sensitive Mask**: 6.91% test accuracy

The results demonstrate that masking strategies can improve performance in non-IID federated learning scenarios, with higher masking coefficients showing better results.

## 🏗️ Project Structure

```
├── 📄 Project_5_Federated_Learning_Under_the_Lens_of_Model_Editing.pdf  # Research paper
├── 📄 Report Draft.docx                                                  # Draft report
├── 📄 result.ipynb                                                       # Main results analysis
├── 📁 centralized/                                                       # Centralized learning experiments
│   ├── centralized-baseline.ipynb                                       # Baseline implementation
│   ├── centrilized_masking.ipynb                                        # Masking implementation
│   ├── centralized_mask_vs_baseline.png                                 # Comparison visualization
│   └── 📁 centralized_baseline/, centralized_masking/                   # Results directories
├── 📁 iid/                                                              # IID federated learning
│   ├── iid-baseline.ipynb                                               # IID baseline
│   ├── iid-with-masking.ipynb                                           # IID with masking
│   └── 📁 results_iid/, results_iid_with__mask/                        # Results directories
└── 📁 Non_iid/                                                          # Non-IID federated learning
    ├── non_iid_baseline.ipynb                                           # Non-IID baseline
    ├── non_iid_masking.ipynb                                            # Non-IID with masking
    ├── non_iid_sensative_masking.ipynb                                  # Sensitive masking variant
    ├── comparison_accuracy.png, comparison_loss_epoches.jpg             # Visualizations
    └── 📁 results_j4/, results_j8/                                      # Results directories
```

## 🚀 Getting Started

### Prerequisites

```bash
# Clone the DINO repository (used in notebooks)
git clone https://github.com/facebookresearch/dino.git

# Install required packages
pip install torch torchvision
pip install matplotlib numpy json os
```

### Running the Experiments

1. **Centralized Learning**:

   ```bash
   # Run baseline
   jupyter notebook centralized/centralized-baseline.ipynb

   # Run with masking
   jupyter notebook centralized/centrilized_masking.ipynb
   ```

2. **IID Federated Learning**:

   ```bash
   # Run baseline
   jupyter notebook iid/iid-baseline.ipynb

   # Run with masking
   jupyter notebook iid/iid-with-masking.ipynb
   ```

3. **Non-IID Federated Learning**:

   ```bash
   # Run baseline
   jupyter notebook Non_iid/non_iid_baseline.ipynb

   # Run with masking
   jupyter notebook Non_iid/non_iid_masking.ipynb

   # Run sensitive masking
   jupyter notebook Non_iid/non_iid_sensative_masking.ipynb
   ```

4. **Results Analysis**:
   ```bash
   # Analyze and visualize results
   jupyter notebook result.ipynb
   ```

## 🔬 Methodology

### Model Architecture

- **Base Model**: Vision Transformer (ViT-Small) with patch size 16
- **Pre-training**: DINO self-supervised learning
- **Dataset**: CIFAR-100 (100-class classification)


### Federated Learning Setup

- **Clients**: Multiple clients with different data distributions
- **Aggregation**: FedAvg-style parameter aggregation
- **Communication Rounds**: Multiple rounds of local training and global aggregation

### Masking Strategy

The project implements attention-based masking for model editing:

- Masks are applied to transformer attention layers
- Different masking coefficients (c02, c05) are tested
- Sensitive masking targets specific model components

### Key Visualizations

- **Accuracy Curves**: Training progression across federated rounds
- **Loss Curves**: Convergence analysis
- **Comparison Charts**: Performance across different scenarios

## 🛠️ Technical Implementation

### Key Components

1. **Model Loading**: Pre-trained DINO ViT models
2. **Data Distribution**: Custom data splitting for federated scenarios
3. **Masking Implementation**: Attention layer modifications
4. **Federated Aggregation**: Parameter averaging across clients
5. **Evaluation**: Comprehensive testing on held-out data

### Experiment Management

- Results are automatically saved to structured directories
- JSON files store training metrics
- Visualization scripts generate comparison plots
- Zip files created for easy result sharing 

## 📋 Requirements

```txt
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.20.0
matplotlib>=3.3.0
jupyter>=1.0.0
```

## 📚 References

The project builds upon:

- **DINO**: Self-supervised Vision Transformers
- **Vision Transformer**: Attention-based image classification
- **Federated Learning**: Distributed machine learning paradigm

## 🤝 Contributing

This is a research project. For questions or collaborations:

1. Review the research paper in the repository
2. Check existing experiment notebooks
3. Follow the established code structure for new experiments

## 📄 Citation

If you use this work, please cite our research paper:

```bibtex
@article{federated_model_editing_2025,
  title={Federated Learning Under the Lens of Model Editing},
  author={[Your Names]},
  year={2025},
  journal={[Conference/Journal Name]}
}
```

## 📊 Project Status

✅ **Completed Components**:

- Centralized learning baseline and masking
- IID federated learning experiments
- Non-IID federated learning experiments
- Comprehensive results analysis
- Visualization and comparison tools

🔄 **Future Work**:

- Additional masking strategies
- Different model architectures
- More complex federated scenarios
- Privacy-preserving techniques

## 👥 Project Team

<table>
<tr>
    <td align="center">
        <a href="https://github.com/pilotparpikhodjaev">
            <img src="https://github.com/pilotparpikhodjaev.png" width="100px;" alt="pilotparpikhodjaev"/><br />
            <sub><b>Javokhirbek Parpikhodjaev</b></sub>
        </a><br />
        <sub>s345099</sub>
    </td>
    <td align="center">
        <a href="https://github.com/bekker18">
            <img src="https://github.com/bekker18.png" width="100px;" alt="bekker18"/><br />
            <sub><b>bekker -  Bekzod Kadirov</b></sub>
        </a><br />
        <sub>s333564</sub>
    </td>
    <td align="center">
        <a href="https://github.com/SpaceDevEngineer">
            <img src="https://github.com/SpaceDevEngineer.png" width="100px;" alt="Temurbek Kuchkorov"/><br />
            <sub><b>Temurbek Kuchkorov</b></sub>
        </a><br />
        <sub>S333520</sub>
    </td>
    <td align="center">
        <a href="https://github.com/timchenko69">
            <img src="https://github.com/timchenko69.png" width="100px;" alt="timchenko69"/><br />
            <sub><b>timchenko69 - Timurbek Karimov</b></sub>
        </a><br />
        <sub>S333565</sub>
    </td>
</tr>
</table>

### Acknowledgments

Special thanks to all contributors who made this research project possible.

---

**Note**: This project was developed for research purposes. The code is designed to run in Google Colab environment with GPU support.
