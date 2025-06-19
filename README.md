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

- **Baseline (j4)**: 2.07% test accuracy
- **With Masking (j4X)**: 3.98% test accuracy
- **With Masking (c02)**: 4.18% test accuracy
- **With Masking (c05)**: 5.12% test accuracy

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
- **Dataset**: CIFAR-10 (inferred from 10-class classification)

### Federated Learning Setup

- **Clients**: Multiple clients with different data distributions
- **Aggregation**: FedAvg-style parameter aggregation
- **Communication Rounds**: Multiple rounds of local training and global aggregation

### Masking Strategy

The project implements attention-based masking for model editing:

- Masks are applied to transformer attention layers
- Different masking coefficients (c02, c05) are tested
- Sensitive masking targets specific model components

## 📈 Results

### Performance Comparison

The [`result.ipynb`](result.ipynb) notebook provides comprehensive analysis including:

- Validation accuracy curves across training rounds
- Loss convergence analysis
- Final test accuracy comparisons
- Statistical significance testing

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
@article{federated_model_editing_2024,
  title={Federated Learning Under the Lens of Model Editing},
  author={[Your Names]},
  year={2024},
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

---

**Note**: This project was developed for research purposes. The code is designed to run in Google Colab environment with GPU support.
