# ResNet18 Response-Based Knowledge Distillation Methods

## 📘 Overview
In this project, we investigate **response-based knowledge distillation** using the **ResNet** architecture for an **image classification** task on a custom **weather dataset**.

Our main objective is to transfer the knowledge from a larger, more accurate teacher model (ResNet50) to a smaller, faster student model (ResNet18) without a significant drop in performance.

> This article is aimed at sharing conceptual insights and training dynamics rather than providing step-by-step coding instructions. 

## 🎯 Motivation
**ResNet50** have demonstrated outstanding performance in image classification tasks. However, the computational cost and memory can be expensive for deployment on edge devices or in real-time applications. Knowedge distillation offer a promising solution by transfering knowledge from a heavy model (teacher) to a a small, lightweight one (student), thus achieving a balance between performance and efficiency

## 🧪 Dataset
We use a weather classification dataset consisting of images sorted into different categories (sunny, rainy, ...). The dataset is splited into training (70%), validation (20%) and testing (10%).

## 🧱 Architecture

**Teacher Model: ResNet50**
- Deep and expensive model trained on the dataset
- Achieve high validation accuracy but has a large number of parameters

**Student Model: ResNet18**
- Fewer layers and parameters
- Learns to mimic the teacher's reponses via knowledge distillation

## 🔥 Knowledge Distillation Strategy
We adopt the response-based (a.k.a. logits-based) knowledge distillation method. The goal is to align the output distributions (logits) of the student model with those of the teacher using the soft targets approach proposed by Hinton et al.

**Key points:**
- Softmax temperature (T): Used to soften the teacher's output probability
- Cross-Entropy Loss: Measures how well the student matches the ground truth labels
- KL-Divergence: Measures how well the student mimics the teacher's soft predictions

**Final Loss Function:**
$$
\mathcal{L}_{KD} = \lambda_{soft} \cdot \text{KL}(\text{Soft}_T(\text{teacher}) \,\|\, \text{Soft}_T(\text{student})) + \lambda_{CE} \cdot \text{CE}(\text{student}, \text{labels})
$$

Where:
- `Soft_T(*)` is the softmax output with temperature `T`.
- `λ_soft`, `λ_CE` are weight coefficients for soft-target and label loss.

**Training Details**
- Optimizer: Adam
- Learning rate: 1e-2 (with LambdaLR scheduler)
- Epochs: 30
- Temperature (T): 2
- Weights: 0.25 (soft loss), 0.75 (hard loss)

The scheduler includes a warmup and decay strategy to help stablize early training and adaptively reduce the learning rate

## 📈 Evaluation Metrics
Models are evaluated using:
- Validation Loss / Accuracy
- Test Loss / Accuracy

Performance is printed per epoch, enabling analysis of training dynamics and convergence behavior.

## 📌 Final Remarks
This project highlights the practical impact of knowledge distillation when using pre-trained teacher models and tuning appropriate hyperparameters. ResNet18, aided by a well-trained ResNet50, can serve as an efficient alternative in resource-constrained environments.