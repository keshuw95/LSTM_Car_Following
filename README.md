# [IEEE-TITS] A Deep Long Short-Term Memory Network Embedded Model Predictive Control Strategies for Car-Following Control of Connected Automated Vehicles in Mixed Traffic

This repository contains the implementation of the research presented in the paper:

> **"A Deep Long Short-Term Memory Network Embedded Model Predictive Control Strategies for Car-Following Control of Connected Automated Vehicles in Mixed Traffic"**  
> Published in *IEEE Transactions on Intelligent Transportation Systems (T-ITS)*, 2024  
> [DOI: 10.1109/TITS.2024.3412329](https://doi.org/10.1109/TITS.2024.3412329)



## Overview
This repository contains two versions of an LSTM-based trajectory prediction model for vehicle motion:
1. **Position-based Model**: Predicts future positions based on historical positions.
2. **Speed-based Model**: Predicts future speeds and integrates them to obtain positions.

Both versions leverage an LSTM network to process historical data and generate future trajectory predictions.

---

## Repository Structure
```bash
├── main.py                 # Training and evaluation script for the position-based model
├── main_speed.py           # Training and evaluation script for the speed-based model
├── model.py                # LSTM architecture for position prediction
├── model_speed.py          # LSTM architecture for speed prediction
├── model/                  # Directory for saving trained position-based models
├── model_speed/            # Directory for saving trained speed-based models
├── data/                   # Directory containing input data (not included in repo)
├── output/                 # Directory for model outputs and evaluation plots
├── README.md               # Documentation
```

---

## 1. Position-based Model (`main.py` and `model.py`)
### **Description**
- **Inputs**: Past vehicle positions \( x_{t-M}, ..., x_t \)
- **Outputs**: Future vehicle positions \( \hat{x}_{t+1}, ..., \hat{x}_{t+N} \)
- **Loss function**:
  - Mean Squared Error (MSE) loss:
    ```math
    \mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{t=1}^{N} (x_t - \hat{x}_t)^2
    ```
  - Velocity consistency constraints: Ensures smooth motion without unrealistic negative speed values.

### **Velocity Consistency Constraints Implementation**
To enforce physically plausible movement, we introduce a velocity constraint term:
- Compute velocity as:
  ```math
  v_t = \frac{x_{t+1} - x_t}{\Delta t}
  ```
- Apply a penalty for negative velocity:
  ```math
  \mathcal{L}_{\text{velocity}} = \lambda \sum_{t=1}^{N-1} \max(0, -v_t)^2
  ```
- Final loss function:
  ```math
  \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda \mathcal{L}_{\text{velocity}}
  ```

### **Model Architecture (`model.py`)**
- Uses an LSTM to encode the trajectory sequence.
- Predicts future positions directly through a linear layer.

### **Training (`main.py`)**
- Data preprocessing extracts vehicle positions.
- Uses a loss function that penalizes negative speeds to enforce realistic motion constraints.
- Model is trained using Adam optimizer with learning rate decay.

---

## 2. Speed-based Model (`main_speed.py` and `model_speed.py`)
### **Description**
- **Inputs**: Past vehicle speeds \( v_{t-M}, ..., v_t \)
- **Outputs**: Future vehicle speeds \( \hat{v}_{t+1}, ..., \hat{v}_{t+N} \)
- **Loss function**:
  - Mean Squared Error (MSE) loss:
    ```math
    \mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{t=1}^{N} (v_t - \hat{v}_t)^2
    ```
  - Unlike the position-based model, **no additional velocity constraints** are applied.

### **Model Architecture (`model_speed.py`)**
- Uses an LSTM to encode historical speed data.
- Predicts future speed using a **ReLU activation** to ensure non-negative speeds.

### **Training (`main_speed.py`)**
- Data preprocessing extracts vehicle speeds.
- Model is trained using standard MSE loss without explicit velocity constraints.
- Final positions are computed via numerical integration:
  ```math
  \hat{x}_{t+1} = x_t + \sum_{k=1}^{t+N} \hat{v}_k \cdot \Delta t
  ```
  where \( \Delta t \) is the discrete time step (e.g., 0.1s).

---

## 3. Installation & Setup
### **Dependencies**
Ensure you have the following installed:
```bash
pip install numpy pandas torch matplotlib
```

### **Running the Models**
- **Train the position-based model:**
  ```bash
  python main.py
  ```
- **Train the speed-based model:**
  ```bash
  python main_speed.py
  ```

---

## 4. Evaluation
The models generate plots comparing predicted and actual trajectories. Evaluation metrics include:
- **Root Mean Squared Error (RMSE)** for position accuracy:
  ```math
  \text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^{N} (x_t - \hat{x}_t)^2 }
  ```
- **Speed integration accuracy** (for speed-based model), ensuring smooth position reconstruction.

---

## 5. Future Improvements
- Incorporate external factors like traffic signals.
- Experiment with transformer-based sequence models.
- Fine-tune loss functions for better motion constraints.

For any questions, please open an issue or reach out! 🚀
