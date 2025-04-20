# Wave Excitation Predictor using LSTM

This MATLAB function `predictor.m` predicts future wave excitation moments using a Long Short-Term Memory (LSTM) neural network.

## Features
- Automatically loads or trains an LSTM model for sequence prediction
- Uses previous excitation data (past values) to forecast future time steps
- Saves the trained model to avoid retraining
- Handles noisy or short input sequences robustly

## Inputs
- `past_excM`: Vector of past excitation moments
- `ARorder`: Number of past time steps used as input (input window size)
- `Np`: Number of future steps to predict
- `t`: Current timestep (not used currently, placeholder for future use)

## Output
- `yf`: A vector of predicted values (size 1 x Np)

## Requirements
- MATLAB with Deep Learning Toolbox
- `trainedLSTM.mat` (optional — auto-generated if missing)

## Example Usage
```matlab
past_data = sin(0:0.1:50); % sample input
ARorder = 50;
Np = 10;
t = 100;
predicted = predictor(past_data, ARorder, Np, t);
disp(predicted);
```

## Author
Salar Setorg  
GitHub: https://github.com/salarsetorg  
ResearchGate: https://www.researchgate.net/profile/Salar-Setorg
