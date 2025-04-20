function yf = predictor(past_excM, ARorder, Np, t)
% predictor: Predicts future wave excitation moments using a trained LSTM model.
%
% Inputs:
% - past_excM : Vector of past excitation moments (input time series data)
% - ARorder   : Number of past time steps used as input (sliding window size)
% - Np        : Number of steps to predict into the future
% - t         : Current time step (not used but reserved for future use)
%
% Output:
% - yf        : Predicted future values (1 x Np)

persistent net isTrained
if isempty(isTrained)
    isTrained = false;
end

if ~isTrained
    if exist('trainedLSTM.mat','file')==2
        tmp = load('trainedLSTM.mat','net');
        if isfield(tmp,'net')
            net = tmp.net;
            isTrained = true;
            disp('Loaded trained LSTM model.');
        end
    end
end

if ~isTrained
    disp('Training LSTM model (one-time)...');
    waveData = double(past_excM(:));
    totalLen = length(waveData);
    if totalLen < ARorder + Np
        error('past_excM too short to train.');
    end

    trainRatio = 0.8;
    trainLen = floor(trainRatio * totalLen);
    trainData = waveData(1:trainLen);

    XTrain = {};
    YTrain = [];
    for i = 1:(length(trainData)-ARorder-Np+1)
        inputSeq = trainData(i:i+ARorder-1)';
        outputSeq = trainData(i+ARorder:i+ARorder+Np-1)';
        XTrain{end+1} = inputSeq;
        YTrain(end+1,:) = outputSeq;
    end

    layers = [
        sequenceInputLayer(1)
        lstmLayer(100,'OutputMode','last')
        fullyConnectedLayer(Np)
        regressionLayer
    ];

    options = trainingOptions('adam',...
        'MaxEpochs',100,...
        'MiniBatchSize',64,...
        'InitialLearnRate',0.005,...
        'Shuffle','every-epoch',...
        'Verbose',false);

    net = trainNetwork(XTrain,YTrain,layers,options);
    save('trainedLSTM.mat','net');
    isTrained = true;
    disp('Model trained and saved.');
end

inData = double(past_excM(:)');
if length(inData)<ARorder
    needed = ARorder - length(inData);
    inData = [repmat(inData(1),1,needed), inData];
end
currentWindow = inData(end-ARorder+1:end);

ypred = predict(net, {currentWindow});
yf = double(ypred(:));

if any(isnan(yf)) || any(isinf(yf))
    yf(isnan(yf)|isinf(yf)) = 0;
end

yf = max(min(yf,10),-10);

if yf(1)>=yf(2)
    yf(1:2) = sort(yf(1:2));
end

end
