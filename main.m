clc; clear;

% dados = load('C:\Users\Raul Melo\Documents\MLP\dataIris.dat');
% dados = load('C:\Users\Raul Melo\Documents\MLP\coluna_vertebral.dat');
dados = load('C:\Users\Raul Melo\Documents\MLP\dermatologia.dat');

epocas = 200;
numRealizacoes = 10;
esc = 1;
numAtributos = 34;                   % Iris 4, coluna 6, derm 34
numNeuroniosOcultos = 10;
numNeuroniosSaida = 6;              % Iris 3, coluna 3, derm 6
offset = 0.5;
numPadroes = size(dados,1);

numNeuronioInicial = 2;
numNeuronioFinal = 18;
incNeuronios = 1;
deltaNeuronio = numNeuronioInicial:incNeuronios:numNeuronioFinal;

dados = [normaliza(dados(:,1:numAtributos))... 
    dados(:,numAtributos+1:numAtributos+numNeuroniosSaida)];

for i = 1:numRealizacoes

    dados = dados(randperm(numPadroes),:);
    
    [X_treino, Y_treino, X_teste, Y_teste] = separaDados(dados,...
        numAtributos,numNeuroniosSaida, 0.8);
    dadosTreino = [X_treino Y_treino];
    
    [W1,W2] = treino( numAtributos, numNeuroniosSaida,...
        numNeuroniosOcultos, X_treino, Y_treino , epocas);
    acc(i) = teste(W1,W2,X_teste,Y_teste);

end


media_acuracia = mean(acc)
desvioPadrao = std(acc)
