function [ W, M ] = treino( numAtributos, numNeuroniosSaida, numNeuroniosOcultos, X_treino, Y_treino , epocas)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here()
    X_treino = X_treino';
    [n,m] = size(X_treino);
    Y_treino = Y_treino';
    taxaAprendizado = 0.1;
    biasCamadaOculta = ones(1,m); 
    X = [biasCamadaOculta;X_treino];
    W = rand(numNeuroniosOcultos,n+1)-0.5;
    M = rand(numNeuroniosSaida,numNeuroniosOcultos+1)-0.5;
    
    for i = 1: epocas
        for i = 1 : m 
        
            camadaOcultaUi = W*X(:,i); 

            h1 = logsig(camadaOcultaUi);
            biasCamadaSaida = 1;
            h = [biasCamadaSaida; h1]; 

            camadaSaidaUi = M*h;  

            Y = logsig(camadaSaidaUi); 

            erro = Y_treino(:,i) - Y; 
            
            Yl = Y.*(ones(size(Y,1),1)-Y); 
            gradienteCmdSaida = erro .*Yl;
%             grad = y'(t)erro(t)
%             mji(t+1) = mji(t) + n[grad]h(t) 
  
            for iter = 1: numNeuroniosSaida
                M(iter,:) = M(iter,:) + (taxaAprendizado .* gradienteCmdSaida(iter,1) .* h)'; 

            end
%             erro retorpropagado sum((y'*erro*mji))
%           (y'*erro*mji)*h'
            gradienteCmdOculta = (gradienteCmdSaida' * M(:,2:numNeuroniosOcultos+1))'...
                .* h1.*(ones(size(h1,1),1)-h1); 

           
            for iter = 1: numNeuroniosOcultos
                W(iter,:) = W(iter,:) + (taxaAprendizado .* gradienteCmdOculta(iter,1) .* X(:,i))'; 
            end

        
        end
    end
    


end

