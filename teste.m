function [ count ] = teste( W1, W2, X_teste, Y_teste )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here()
     X = X_teste';
    [n,m] = size(X);
    biasCamadaOculta = ones(1,m); % vetor coluna
    X = [biasCamadaOculta;X];
    count = 0;
    for i = 1: m
       
        camadaOcultaUi = W1*X(:,i); 

        h = logsig(camadaOcultaUi); 
        biasCamadaSaida = 1;
        h = [biasCamadaSaida; h]; 
        
        camadaSaidaUi = W2*h;  

        Y = logsig(camadaSaidaUi);
        Y = binariza(Y);
        D = Y_teste(i,:)';
        
        if isequal(Y, D)
            count = count + 1;
        end
        
    end
    count = 100*(sum(count))/m;
end

