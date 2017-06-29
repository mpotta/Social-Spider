function average = returnMSE(spider, samples, numInput, numHidden, numOutput)

m = size(samples,1);
n = size(samples,2);

weightsIH = spider(1,1:numInput*numHidden);
weightsHO = spider(1,numInput*numHidden+1 : numInput*numHidden + numHidden*numOutput);
biasHidden = spider(1,numInput*numHidden + numHidden*numOutput + 1 : numInput*numHidden + numHidden*numOutput + numHidden);
biasOutput = spider(1,numInput*numHidden + numHidden*numOutput + numHidden+1 : numInput*numHidden + numHidden*numOutput + numHidden + numOutput);

hiddenLInput = zeros(1,numHidden);
hiddenLOutput = zeros(1,numHidden);

outputLInput = zeros(1,numOutput);
outputLOutput = zeros(1,numOutput);

MSE = 0; average = 0;

for i = 1:m
        % Compute Hidden Layer Node Values
        for k = 1:numHidden 
            hiddenLInput(1,k) = biasHidden(1,k);
                for j = 1:numInput
                    hiddenLInput(1,k) = hiddenLInput(1,k) + samples(i,j)*weightsIH(1,(j-1)*numHidden + k);
                end;
            hiddenLOutput(1,k) = logsig(hiddenLInput(1,k));
        end;
       
        % Compute Output Layer Node Values
        for k = 1:numOutput 
            outputLInput(1,k) = biasOutput(1,k);
                for j = 1:numHidden
                    outputLInput(1,k) = outputLInput(1,k) + hiddenLOutput(1,j)*weightsHO(1,(j-1)*numOutput + k);
                end;
            outputLOutput(1,k) = logsig(outputLInput(1,k));  
        end;
        
        % Compute DeltaOH
        for l = 1:numOutput
            MSE = MSE + (samples(i, numInput+l) - outputLOutput(1,l))^2;
        end;
        
        average = average + MSE/numOutput;
end;

average = average/m;
return 