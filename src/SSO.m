epoch = 250;

numInput =22;
numHidden = 10;
numOutput = 2;

n = numInput*numHidden + numHidden*numOutput + numHidden + numOutput;

pLow = -1;
pHigh = 1;

alpha = rand;
beta = rand;
delta = rand;

PF = 0.5;

% STEP1: Determine Population Size
N = 50;
Nf = floor((0.9 - rand*0.25)*N);
Nm = N - Nf;
fit=zeros(epoch,N);

% STEP2: Initialize Initial Population
spiders = zeros(N,n);
radius = 0;
fitness = zeros(1,N);
weight = zeros(1,N);

for j = 1:n
    radius = radius + ((pHigh - pLow)/(2*N));
end;

for i = 1:Nf
        spiders(i,:) = pLow + rand*(pHigh - pLow);
end;

for i = 1:Nm
        spiders(Nf+i,:) = pLow + rand*(pHigh - pLow);
end;

%STEP3: Calculate Spider Weight
for i = 1:N
    fitness(1,i) = 1/(1 + returnMSE(spiders(i,:),samples,numInput,numHidden,numOutput));
end;

[val, m] = min(fitness(1,:));
worst = val;
worstIndex = m;
[val, m] = max(fitness(1,:));
best = val;

for i = 1:N
    weight(1,i) = (fitness(1,i) - worst)/(best - worst);
end;


% Training
for e=1:epoch   
   
    VibC = 0; disC = 0;
    VibB = 0; disB = 0;
    VibF = 0; disF = 0;
    
%STEP4: Move Female Spiders
for i = 1:Nf
    Dist = pdist2(spiders, spiders(i,:));
    s = size(Dist,1);
    Dist = [Dist'; 1:s;];
    Dist = sortrows(Dist')';
    
    for j = 1:s
        if(weight(1,Dist(2,j))>weight(1,i)) 
            disC = Dist(1,j)*Dist(1,j); 
            VibC = weight(1,Dist(2,j))*exp(-disC);
            break;
        end;
    end;
    
    [val, m] = max(fitness(1,:));
    bestIndex = m;
    
    disB = pdist2(spiders(bestIndex,:),spiders(i,:))*pdist2(spiders(bestIndex,:),spiders(i,:));
    VibB = weight(1,bestIndex)*exp(-disB);
        
    if(rand < PF)
        spiders(i,:) = spiders(i,:) + alpha*VibC*(spiders(Dist(2,j),:) - spiders(i,:)) + beta*VibB*(spiders(bestIndex,:) - spiders(i,:)) + delta*(rand - 0.5);
    else
        spiders(i,:) = spiders(i,:) - alpha*VibC*(spiders(Dist(2,j),:) - spiders(i,:)) - beta*VibB*(spiders(bestIndex,:) - spiders(i,:)) + delta*(rand - 0.5);
    end;  
    
    Dist = 0; s = 0;
end;

%STEP5: Move Male Spiders
medianWeight = median(sort(weight(1,Nf+1:N)));
avgMale = zeros(1,n); sum = 0;

for j = 1:n
    for k = 1:Nm
       avgMale(1,j) = avgMale(1,j) + weight(1,Nf+k)*spiders(Nf+k,j);
    end;
end; 

for p = 1:Nm
     sum = sum + weight(1,Nf+p);
end;

avgMale(1,:) = avgMale(1,:)/sum;

for i = 1:Nm           
    if(weight(1,Nf+i) > medianWeight) 
        Dist = pdist2(spiders(1:Nf,:), spiders(Nf+i,:));
        s = size(Dist,1);
        Dist = [Dist'; 1:s];
        Dist = sortrows(Dist')';

        disF = Dist(1,1)*Dist(1,1);
        VibF = weight(1,Dist(2,1))*exp(-disF);
        
        spiders(Nf+i,:) = spiders(Nf+i,:) + alpha*VibF*(spiders(Dist(2,1),:) - spiders(Nf+i,:)) + delta*(rand-0.5);
    else
        spiders(Nf+i,:) = spiders(Nf+i,:) + alpha*(avgMale(1,:) - spiders(Nf+i,:));
    end;
        
    Dist = 0; s = 0;
end;



%STEP6: Mating Process

for i = 1:Nm
    if(weight(1,Nf+i) > medianWeight)
        
        Dist = pdist2(spiders(1:Nf,:), spiders(Nf+i,:));
        s = size(Dist,1);
        Dist = [Dist'; 1:s];
        Dist = sortrows(Dist')';
        m =0; Eg = zeros(1,1);
        for k = 1:s
            if(Dist(1,k)<radius && m==0)
                Eg(1,1) = k; m=m+1; 
            elseif (Dist(1,k)<radius) 
                Eg = [Eg,k]; m = m+1;
            end;
        end;
        
        if(any(Eg) ~= 0)  % No mating
        
        Eg = [Eg,Nf+i];
        summation = 0;
        
        Ps = zeros(1,size(Eg,2));
        for l = 1: size(Eg,2)
            summation = summation + weight(1,Eg(1,l));
        end;
        for l = 1: size(Eg,2)
            Ps(1,l) = weight(1,Eg(1,l))/summation;
        end;
        
        PsCuml = zeros(1,size(Eg,2));
        PsCuml(1,1) = Ps(1,1);
        for p = 2: size(Eg,2)
            PsCuml(1,p) = PsCuml(1,p-1) + Ps(1,p);
        end;
        
        matingMatrix = [Eg; Ps; PsCuml; zeros(1,size(Eg,2))];
        newSpider = zeros(1,n);
        for j = 1:n
                matingMatrix(4,l) = rand;
                for k = 1:size(Eg,2)
                    if(matingMatrix(4,l)<matingMatrix(3,k))
                        selectedSpiderIndex = matingMatrix(1,k);
                        break;
                    end;
                end;
                newSpider(1,j) = spiders(selectedSpiderIndex,j);
        end;
        newSpiderFitness = 1/(1+returnMSE(newSpider(1,:),samples,numInput,numHidden,numOutput));
        
        [val, m] = min(fitness(1,:));
        worst = val;
        worstIndex = m;
        
        if(newSpiderFitness>worst) 
           fitness(1,worstIndex) = newSpiderFitness; 
           spiders(worstIndex,:) = newSpider(1,:);
        end;
        end;
    end;
end;
end;


%Testing
[val, m] = max(fitness(1,:));
bestIndex = m;

[finalOutput,accuracy] = returnZ2(spiders(bestIndex,:),samples,numInput,numHidden,numOutput);
        
    
    
    
    



