%Script to compute mean log likelihood reference
clear all;
N = 4;

mll = zeros(1, length(N));
romObj = ROM_SPDE('');
Tf = romObj.trainingDataMatfile.Tf;
natNodes = true(romObj.fineScaleDomain.nNodes, 1);
natNodes(romObj.fineScaleDomain.essentialNodes) = false;
nNatNodes = sum(natNodes);
Tf = Tf(natNodes, :);
nSamples = romObj.nSets(1);
for j = 1:length(N)
    Nj = N(j)
    
    mll = 0;
    meanLogLikelihoodSq = 0;
    converged = false;
    i = 1;
    while(~converged)
        randSamples = randperm(nSamples);
        randSamples_params = randSamples(1:Nj);
        randSamples_samples = randSamples((Nj + 1):(2*Nj));
        Tftemp = Tf(:, randSamples_params);
        mu_data = mean(Tftemp, 2);
        var_data = var(Tftemp')';
        
        term1 = .5*log(geomean(var_data));
        term2 = .5*mean(mean((Tf(:, randSamples_samples) - mu_data).^2, 2)./var_data);
        
        mll = ((i - 1)/i)*mll + (1/i)*(term1 + term2);
        meanLogLikelihoodSq = ((i - 1)/i)*meanLogLikelihoodSq + (1/i)*(term1 + term2)^2;
        if(mod(i, 10000) == 0)
            i
            mll
            err = sqrt((meanLogLikelihoodSq - mll^2)/i);
            relErr = abs(err/mll)
            if((relErr < 1e-2 && i > 1e5) || i > 1e7)
                converged = true;
            end
        end
        i = i + 1;
    end
    mll = mll + .5*log(2*pi);
    mll = mll + .87; %remove this!
    meanLogLikelihood(j) = mll;
    meanLogLikelihoodErr(j) = err;
    
end

save('mll.mat')