classdef VariationalInference
    %Variational inference class
    
    properties
        log_empiricalDist                       %handle to [log empirical pdf, d/dx log empirical pdf]
        variationalDist                         %string specifying variational pdf
        dim                                     %dimension of empirical/variational pdf
        
        
        ELBOgrad                                %current estimate of evidence lower bound gradient
        ELBOgradErr
        ELBOgradParams                          %Holding parameters of estimation of ELBO gradient;
                                                %nSamples
                                                
    end
    
    
    
    
    properties(SetAccess = private)
        varDistParams                          %structure holding params of variational pdf
    end
    
    
    
    methods
        function VIobj = VariationalInference(log_empiricalDist, variationalDist, varDistParams, ELBOgradParams)
            %constructor
            disp(strcat('Using', ' ', variationalDist, ' as variational distribution'))
            VIobj.log_empiricalDist = log_empiricalDist;
            VIobj.variationalDist = variationalDist;
            VIobj.varDistParams = varDistParams;
            VIobj.ELBOgradParams = ELBOgradParams;
            
            if strcmp(VIobj.variationalDist, 'diagonalGauss')
                VIobj.dim = numel(VIobj.varDistParams.mu);
                VIobj.varDistParams.logSigmaMinus2 = -2*log(VIobj.varDistParams.sigma);
            elseif strcmp(VIobj.variationalDist, 'fullRankGauss')
                VIobj.dim = numel(VIobj.varDistParams.mu);
            else
                error('Unknown variational distribution')
            end
        end
        
        
        
        
        function [ELBOgrad, ELBOgradErr] = sampleELBOgrad(VIobj)
            %Estimation of gradient of evidence lower bound (ELBO)
            if strcmp(VIobj.variationalDist, 'diagonalGauss')
                d_mu_mean = 0;
                d_sigma_mean = 0;
                d_muSq_mean = 0;
                d_sigmaSq_mean = 0;
                
                for i = 1:VIobj.ELBOgradParams.nSamples
                    sample = normrnd(0, 1, 1, VIobj.dim);
                    
                    %transform standard normal sample to sample of VI distribution
                    variationalSample = VIobj.varDistParams.mu + VIobj.varDistParams.sigma.*sample;
                    
                    %Gradient w.r.t. dependent variable x
                    [~, d_log_empirical] = VIobj.log_empiricalDist(variationalSample);
                    d_log_empirical = d_log_empirical';
                    
                    %Mean gradient w.r.t. mu; d/dmu = (dX/dmu)*d/dX, X = mu + sigma*sample
                    % --> d/dmu = d/dX
                    d_mu_mean = (1/i)*((i - 1)*d_mu_mean + d_log_empirical);
                    
                    %Mean gradient w.r.t. d/d_sigma_k; d/dsigma = (dX/dsigma)*d/dX,
                    %X = mu + sigma*sample --> d/dsigma = sample*(d/dX)
                    %second term is due to gradient of variational dist (given analytically)
                    d_sigma_mean = (1/i)*((i - 1)*d_sigma_mean +...
                        (d_log_empirical.*sample + 1./VIobj.varDistParams.sigma));
                    
                    %Might be untrue as gradient of variational distribution is missing
                    %w.r.t. mu
                    d_muSq_mean = (1/i)*((i - 1)*d_muSq_mean + (d_log_empirical).^2);
                    %w.r.t. d/d_sigma_k
                    d_sigmaSq_mean = (1/i)*((i - 1)*d_sigmaSq_mean +...
                        (d_log_empirical.*sample + 1./VIobj.varDistParams.sigma).^2);
                end

                %Transformation d/dsigma --> d/dlog(sigma^-2)
                d_logSigma_Minus2mean = -.5*(d_sigma_mean.*VIobj.varDistParams.sigma);
                ELBOgrad = [d_mu_mean d_logSigma_Minus2mean];
                
                d_muErr = sqrt(abs(d_muSq_mean - d_mu_mean.^2))/sqrt(VIobj.ELBOgradParams.nSamples);
                %error w.r.t. d/d_sigma_k
                d_sigmaErr = sqrt(.25*(VIobj.varDistParams.sigma.^2).*d_sigmaSq_mean...
                    - d_logSigma_Minus2mean.^2)/sqrt(VIobj.ELBOgradParams.nSamples);
                ELBOgradErr = [d_muErr d_sigmaErr];
                
            elseif strcmp(VIobj.variationalDist, 'fullRankGauss')
                %Sigma = L*L^T
                d_mu_mean = 0;
                d_L_mean = 0;
                d_muSq_mean = 0;
                d_LSq_mean = 0;
                
                for i = 1:VIobj.ELBOgradParams.nSamples
                    sample = normrnd(0, 1, 1, VIobj.dim);
                    
                    %transform standard normal sample to sample of VI distribution
                    variationalSample = VIobj.varDistParams.mu + sample*VIobj.varDistParams.LT;
                    
                    %Gradient w.r.t. dependent variable x
                    [~, d_log_empirical] = VIobj.log_empiricalDist(variationalSample);
                    d_log_empirical = d_log_empirical';
                    
                    %Mean gradient w.r.t. mu; d/dmu = (dX/dmu)*d/dX, X = mu + sigma*sample
                    % --> d/dmu = d/dX
                    d_mu_mean = (1/i)*((i - 1)*d_mu_mean + d_log_empirical);
                    
                    %Mean gradient w.r.t. d/d_sigma_k; d/dsigma = (dX/dsigma)*d/dX,
                    %X = mu + sigma*sample --> d/dsigma = sample*(d/dX)
                    %second term is due to gradient of variational dist (given analytically)
                    d_L_mean = (1/i)*((i - 1)*d_L_mean +...
                        .5*(d_log_empirical'*sample + VIobj.varDistParams.LInv)...
                        + .5*(d_log_empirical'*sample + VIobj.varDistParams.LInv)');
                    
                    %Might be untrue as gradient of variational distribution is missing
                    %w.r.t. mu
                    d_muSq_mean = (1/i)*((i - 1)*d_muSq_mean + (d_log_empirical).^2);
                    %w.r.t. d/d_sigma_k
                    d_LSq_mean = (1/i)*((i - 1)*d_LSq_mean +...
                        (d_log_empirical'*sample + VIobj.varDistParams.LInv).^2);
                end
                
                %We want to fix upper triangular elements to 0
                d_L_mean = tril(d_L_mean);
                d_LSq_mean = tril(d_L_mean);

                %Transformation d/dsigma --> d/dlog(sigma^-2)
                ELBOgrad = [d_mu_mean d_L_mean(:)'];
                
                d_muErr = sqrt(abs(d_muSq_mean - d_mu_mean.^2))/sqrt(VIobj.ELBOgradParams.nSamples);
                %error w.r.t. d/d_L
                d_LErr = sqrt(d_LSq_mean - d_L_mean.^2)/sqrt(VIobj.ELBOgradParams.nSamples);
                ELBOgradErr = [d_muErr d_LErr(:)'];
            else
                error('Unknown variational distribution')
            end
        end
        
        
        
        
        function VIobj = setVarDistParams(VIobj, params)
            if strcmp(VIobj.variationalDist, 'diagonalGauss')
                VIobj.varDistParams.mu = params.mu;
                VIobj.varDistParams.sigma = params.sigma;
                VIobj.varDistParams.logSigmaMinus2 = -2*log(params.sigma);
            elseif strcmp(VIobj.variationalDist, 'fullRankGauss')
                %Keep in mind that all forms of parameters need to be in struct params here,
                %i.e. Sigma, mu, L, LT, LMinusT
                VIobj.varDistParams = params;
            else
                error('Unknown variational distribution')
            end
        end
    end 
end

