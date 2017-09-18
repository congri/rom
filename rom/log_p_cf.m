function [log_p, d_log_p, Tc] = log_p_cf(Tf_i_minus_mu, domainc, Xi, theta_cf, condTransOpts)
%Coarse-to-fine map
%ignore constant prefactor
%log_p = -.5*logdet(S, 'chol') - .5*(Tf - mu)'*(S\(Tf - mu));
%diagonal S
if domainc.useConvection
    %We are in convection-diffusion mode here
    conductivity = conductivityBackTransform(Xi(1:domainc.nEl), condTransOpts);
    %is this correctly reshaped?
    convectionField = reshape(Xi((domainc.nEl + 1):end), domainc.nEl, 2)';
else
    %only diffusion
    conductivity = conductivityBackTransform(Xi, condTransOpts);
end
D = zeros(2, 2, domainc.nEl);
%Conductivity matrix D, only consider isotropic materials here
for j = 1:domainc.nEl
    D(:, :, j) =  conductivity(j)*eye(2);
end
if domainc.useConvection
    FEMout = heat2d(domainc, D, convectionField);
else
    FEMout = heat2d(domainc, D);
end

Tc = FEMout.Tff';
Tc = Tc(:);
Tf_i_minus_mu_minus_WTc = Tf_i_minus_mu - theta_cf.W*Tc;
%only for diagonal S!
log_p = -.5*(theta_cf.sumLogS + (Tf_i_minus_mu_minus_WTc)'*(theta_cf.Sinv_vec.*(Tf_i_minus_mu_minus_WTc)));

if nargout > 1
    %Gradient of FEM equation system w.r.t. conductivities
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %TO BE DONE: DERIVATIVE W.R.T. MATRIX COMPONENTS FOR ANISOTROPY
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if domainc.useConvection
        d_r = FEMgrad(FEMout, domainc, conductivity, convectionField);
    else
        d_r = FEMgrad(FEMout, domainc, conductivity);
    end
    d_rx = d_r;
    if strcmp(condTransOpts.type, 'log')
        %We need gradient of r w.r.t. log conductivities X, multiply each row with resp. conductivity
        d_rx(1:domainc.nEl, :) = diag(conductivity)*d_r(1:domainc.nEl, :);
    elseif strcmp(condTransOpts.type, 'logit')
        %We need gradient w.r.t. x, where x is - log((lambda_up - lambda_lo)/(lambda - lambda_lo) - 1)
        X = conductivityTransform(conductivity, condTransOpts);
        dLambda_dX = (condTransOpts.limits(2) - condTransOpts.limits(1))./(exp(X) + 2 + exp(-X));
        d_rx(1:domainc.nEl, :) = diag(dLambda_dX)*d_r(1:domainc.nEl, :);
    elseif strcmp(condTransOpts.type, 'log_lower_bound')
        %transformation is X = log(Lambda - lambda_lo)
        dLambda_dX = conductivity - condTransOpts.limits(1);
        d_rx(1:domainc.nEl, :) = diag(dLambda_dX)*d_r(1:domainc.nEl, :);
    elseif strcmp(condTransOpts.type, 'square')
        %We need gradient of r w.r.t. sqrt conductivities X. d/dX = 2X d/dlambda
        d_rx(1:domainc.nEl, :) = diag(2*Xi(1:domainc.nEl))*d_r(1:domainc.nEl, :);
    else
        error('Unknown conductivity transformation')
    end
    adjoints = get_adjoints(FEMout.globalStiffness, theta_cf, domainc, Tf_i_minus_mu_minus_WTc);
    d_log_p = - d_rx*adjoints;

    
    %Finite difference gradient check
    FDcheck = false;
    if FDcheck
        disp('Gradient check log p_cf')
        d = 1e-8;
        if domainc.useConvection
            FDgrad = zeros(3*domainc.nEl, 1);
        else
            FDgrad = zeros(domainc.nEl, 1);
        end
        for e = 1:domainc.nEl
            conductivityFD = conductivity;
            conductivityFD(e) = conductivityFD(e) + d;
            
            DFD = zeros(2, 2, domainc.nEl);
            for j = 1:domainc.nEl
                DFD(:, :, j) =  conductivityFD(j)*eye(2);
            end
            FEMoutFD = heat2d(domainc, DFD, convectionField);
            checkLocalStiffness = false;
            if checkLocalStiffness
                k = FEMout.diffusionStiffness(:, :, e);
                kFD = FEMoutFD.diffusionStiffness(:, :, e);
                d_k = FEMout.diffusionStiffness(:, :, e)/conductivity(e);
                d_kFD = (FEMoutFD.diffusionStiffness(:, :, e) - FEMout.diffusionStiffness(:, :, e))/d;
                relgrad_k = d_k./d_kFD
                pause
            end
            TcFD = FEMoutFD.Tff';
            TcFD = TcFD(:);
            
            WTcFD = theta_cf.W*TcFD;
            log_pFD = -.5*(theta_cf.sumLogS + (Tf_i_minus_mu - WTcFD)'*(theta_cf.Sinv_vec.*(Tf_i_minus_mu - WTcFD)));
            if strcmp(condTransOpts.type, 'log')
                FDgrad(e) = conductivity(e)*(log_pFD - log_p)/d;
            elseif strcmp(condTransOpts.type, 'logit')
                FDgrad(e) = dLambda_dX(e)*(log_pFD - log_p)/d;
            elseif strcmp(condTransOpts.type, 'log_lower_bound')
                FDgrad(e) = dLambda_dX(e)*(log_pFD - log_p)/d;
            else
                error('Unknown conductivity transformation')
            end
        end
        if domainc.useConvection
            for xy = 1:2    %x or y component of convection field
                for e = 1:domainc.nEl
                    convectionFieldFD = convectionField;
                    convectionFieldFD(xy, e) = convectionFieldFD(xy, e) + d;
                    FEMoutFD = heat2d(domainc, D, convectionFieldFD);
                    TcFD = FEMoutFD.Tff';
                    TcFD = TcFD(:);
                    WTcFD = theta_cf.W*TcFD;
                    log_pFD = -.5*(theta_cf.sumLogS + (Tf_i_minus_mu - WTcFD)'*...
                        (theta_cf.Sinv_vec.*(Tf_i_minus_mu - WTcFD)));
                    FDgrad(xy*domainc.nEl + e) = (log_pFD - log_p)/d;
                end
            end
        end
        relgrad = FDgrad./d_log_p
        plot(1:numel(FDgrad), FDgrad, 1:numel(FDgrad), d_log_p)
        axis square
        drawnow
        pause(1)
        if(norm(relgrad - 1) > 1e-1)
            convectionField
            log_p
            log_pFD
            d_log_p
            FDgrad
            diff = log_pFD - log_p
        end
    end %FD gradient check
    
    
end

end

