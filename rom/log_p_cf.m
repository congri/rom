function [log_p, d_log_p, Tc] = log_p_cf(Tf_i_minus_mu, domainc, Xi, theta_cf, condTransOpts)
%Coarse-to-fine map
%ignore constant prefactor
%log_p = -.5*logdet(S, 'chol') - .5*(Tf - mu)'*(S\(Tf - mu));
%diagonal S
useConvection = (numel(Xi) > domainc.nEl);
if useConvection
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
if useConvection
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
    
    d_r = FEMgrad(FEMout, domainc, conductivity);
    if strcmp(condTransOpts.type, 'log')
        %We need gradient of r w.r.t. log conductivities X, multiply each row with resp. conductivity
        d_rx = diag(conductivity)*d_r;
    elseif strcmp(condTransOpts.type, 'logit')
        %We need gradient w.r.t. x, where x is - log((lambda_up - lambda_lo)/(lambda - lambda_lo) - 1)
        X = conductivityTransform(conductivity, condTransOpts);
        dLambda_dX = (condTransOpts.limits(2) - condTransOpts.limits(1))./(exp(X) + 2 + exp(-X));
        d_rx = diag(dLambda_dX)*d_r;
    elseif strcmp(condTransOpts.type, 'log_lower_bound')
        %transformation is X = log(Lambda - lambda_lo)
        dLambda_dX = conductivity - condTransOpts.limits(1);
        d_rx = diag(dLambda_dX)*d_r;
    else
        error('Unknown conductivity transformation')
    end
    adjoints = get_adjoints(FEMout.globalStiffness, theta_cf, domainc, Tf_i_minus_mu_minus_WTc);
    d_log_p = - d_rx*adjoints;

    
    %Finite difference gradient check
    FDcheck = false;
    if FDcheck
        disp('Gradient check log p_cf')
        d = 1e-7;
        FDgrad = zeros(domainc.nEl, 1);
        for e = 1:domainc.nEl
            conductivityFD = conductivity;
            conductivityFD(e) = conductivityFD(e) + d;
            
            DFD = zeros(2, 2, domainc.nEl);
            for j = 1:domainc.nEl
                DFD(:, :, j) =  conductivityFD(j)*eye(2);
            end
            FEMoutFD = heat2d(domainc, DFD);
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
        relgrad = FDgrad./d_log_p
        if(norm(relgrad - 1) > 1e-1)
            log_p
            log_pFD
            d_log_p
            FDgrad
            diff = log_pFD - log_p
            pause
        end
%         d_r
%         d_rx
%         adjoints
%         d_log_p
%         FDgrad
%         conductivity
% 
%         conductivityFDcheck = conductivity + .001*(FDgrad./conductivity);
%         DFDcheck = zeros(2, 2, domainc.nEl);
%         for j = 1:domainc.nEl
%             DFDcheck(:, :, j) =  conductivityFDcheck(j)*eye(2);
%         end
%         FEMoutFDcheck = heat2d(domainc, physicalc, control, DFDcheck);
%         TcFDcheck = FEMoutFDcheck.Tff';
%         TcFDcheck = TcFDcheck(:);
%         WTcFDcheck = W*TcFDcheck;
%         log_pFDcheck = -.5*sum(log(diag(S))) - .5*(Tf_i - WTcFDcheck)'*(Sinv*(Tf_i - WTcFDcheck));
%         checkFD = log_pFDcheck - log_p
%         
%         conductivitycheck = conductivity + .001*(d_log_p./conductivity);
%         Dcheck = zeros(2, 2, domainc.nEl);
%         for j = 1:domainc.nEl
%             Dcheck(:, :, j) =  conductivitycheck(j)*eye(2);
%         end
%         FEMoutcheck = heat2d(domainc, physicalc, control, Dcheck);
%         Tccheck = FEMoutcheck.Tff';
%         Tccheck = Tccheck(:);
%         WTccheck = W*Tccheck;
%         log_pcheck = -.5*sum(log(diag(S))) - .5*(Tf_i - WTccheck)'*(Sinv*(Tf_i - WTccheck));
%         check = log_pcheck - log_p
    end %FD gradient check
    
    
end

end

