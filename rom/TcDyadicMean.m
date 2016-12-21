function [Tc_dyadic_mean_mean] = TcDyadicMean(Tc_samples, nData, MCMC)
%Computes (1/N) sum_i <Tc^T*Tc>_qi

for i = 1:nData
    for m = 1:MCMC(i).nSamples
        Tc_dyadic(:, :, m) = Tc_samples(:, m, i)*Tc_samples(:, m, i)';
    end
    %mean of dyadic product under q_i
    Tc_dyadic_mean(:, :, i) = mean(Tc_dyadic, 3);
end
%Mean along data samples
Tc_dyadic_mean_mean = mean(Tc_dyadic_mean, 3);


end

