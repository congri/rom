%Monte Carlo estimate of integral (2) in Stelios' notes using training data

Tffile = matfile('~/matlab/data/fineData/systemSize=512x512/correlated_binary/IsoSEcov/l=20_sigmafSq=1/volumeFraction=0.3/locond=1_hicond=100/BCcoeffs=[-5 3 0 0]/set1-samples=1024.mat');

Tftemp = Tffile.Tf(:, 1);
Tf_true_mean = zeros(size(Tftemp, 1), 1);
Tf_true_sq_mean = zeros(size(Tftemp, 1), 1);
% nSamples = size(Tffile.Tf, 2);
nSamples = 1024;
window = 512;
nWindows = ceil(nSamples/window);
tic
for i = 1:nWindows
    initial = 1 + (i - 1)*window;
    final = i*window;
    if final > nSamples
        final = nSamples;
    end
    
    Tftemp = Tffile.Tf(:, initial:final);
    Tf_mean_temp = mean(Tftemp, 2);
    Tf_sq_mean_temp = mean(Tftemp.^2, 2);
    clear Tftemp;
    Tf_true_mean = ((i - 1)/i)*Tf_true_mean + (1/i)*Tf_mean_temp;
    Tf_true_sq_mean = ((i - 1)/i)*Tf_true_sq_mean + (1/i)*Tf_sq_mean_temp;
end

Tf_true_var = Tf_true_sq_mean - Tf_true_mean.^2;
toc

sv = false;
if sv
%     savedir = '~/matlab/data/trueMC/';
savedir = './';
    if ~exist(savedir, 'dir')
        mkdir(savedir);
    end
    save(strcat(savedir, '', '_nSamples=', num2str(nSamples), '.mat'), 'Tf_true_mean', 'Tf_true_var')
end