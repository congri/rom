%Script that loads finescale data

%Which data to load? We will get error if data doesn't exist
nf = 256;
loCond = 1;
upCond = 10;
nSamples = 1024;
corrlength = '5';
volfrac = '0.3';  %high conducting phase volume fraction
sigma_f2 = '1';
cond_distribution = 'correlated_binary';
bc = '[-50 164 112 -30]';


%Folder where finescale data is saved
fineDataPath = '~/matlab/data/fineData/';
%System size
fineDataPath = strcat(fineDataPath, 'systemSize=', num2str(nf), 'x', num2str(nf), '/');
%Type of conductivity distribution
fineDataPath = strcat(fineDataPath, cond_distribution, '/', 'IsoSEcov/', 'l=',...
    corrlength, '_sigmafSq=', sigma_f2, '/volumeFraction=',...
    volfrac, '/', 'locond=', num2str(loCond),...
    '_upcond=', num2str(upCond), '/', 'BCcoeffs=', bc, '/');
clear nf corrlength volfrac sigma_f2 cond_distribution bc;


%Name of training data file
trainFileName = strcat('set1-samples=', num2str(nSamples), '.mat');
%Name of parameter file
paramFileName = strcat('params','.mat');

%load data params
load(strcat(fineDataPath, paramFileName));

%load finescale temperatures partially
Tffile = matfile(strcat(fineDataPath, trainFileName));