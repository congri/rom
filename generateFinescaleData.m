function [] = generateFinescaleData()   %Void function to keep workspace clean
%Function to generate and save finescale data
%CHANGE JOBFILE IF YOU CHANGE LINE NUMBERS!
tic;
addpath('./params')
addpath('./heatFEM')
addpath('./rom')
addpath('./genConductivity')
addpath('./computation')

%% Temperature field and gradient generating the boundary conditions
boundaryConditions;

%% Generate finescale domain
nf = 128;       %Finescale mesh size, should be 2^n
disp('Generate finescale domain...')
domainf = Domain(nf, nf);
domainf = setBoundaries(domainf, 2:(4*nf), Tb, qb);       %Only fix lower left corner as essential node
disp('done')
toc

%% Finescale data params
disp('Setting up finescale data parameters...')
FD = FinescaleData(1, 100);
FD.nSets = 2;
FD.nSamples = [4 2];
FD.distributionType = 'correlated_binary';
FD.distributionParams = {.3 [5 5] 1};

if strcmp(FD.distributionType, 'correlated_binary')
    %Folder where finescale data is saved
    fineDataPath = '~/matlab/data/fineData/';
    %System size
    fineDataPath = strcat(fineDataPath, 'systemSize=', num2str(domainf.nElX), 'x', num2str(domainf.nElY), '/');
    %Type of conductivity distribution
    fineDataPath = strcat(fineDataPath, FD.distributionType, '/',...
        'IsoSEcov/', 'l=', num2str(FD.distributionParams{2}(1)),...
        '_sigmafSq=', num2str(FD.distributionParams{3}), '/volumeFraction=',...
        num2str(FD.distributionParams{1}), '/', 'locond=', num2str(FD.loCond),...
        '_upcond=', num2str(FD.upCond), '/', 'BCcoeffs=', mat2str(boundaryCoeffs), '/');
else
    error('No savepath set for other distribution than correlated_binary')
end

if ~exist(fineDataPath, 'dir')
    mkdir(fineDataPath);
end

%Generate finescale conductivity samples and solve FEM
for i = 1:FD.nSets
    filename = strcat(fineDataPath, 'set', num2str(i), '-samples=', num2str(FD.nSamples(i)));
    FD.solveFEM(domainf, i, filename);
end

%save params
save(strcat(fineDataPath, 'params.mat'), 'domainf', 'FD', 'boundaryCoeffs', 'Tb', 'qb', 'nf');

end