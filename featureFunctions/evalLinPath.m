%Computes lineal path function of microstructure
clear all
%Which data to load? We will get error if data doesn't exist
nf = 512;
contrast = 100;
nSamples = 200;
%Folder where finescale data is saved
fineDataPath = strcat('/home/constantin/matlab/data/fineData/');
%Name of training data file
trainFileName = strcat('train_', 'nf=', num2str(nf), '_contrast=', num2str(contrast), '_samples=',...
    num2str(nSamples), '_corrlength=', num2str(30));
%load finescale temperatures partially
Tffile = matfile(strcat(fineDataPath, trainFileName));

fineData.lo = 1;
fineData.up = contrast;

sample = 5;
cond = Tffile.cond(sample, :);        %Finescale conductivities - load partially to save memory

nElX = sqrt(length(cond));
maxPathLength = 200;
L{1} = zeros(1, maxPathLength);
L{2} = L{1};
L{3} = L{1};
L{4} = L{1};
dir{1} = 'x';
dir{2} = 'y';
dir{3} = 'x';
dir{4} = 'y';
parPoolInit(4);
parfor k = 1:4
    for i = 1:maxPathLength
        if k < 3
        L{k}(i) = linealPath(cond, i, dir{k}, 1, fineData, [1 1], [512 512]);
        else
        L{k}(i) = twoPointCorrelation(cond, i, dir{k}, 1, fineData, [1 1], [512 512])
        end
    end
end

figure
subplot(1,3,1)
p1 = plot(1:maxPathLength, L{1}, 1:maxPathLength, L{2});
p1(1).LineWidth = 3;
p1(2).LineWidth = 3;
legend('x-direction', 'y-direction')
axis square
xlim([1 maxPathLength])
xlabel('pixels distance')
ylabel('Lineal path')
subplot(1,3,2)
p2 = plot(1:maxPathLength, L{3}, 1:maxPathLength, L{4});
p2(1).LineWidth = 3;
p2(2).LineWidth = 3;
legend('x-direction', 'y-direction')
axis square
xlim([1 maxPathLength])
xlabel('pixels distance')
ylabel('2-point corr')
cond = reshape(cond, 512, 512);
subplot(1,3,3)
pcolor(cond)
axis square
xlabel('x')
ylabel('y')

%% Least squares fit to theoretical curve
Lmean = .5*(L{1} + L{2});
x = 1:maxPathLength;
f = fit(x', Lmean', 'exp1')
figure
plot(x, Lmean, x, f.a*exp(f.b*x))
