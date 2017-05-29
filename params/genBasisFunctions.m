%% Some predefined basis functions for linear model p_c

phi = {};
conductivities = [romObj.lowerConductivity romObj.upperConductivity];

%constant bias
phi{end + 1} = @(lambda) 1;

% generalized means
% min_z = 1;
% max_z = 1;
% z_incr = .5;
% z = min_z:z_incr:max_z;
% for zz = z
%     phi{end + 1} = @(lambda) generalizedMean(lambda(:), zz);
% end
%log of generalized means
% for zz = z
%     phi{end + 1} = @(lambda) log(generalizedMean(lambda, zz));
% end
% 
% %% Lineal path
% % dLinPathMax = 31;
% % dLinPathMin = 0;
% % dLinPathIncr = 4;
nElc = [romObj.coarseScaleDomain.nElX romObj.coarseScaleDomain.nElY];
nElf = [romObj.fineScaleDomain.nElX romObj.fineScaleDomain.nElY];
%Take 1px/ total number of px as cutoff
log_cutoff = 10*((nElf(1)*nElf(2))/(nElc(1)*nElc(2)))^(-1);
% 
% % 
% %Phase 1, isotropic
% % dLinPathMax = 31;
% % dLinPathMin = 0;
% % dLinPathIncr = 4;
% % for d = dLinPathMin:dLinPathIncr:dLinPathMax
% %     phi{end + 1} = @(lambda) linealPath(lambda, d, 'x', 1, conductivities, nElc, nElf) +...
% %         linealPath(lambda, d, 'y', 1, conductivities, nElc, nElf);
% % end
% 
% % dLinPathMax = 16;
% % dLinPathMin = 2;
% % dLinPathIncr = 2;
% % %Phase 2, isotropic
% % for d = dLinPathMin:dLinPathIncr:dLinPathMax
% %     phi{end + 1} = @(lambda) linealPath(lambda, d, 'x', 2, conductivities, nElc, nElf) +...
% %         linealPath(lambda, d, 'y', 2, conductivities, nElc, nElf);
% % end
% 
% % volume fraction
% phi{end + 1} = @(lambda) linealPath(lambda, 0, 'x', 1, conductivities, nElc, nElf);
% 
% % % Phase 1
% dLinPathMax = 30;
% dLinPathMin = 2;
% dLinPathIncr = 2;
% for d = dLinPathMin:dLinPathIncr:dLinPathMax
%    phi{end + 1} = @(lambda) linealPath(lambda, d, 'x', 1, conductivities, nElc, nElf);
% end
% 
% dLinPathMax = 30;
% dLinPathMin = 2;
% dLinPathIncr = 2;
% for d = dLinPathMin:dLinPathIncr:dLinPathMax
%    phi{end + 1} = @(lambda) linealPath(lambda, d, 'y', 1, conductivities, nElc, nElf);
% end
% % 
% % 
% % %log Phase 1
% % % dLinPathMax = 31;
% % % dLinPathMin = 0;
% % % dLinPathIncr = 4;
% % % for d = dLinPathMin:dLinPathIncr:dLinPathMax
% % %     phi{end + 1} = @(lambda) log(linealPath(lambda, d, 'x', 1, conductivities, nElc, nElf));
% % % end
% % % 
% % % dLinPathMax = 31;
% % % dLinPathMin = 4;
% % % dLinPathIncr = 4;
% % % for d = dLinPathMin:dLinPathIncr:dLinPathMax
% % %     phi{end + 1} = @(lambda) log(linealPath(lambda, d, 'y', 1, conductivities, nElc, nElf));
% % % end
% % 
% %Phase 2
% dLinPathMax = 30;
% dLinPathMin = 2;
% dLinPathIncr = 2;
% for d = dLinPathMin:dLinPathIncr:dLinPathMax
%    phi{end + 1} = @(lambda) linealPath(lambda, d, 'x', 2, conductivities, nElc, nElf);
% end
% dLinPathMax = 30;
% dLinPathMin = 2;
% dLinPathIncr = 2;
% for d = dLinPathMin:dLinPathIncr:dLinPathMax
%    phi{end + 1} = @(lambda) linealPath(lambda, d, 'y', 2, conductivities, nElc, nElf);
% end
% 
% 
% % %log Phase 2
% % dLinPathMax = 8;
% % dLinPathMin = 0;
% % dLinPathIncr = 1;
% % for d = dLinPathMin:dLinPathIncr:dLinPathMax
% %     phi{end + 1} = @(lambda) log(linealPath(lambda, d, 'x', 2, conductivities, nElc, nElf));
% % end
% % dLinPathMax = 8;
% % dLinPathMin = 1;
% % dLinPathIncr = 1;
% % for d = dLinPathMin:dLinPathIncr:dLinPathMax
% %     phi{end + 1} = @(lambda) log(linealPath(lambda, d, 'y', 2, conductivities, nElc, nElf));
% % end
% 
% %% 2-point corr
% % d2pointCorrMax = 31;
% % d2pointCorrMin = 4;
% % d2pointCorrIncr = 4;
% % %Phase 1
% % for d = d2pointCorrMin:d2pointCorrIncr:d2pointCorrMax
% %     %distinct in x and y direction
% %     phi{end + 1} = @(lambda) twoPointCorrelation(lambda, d, 'x', 1, conductivities, nElc, nElf);
% % end
% % d2pointCorrMax = 31;
% % d2pointCorrMin = 4;
% % d2pointCorrIncr = 4;
% % for d = d2pointCorrMin:d2pointCorrIncr:d2pointCorrMax
% %     %distinct in x and y direction
% %     phi{end + 1} = @(lambda) twoPointCorrelation(lambda, d, 'y', 1, conductivities, nElc, nElf);
% % end
% %log
% % for d = d2pointCorrMin:d2pointCorrIncr:d2pointCorrMax
% %     %distinct in x and y direction
% %     phi{end + 1} = @(lambda) log(twoPointCorrelation(lambda, d, 'x', 1, conductivities, nElc, nElf));
% % end
% % for d = d2pointCorrMin:d2pointCorrIncr:d2pointCorrMax
% %     %distinct in x and y direction
% %     phi{end + 1} = @(lambda) log(twoPointCorrelation(lambda, d, 'y', 1, conductivities, nElc, nElf));
% % end
% 
% %Phase 2
% % d2pointCorrMax = 16;
% % d2pointCorrMin = 2;
% % d2pointCorrIncr = 2;
% % for d = d2pointCorrMin:d2pointCorrIncr:d2pointCorrMax
% %     %distinct in x and y direction
% %     phi{end + 1} = @(lambda) twoPointCorrelation(lambda, d, 'x', 2, conductivities, nElc, nElf);
% % end
% % d2pointCorrMax = 16;
% % d2pointCorrMin = 2;
% % d2pointCorrIncr = 2;
% % for d = d2pointCorrMin:d2pointCorrIncr:d2pointCorrMax
% %     %distinct in x and y direction
% %     phi{end + 1} = @(lambda) twoPointCorrelation(lambda, d, 'y', 2, conductivities, nElc, nElf);
% % end
% %log
% % for d = d2pointCorrMin:d2pointCorrIncr:d2pointCorrMax
% %     %distinct in x and y direction
% %     phi{end + 1} = @(lambda) log(twoPointCorrelation(lambda, d, 'x', 2, conductivities, nElc, nElf));
% % end
% % for d = d2pointCorrMin:d2pointCorrIncr:d2pointCorrMax
% %     %distinct in x and y direction
% %     phi{end + 1} = @(lambda) log(twoPointCorrelation(lambda, d, 'y', 2, conductivities, nElc, nElf));
% % end
% 
% 
% 
% 
% %% nPixelCross
% %min
% phi{end + 1} = @(lambda) nPixelCross(lambda, 'x', 1, conductivities, nElc, nElf, 'min');
% phi{end + 1} = @(lambda) nPixelCross(lambda, 'y', 1, conductivities, nElc, nElf, 'min');
% phi{end + 1} = @(lambda) nPixelCross(lambda, 'x', 2, conductivities, nElc, nElf, 'min');
% phi{end + 1} = @(lambda) nPixelCross(lambda, 'y', 2, conductivities, nElc, nElf, 'min');
% 
% %max
% phi{end + 1} = @(lambda) nPixelCross(lambda, 'x', 1, conductivities, nElc, nElf, 'max');
% phi{end + 1} = @(lambda) nPixelCross(lambda, 'y', 1, conductivities, nElc, nElf, 'max');
% phi{end + 1} = @(lambda) nPixelCross(lambda, 'x', 2, conductivities, nElc, nElf, 'max');
% phi{end + 1} = @(lambda) nPixelCross(lambda, 'y', 2, conductivities, nElc, nElf, 'max');
% % 
% %min log
% phi{end + 1} = @(lambda) log(nPixelCross(lambda, 'x', 1, conductivities, nElc, nElf, 'min') + log_cutoff);
% phi{end + 1} = @(lambda) log(nPixelCross(lambda, 'y', 1, conductivities, nElc, nElf, 'min') + log_cutoff);
% phi{end + 1} = @(lambda) log(nPixelCross(lambda, 'x', 2, conductivities, nElc, nElf, 'min') + log_cutoff);
% phi{end + 1} = @(lambda) log(nPixelCross(lambda, 'y', 2, conductivities, nElc, nElf, 'min') + log_cutoff);
% 
% %max log
% phi{end + 1} = @(lambda) log(nPixelCross(lambda, 'x', 1, conductivities, nElc, nElf, 'max') + log_cutoff);
% phi{end + 1} = @(lambda) log(nPixelCross(lambda, 'y', 1, conductivities, nElc, nElf, 'max') + log_cutoff);
% phi{end + 1} = @(lambda) log(nPixelCross(lambda, 'x', 2, conductivities, nElc, nElf, 'max') + log_cutoff);
% phi{end + 1} = @(lambda) log(nPixelCross(lambda, 'y', 2, conductivities, nElc, nElf, 'max') + log_cutoff);
% 
% %mean, this is deeply (linearly?) connected to volume fraction
% phi{end + 1} = @(lambda) nPixelCross(lambda, 'x', 1, conductivities, nElc, nElf, 'mean');
% phi{end + 1} = @(lambda) nPixelCross(lambda, 'y', 1, conductivities, nElc, nElf, 'mean');
% phi{end + 1} = @(lambda) nPixelCross(lambda, 'x', 2, conductivities, nElc, nElf, 'mean');
% phi{end + 1} = @(lambda) nPixelCross(lambda, 'y', 2, conductivities, nElc, nElf, 'mean');
% 
% 
% % pathLengths = (0:1:30)';
% % lpa = @(lambda) linPathParams(lambda, pathLengths, conductivities, romObj.coarseScaleDomain, romObj.fineScaleDomain, 'a');
% % lpb = @(lambda) abs(linPathParams(lambda, pathLengths, conductivities, romObj.coarseScaleDomain, romObj.fineScaleDomain, 'b'));
% 
% phi{end + 1} = @(lambda) meanPoreSize(lambda, 2, conductivities, nElc, nElf, 'mean');
% phi{end + 1} = @(lambda) meanPoreSize(lambda, 2, conductivities, nElc, nElf, 'var');
% % phi{end + 1} = @(lambda) sqrt(meanPoreSize(lambda, 2, conductivities, nElc, nElf, 'var'));  %std
% % %log
% % phi{end + 1} = @(lambda) log(meanPoreSize(lambda, 2, conductivities, nElc, nElf, 'mean') + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanPoreSize(lambda, 2, conductivities, nElc, nElf, 'var') + log_cutoff);
% % 
% phi{end + 1} = @(lambda) specificSurface(lambda, 2, conductivities, nElc, nElf);
% phi{end + 1} = @(lambda) log(specificSurface(lambda, 2, conductivities, nElc, nElf) + log_cutoff);
% 
% %high conducting phase
% %chessboard distance
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'hi', 'chessboard', 'mean');
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'hi', 'chessboard', 'var');
% phi{end + 1} = @(lambda) sqrt(distanceProps(lambda,...
%     conductivities, 'hi', 'chessboard', 'var')); %std
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'hi', 'chessboard', 'max');
% % 
% % %cityblock distance
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'hi', 'cityblock', 'mean');
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'hi', 'cityblock', 'var');
% phi{end + 1} = @(lambda) sqrt(distanceProps(lambda,...
%     conductivities, 'hi', 'cityblock', 'var')); %std
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'hi', 'cityblock', 'max');
% % 
% % %euclidean distance
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'hi', 'euclidean', 'mean');
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'hi', 'euclidean', 'var');
% phi{end + 1} = @(lambda) sqrt(distanceProps(lambda,...
%     conductivities, 'hi', 'euclidean', 'var')); %std
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'hi', 'euclidean', 'max');
% 
% %quasi-euclidean distance
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'hi', 'quasi-euclidean', 'mean');
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'hi', 'quasi-euclidean', 'var');
% phi{end + 1} = @(lambda) sqrt(distanceProps(lambda,...
%     conductivities, 'hi', 'quasi-euclidean', 'var')); %std
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'hi', 'quasi-euclidean', 'max');
% 
% %low conducting phase
% %chessboard distance
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'lo', 'chessboard', 'mean');
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'lo', 'chessboard', 'var');
% phi{end + 1} = @(lambda) sqrt(distanceProps(lambda,...
%     conductivities, 'lo', 'chessboard', 'var')); %std
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'lo', 'chessboard', 'max');
% % 
% % %cityblock distance
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'lo', 'cityblock', 'mean');
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'lo', 'cityblock', 'var');
% phi{end + 1} = @(lambda) sqrt(distanceProps(lambda,...
%     conductivities, 'lo', 'cityblock', 'var')); %std
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'lo', 'cityblock', 'max');
% % 
% % %euclidean distance
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'lo', 'euclidean', 'mean');
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'lo', 'euclidean', 'var');
% phi{end + 1} = @(lambda) sqrt(distanceProps(lambda,...
%     conductivities, 'lo', 'euclidean', 'var')); %std
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'lo', 'euclidean', 'max');
% 
% %quasi-euclidean distance
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'lo', 'quasi-euclidean', 'mean');
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'lo', 'quasi-euclidean', 'var');
% phi{end + 1} = @(lambda) sqrt(distanceProps(lambda,...
%     conductivities, 'lo', 'quasi-euclidean', 'var')); %std
% phi{end + 1} = @(lambda) distanceProps(lambda,...
%     conductivities, 'lo', 'quasi-euclidean', 'max');
% 
% %log's
% %high conducting phase
% %chessboard distance
% phi{end + 1} = @(lambda) log(distanceProps(lambda,...
%     conductivities, 'hi', 'chessboard', 'mean') + log_cutoff);
% % phi{end + 1} = @(lambda) log(distanceProps(lambda,...
% %     conductivities, 'hi', 'chessboard', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(sqrt(distanceProps(lambda,...
% %     conductivities, 'hi', 'chessboard', 'var')) + log_cutoff); %std
% phi{end + 1} = @(lambda) log(distanceProps(lambda,...
%     conductivities, 'hi', 'chessboard', 'max') + log_cutoff);
% % 
% % %cityblock distance
% phi{end + 1} = @(lambda) log(distanceProps(lambda,...
%     conductivities, 'hi', 'cityblock', 'mean') + log_cutoff);
% % phi{end + 1} = @(lambda) log(distanceProps(lambda,...
% %     conductivities, 'hi', 'cityblock', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(sqrt(distanceProps(lambda,...
% %     conductivities, 'hi', 'cityblock', 'var')) + log_cutoff); %std
% phi{end + 1} = @(lambda) log(distanceProps(lambda,...
%     conductivities, 'hi', 'cityblock', 'max') + log_cutoff);
% % 
% % %euclidean distance
% phi{end + 1} = @(lambda) log(distanceProps(lambda,...
%     conductivities, 'hi', 'euclidean', 'mean') + log_cutoff);
% % phi{end + 1} = @(lambda) log(distanceProps(lambda,...
% %     conductivities, 'hi', 'euclidean', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(sqrt(distanceProps(lambda,...
% %     conductivities, 'hi', 'euclidean', 'var')) + log_cutoff); %std
% phi{end + 1} = @(lambda) log(distanceProps(lambda,...
%     conductivities, 'hi', 'euclidean', 'max') + log_cutoff);
% % 
% % %quasi-euclidean distance
% % phi{end + 1} = @(lambda) log(distanceProps(lambda,...
% %     conductivities, 'hi', 'quasi-euclidean', 'mean') + log_cutoff);
% % phi{end + 1} = @(lambda) log(distanceProps(lambda,...
% %     conductivities, 'hi', 'quasi-euclidean', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(sqrt(distanceProps(lambda,...
% %     conductivities, 'hi', 'quasi-euclidean', 'var')) + log_cutoff); %std
% % phi{end + 1} = @(lambda) log(distanceProps(lambda,...
% %     conductivities, 'hi', 'quasi-euclidean', 'max') + log_cutoff);
% % 
% % %low conducting phase
% % %chessboard distance
% phi{end + 1} = @(lambda) log(distanceProps(lambda,...
%     conductivities, 'lo', 'chessboard', 'mean') + log_cutoff);
% % phi{end + 1} = @(lambda) log(distanceProps(lambda,...
% %     conductivities, 'lo', 'chessboard', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(sqrt(distanceProps(lambda,...
% %     conductivities, 'lo', 'chessboard', 'var')) + log_cutoff); %std
% phi{end + 1} = @(lambda) log(distanceProps(lambda,...
%     conductivities, 'lo', 'chessboard', 'max') + log_cutoff);
% % 
% % %cityblock distance
% phi{end + 1} = @(lambda) log(distanceProps(lambda,...
%     conductivities, 'lo', 'cityblock', 'mean') + log_cutoff);
% % phi{end + 1} = @(lambda) log(distanceProps(lambda,...
% %     conductivities, 'lo', 'cityblock', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(sqrt(distanceProps(lambda,...
% %     conductivities, 'lo', 'cityblock', 'var')) + log_cutoff); %std
% phi{end + 1} = @(lambda) log(distanceProps(lambda,...
%     conductivities, 'lo', 'cityblock', 'max') + log_cutoff);
% % 
% % %euclidean distance
% phi{end + 1} = @(lambda) log(distanceProps(lambda,...
%     conductivities, 'lo', 'euclidean', 'mean') + log_cutoff);
% % phi{end + 1} = @(lambda) log(distanceProps(lambda,...
% %     conductivities, 'lo', 'euclidean', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(sqrt(distanceProps(lambda,...
% %     conductivities, 'lo', 'euclidean', 'var')) + log_cutoff); %std
% phi{end + 1} = @(lambda) log(distanceProps(lambda,...
%     conductivities, 'lo', 'euclidean', 'max') + log_cutoff);
% 
% %quasi-euclidean distance
% phi{end + 1} = @(lambda) log(distanceProps(lambda,...
%     conductivities, 'lo', 'quasi-euclidean', 'mean') + log_cutoff);
% phi{end + 1} = @(lambda) log(distanceProps(lambda,...
%     conductivities, 'lo', 'quasi-euclidean', 'var') + log_cutoff);
% phi{end + 1} = @(lambda) log(sqrt(distanceProps(lambda,...
%     conductivities, 'lo', 'quasi-euclidean', 'var')) + log_cutoff); %std
% phi{end + 1} = @(lambda) log(distanceProps(lambda,...
%     conductivities, 'lo', 'quasi-euclidean', 'max') + log_cutoff);
% 
% %Counts the number of separate high conducting blobs
% phi{end + 1} = @(lambda) numberOfObjects(lambda,...
%     conductivities, 'hi');
% %Counts the number of separate low conducting blobs
% phi{end + 1} = @(lambda) numberOfObjects(lambda,...
%     conductivities, 'lo');
% %log
% phi{end + 1} = @(lambda) log(numberOfObjects(lambda,...
%     conductivities, 'hi') + log_cutoff);
% %Counts the number of separate low conducting blobs
% phi{end + 1} = @(lambda) log(numberOfObjects(lambda,...
%     conductivities, 'lo') + log_cutoff);
% 
% 
% %% High conducting phase
% %means
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Area', 'mean');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'ConvexArea', 'mean');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Eccentricity', 'mean');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Extent', 'mean');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'MajorAxisLength', 'mean');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'MinorAxisLength', 'mean');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'mean');
% phi{end + 1} = @(lambda) sind(meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'mean'));
% phi{end + 1} = @(lambda) cosd(meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'mean'));
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Perimeter', 'mean');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Solidity', 'mean');
% % %Variances
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'hi', 'Area', 'var');
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'hi', 'ConvexArea', 'var');
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'hi', 'Eccentricity', 'var');
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'hi', 'Extent', 'var');
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'hi', 'MajorAxisLength', 'var');
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'hi', 'MinorAxisLength', 'var');
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'hi', 'Orientation', 'var');
% % phi{end + 1} = @(lambda) sind(meanImageProps(lambda,...
% %     conductivities, 'hi', 'Orientation', 'var'));
% % phi{end + 1} = @(lambda) cosd(meanImageProps(lambda,...
% %     conductivities, 'hi', 'Orientation', 'var'));
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'hi', 'Perimeter', 'var');
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'hi', 'Solidity', 'var');
% % %Maxima
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Area', 'max');
% phi{end + 1} = @(lambda) meanImageProps(lambda, conductivities, 'hi', 'ConvexArea', 'max');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Eccentricity', 'max');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Extent', 'max');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'MajorAxisLength', 'max');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'MinorAxisLength', 'max');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'max');
% phi{end + 1} = @(lambda) sind(meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'max'));
% phi{end + 1} = @(lambda) cosd(meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'max'));
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Perimeter', 'max');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Solidity', 'max');
% % %Minima
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Area', 'min');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'ConvexArea', 'min');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Eccentricity', 'min');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Extent', 'min');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'MajorAxisLength', 'min');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'MinorAxisLength', 'min');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'min');
% phi{end + 1} = @(lambda) sind(meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'min'));
% phi{end + 1} = @(lambda) cosd(meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'min'));
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Perimeter', 'min');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'hi', 'Solidity', 'min');
% %standard deviations
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'hi', 'Area', 'var').^.5;
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'hi', 'ConvexArea', 'var').^.5;
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'hi', 'Eccentricity', 'var').^.5;
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'hi', 'Extent', 'var').^.5;
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'hi', 'MajorAxisLength', 'var').^.5;
% % phi{end + 1} = @(lambda) abs(meanImageProps(lambda,...
% %     conductivities, 'hi', 'MinorAxisLength', 'var')).^.5;
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'hi', 'Orientation', 'var').^.5;
% % phi{end + 1} = @(lambda) sind(meanImageProps(lambda,...
% %     conductivities, 'hi', 'Orientation', 'var').^.5);
% % phi{end + 1} = @(lambda) cosd(meanImageProps(lambda,...
% %     conductivities, 'hi', 'Orientation', 'var').^.5);
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'hi', 'Perimeter', 'var').^.5;
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'hi', 'Solidity', 'var').^.5;
% % %maximum bubble extents
% phi{end + 1} = @(lambda) maxExtent(lambda,...
%     conductivities, 'hi', 'x');
% phi{end + 1} = @(lambda) maxExtent(lambda,...
%     conductivities, 'hi', 'y');
% %mean bubble extents
% phi{end + 1} = @(lambda) meanExtent(lambda,...
%     conductivities, 'hi', 'x', 'mean');
% phi{end + 1} = @(lambda) meanExtent(lambda,...
%     conductivities, 'hi', 'y', 'mean');
% % phi{end + 1} = @(lambda) meanExtent(lambda,...
% %     conductivities, 'hi', 'x', 'var');
% % phi{end + 1} = @(lambda) meanExtent(lambda,...
% %     conductivities, 'hi', 'y', 'var');
% % 
% % %% Low conducting phase
% % %means
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Area', 'mean');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'ConvexArea', 'mean');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Eccentricity', 'mean');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Extent', 'mean');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'MajorAxisLength', 'mean');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'MinorAxisLength', 'mean');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'mean');
% phi{end + 1} = @(lambda) sind(meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'mean'));
% phi{end + 1} = @(lambda) cosd(meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'mean'));
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Perimeter', 'mean');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Solidity', 'mean');
% % %Variances
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Area', 'var');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'ConvexArea', 'var');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Eccentricity', 'var');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Extent', 'var');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'MajorAxisLength', 'var');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'MinorAxisLength', 'var');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'var');
% phi{end + 1} = @(lambda) sind(meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'var'));
% phi{end + 1} = @(lambda) cosd(meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'var'));
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Perimeter', 'var');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Solidity', 'var');
% % %Maxima
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Area', 'max');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'ConvexArea', 'max');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Eccentricity', 'max');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Extent', 'max');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'MajorAxisLength', 'max');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'MinorAxisLength', 'max');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'max');
% phi{end + 1} = @(lambda) sind(meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'max'));
% phi{end + 1} = @(lambda) cosd(meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'max'));
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Perimeter', 'max');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Solidity', 'max');
% % %Minima
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Area', 'min');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'ConvexArea', 'min');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Eccentricity', 'min');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Extent', 'min');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'MajorAxisLength', 'min');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'MinorAxisLength', 'min');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'min');
% phi{end + 1} = @(lambda) sind(meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'min'));
% phi{end + 1} = @(lambda) cosd(meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'min'));
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Perimeter', 'min');
% phi{end + 1} = @(lambda) meanImageProps(lambda,...
%     conductivities, 'lo', 'Solidity', 'min');
% % %standard deviations
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'lo', 'Area', 'var').^.5;
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'lo', 'ConvexArea', 'var').^.5;
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'lo', 'Eccentricity', 'var').^.5;
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'lo', 'Extent', 'var').^.5;
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'lo', 'MajorAxisLength', 'var').^.5;
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'lo', 'MinorAxisLength', 'var').^.5;
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'lo', 'Orientation', 'var').^.5;
% % phi{end + 1} = @(lambda) sind(meanImageProps(lambda,...
% %     conductivities, 'lo', 'Orientation', 'var').^.5);
% % phi{end + 1} = @(lambda) cosd(meanImageProps(lambda,...
% %     conductivities, 'lo', 'Orientation', 'var').^.5);
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'lo', 'Perimeter', 'var').^.5;
% % phi{end + 1} = @(lambda) meanImageProps(lambda,...
% %     conductivities, 'lo', 'Solidity', 'var').^.5;
% % %maximum bubble extents
% phi{end + 1} = @(lambda) maxExtent(lambda,...
%     conductivities, 'lo', 'x');
% phi{end + 1} = @(lambda) maxExtent(lambda,...
%     conductivities, 'lo', 'y');
% %mean bubble extents
% phi{end + 1} = @(lambda) meanExtent(lambda,...
%     conductivities, 'lo', 'x', 'mean');
% phi{end + 1} = @(lambda) meanExtent(lambda,...
%     conductivities, 'lo', 'y', 'mean');
% % phi{end + 1} = @(lambda) meanExtent(lambda,...
% %     conductivities, 'lo', 'x', 'var');
% % phi{end + 1} = @(lambda) meanExtent(lambda,...
% %     conductivities, 'lo', 'y', 'var');
% % 
% % 
% % 
% % 
% % %logs
% % %% High conducting phase
% %means
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Area', 'mean') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'ConvexArea', 'mean') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Eccentricity', 'mean') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Extent', 'mean') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'MajorAxisLength', 'mean') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'MinorAxisLength', 'mean') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'mean') + 90 + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(abs(sind(meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'mean') + 90)) + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(abs(cosd(meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'mean') + 90)) + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Perimeter', 'mean') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Solidity', 'mean') + log_cutoff);
% % %Variances
% % phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
% %     conductivities, 'hi', 'Area', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
% %     conductivities, 'hi', 'ConvexArea', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
% %     conductivities, 'hi', 'Eccentricity', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
% %     conductivities, 'hi', 'Extent', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
% %     conductivities, 'hi', 'MajorAxisLength', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(abs(meanImageProps(lambda,...
% %     conductivities, 'hi', 'MinorAxisLength', 'var')) + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
% %     conductivities, 'hi', 'Orientation', 'var') + 90 + eps); %+90 s.t. log argument is positive
% % phi{end + 1} = @(lambda) log(abs(sind(meanImageProps(lambda,...
% %     conductivities, 'hi', 'Orientation', 'var') + 90 + eps))); %+90 s.t. log argument is positive
% % phi{end + 1} = @(lambda) log(abs(cosd(meanImageProps(lambda,...
% %     conductivities, 'hi', 'Orientation', 'var') + 90 + eps))); %+90 s.t. log argument is positive
% % phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
% %     conductivities, 'hi', 'Perimeter', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
% %     conductivities, 'hi', 'Solidity', 'var') + log_cutoff);
% %Maxima
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Area', 'max') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'ConvexArea', 'max') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Eccentricity', 'max') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Extent', 'max') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'MajorAxisLength', 'max') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'MinorAxisLength', 'max') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'max') + 90 + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(abs(sind(meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'max') + 90)) + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(abs(cosd(meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'max') + 90)) + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Perimeter', 'max') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Solidity', 'max') + log_cutoff);
% %Minima
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Area', 'min') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'ConvexArea', 'min') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Eccentricity', 'min') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Extent', 'min') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'MajorAxisLength', 'min') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'MinorAxisLength', 'min') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'min') + 90 + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(abs(sind(meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'min') + 90)) + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(abs(cosd(meanImageProps(lambda,...
%     conductivities, 'hi', 'Orientation', 'min') + 90)) + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Perimeter', 'min') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'hi', 'Solidity', 'min') + log_cutoff);
% % 
% % %maximum bubble extents
% phi{end + 1} = @(lambda) log(maxExtent(lambda,...
%     conductivities, 'hi', 'x') + log_cutoff);
% phi{end + 1} = @(lambda) log(maxExtent(lambda,...
%     conductivities, 'hi', 'y') + log_cutoff);
% %mean bubble extents
% phi{end + 1} = @(lambda) log(meanExtent(lambda,...
%     conductivities, 'hi', 'x', 'mean') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanExtent(lambda,...
%     conductivities, 'hi', 'y', 'mean') + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanExtent(lambda,...
% %     conductivities, 'hi', 'x', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanExtent(lambda,...
% %     conductivities, 'hi', 'y', 'var') + log_cutoff);
% % 
% %% Low conducting phase
% %means
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Area', 'mean') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'ConvexArea', 'mean') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Eccentricity', 'mean') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Extent', 'mean') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'MajorAxisLength', 'mean') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'MinorAxisLength', 'mean') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'mean') + 90 + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(abs(sind(meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'mean') + 90)) + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(abs(cosd(meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'mean') + 90)) + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Perimeter', 'mean') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Solidity', 'mean') + log_cutoff);
% % %Variances
% % phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
% %     conductivities, 'lo', 'Area', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
% %     conductivities, 'lo', 'ConvexArea', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
% %     conductivities, 'lo', 'Eccentricity', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
% %     conductivities, 'lo', 'Extent', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
% %     conductivities, 'lo', 'MajorAxisLength', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
% %     conductivities, 'lo', 'MinorAxisLength', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
% %     conductivities, 'lo', 'Orientation', 'var') + 90 + eps); %+90 s.t. log argument is positive
% % phi{end + 1} = @(lambda) log(abs(sind(meanImageProps(lambda,...
% %     conductivities, 'lo', 'Orientation', 'var') + 90 + eps))); %+90 s.t. log argument is positive
% % phi{end + 1} = @(lambda) log(abs(cosd(meanImageProps(lambda,...
% %     conductivities, 'lo', 'Orientation', 'var') + 90 + eps))); %+90 s.t. log argument is positive
% % phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
% %     conductivities, 'lo', 'Perimeter', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
% %     conductivities, 'lo', 'Solidity', 'var') + log_cutoff);
% %Maxima
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Area', 'max') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'ConvexArea', 'max') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Eccentricity', 'max') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Extent', 'max') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'MajorAxisLength', 'max') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'MinorAxisLength', 'max') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'max') + 90 + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(abs(sind(meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'max') + 90)) + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(abs(cosd(meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'max') + 90)) + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Perimeter', 'max') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Solidity', 'max') + log_cutoff);    %unstable
% %Minima
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Area', 'min') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'ConvexArea', 'min') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Eccentricity', 'min') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Extent', 'min') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'MajorAxisLength', 'min') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'MinorAxisLength', 'min') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'min') + 90 + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(abs(sind(meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'min') + 90 )) + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(abs(cosd(meanImageProps(lambda,...
%     conductivities, 'lo', 'Orientation', 'min') + 90)) + log_cutoff); %+90 s.t. log argument is positive
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Perimeter', 'min') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanImageProps(lambda,...
%     conductivities, 'lo', 'Solidity', 'min') + log_cutoff);
% % 
% % %maximum bubble extents
% phi{end + 1} = @(lambda) log(maxExtent(lambda,...
%     conductivities, 'lo', 'x') + log_cutoff);
% phi{end + 1} = @(lambda) log(maxExtent(lambda,...
%     conductivities, 'lo', 'y') + log_cutoff);      %is often -Inf
% % %mean bubble extents
% phi{end + 1} = @(lambda) log(meanExtent(lambda,...
%     conductivities, 'lo', 'x', 'mean') + log_cutoff);
% phi{end + 1} = @(lambda) log(meanExtent(lambda,...
%     conductivities, 'lo', 'y', 'mean') + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanExtent(lambda,...
% %     conductivities, 'lo', 'x', 'var') + log_cutoff);
% % phi{end + 1} = @(lambda) log(meanExtent(lambda,...
% %     conductivities, 'lo', 'y', 'var') + log_cutoff);
% % 
% % 
% % %Energy of conductivity field if it were Ising model
% % phi{end + 1} = @(lambda) isingEnergy(lambda);
% phi{end + 1} = @(lambda) log(isingEnergy(lambda));
% % 
% % 
% % %Maxwell-Garnett effective conductivity
% phi{end + 1} = @(lambda) maxwellGarnett(lambda, conductivities, 'plain', 'hi');
phi{end + 1} = @(lambda) maxwellGarnett(lambda, conductivities, 'log', 'hi');
% % phi{end + 1} = @(lambda) maxwellGarnett(lambda, conductivities, 'logit', 'hi');
% phi{end + 1} = @(lambda) maxwellGarnett(lambda, conductivities, 'plain', 'lo');
% phi{end + 1} = @(lambda) maxwellGarnett(lambda, conductivities, 'log', 'lo');
% % phi{end + 1} = @(lambda) maxwellGarnett(lambda, conductivities, 'logit', 'lo');
% % 
% % 
% %% Self-consistent approximation
% phi{end + 1} = @(lambda) SCA(lambda, conductivities, 'plain');
% phi{end + 1} = @(lambda) SCA(lambda, conductivities, 'log');
% % phi{end + 1} = @(lambda) SCA(lambda, conductivities, 'logit');
% % 
% 
% %% Weak contrast expansion Torquato 20.77, only use for |lambda_1 - lambda_2| < 1
% % phi{end + 1} = @(lambda) weakContrastExpansion(lambda, conductivities, 'plain', 'lo');
% % phi{end + 1} = @(lambda) weakContrastExpansion(lambda, conductivities, 'log', 'lo');
% % phi{end + 1} = @(lambda) weakContrastExpansion(lambda, conductivities, 'logit', 'lo');
% % phi{end + 1} = @(lambda) weakContrastExpansion(lambda, conductivities, 'plain', 'hi');
% % phi{end + 1} = @(lambda) weakContrastExpansion(lambda, conductivities, 'log', 'hi');
% % phi{end + 1} = @(lambda) weakContrastExpansion(lambda, conductivities, 'logit', 'hi');
% 
% %% Differential effective medium, Torquato 18.23
% phi{end + 1} = @(lambda) differentialEffectiveMedium(lambda, conductivities, 'plain', 'lo');
% phi{end + 1} = @(lambda) differentialEffectiveMedium(lambda, conductivities, 'log', 'lo');
% % phi{end + 1} = @(lambda) differentialEffectiveMedium(lambda, conductivities, 'logit', 'lo');
% phi{end + 1} = @(lambda) differentialEffectiveMedium(lambda, conductivities, 'plain', 'hi');
% phi{end + 1} = @(lambda) differentialEffectiveMedium(lambda, conductivities, 'log', 'hi');
% % phi{end + 1} = @(lambda) differentialEffectiveMedium(lambda, conductivities, 'logit', 'hi');
% 
% 
% %% Is there a connected path from left/right, bottom/top in phase i?
% % phi{end + 1} = @(lambda) connectedPathExist(lambda, 1, conductivities, 'x', nElf, nElc, 'exist');
% % phi{end + 1} = @(lambda) connectedPathExist(lambda, 1, conductivities, 'y', nElf, nElc, 'exist');
% % phi{end + 1} = @(lambda) connectedPathExist(lambda, 2, conductivities, 'x', nElf, nElc, 'exist');
% % phi{end + 1} = @(lambda) connectedPathExist(lambda, 2, conductivities, 'y', nElf, nElc, 'exist');
% 
% phi{end + 1} = @(lambda) connectedPathExist(lambda, 1, conductivities, 'x', nElf, nElc, 'invdist');
% phi{end + 1} = @(lambda) connectedPathExist(lambda, 1, conductivities, 'y', nElf, nElc, 'invdist');
% phi{end + 1} = @(lambda) connectedPathExist(lambda, 2, conductivities, 'x', nElf, nElc, 'invdist');
% phi{end + 1} = @(lambda) connectedPathExist(lambda, 2, conductivities, 'y', nElf, nElc, 'invdist');
% 
% 
% %% Generalized means along straight lines
% phi{end + 1} = @(lambda) generalizedMeanPath(lambda, 'x', -1, 'mean');
% phi{end + 1} = @(lambda) generalizedMeanPath(lambda, 'y', -1, 'mean');
% phi{end + 1} = @(lambda) generalizedMeanPath(lambda, 'x', -1, 'max');
% phi{end + 1} = @(lambda) generalizedMeanPath(lambda, 'y', -1, 'max');
% phi{end + 1} = @(lambda) generalizedMeanPath(lambda, 'x', -1, 'min');
% phi{end + 1} = @(lambda) generalizedMeanPath(lambda, 'y', -1, 'min');
% 
% phi{end + 1} = @(lambda) generalizedMeanPath(lambda, 'x', 0, 'mean');
% phi{end + 1} = @(lambda) generalizedMeanPath(lambda, 'y', 0, 'mean');
% phi{end + 1} = @(lambda) generalizedMeanPath(lambda, 'x', 0, 'max');
% phi{end + 1} = @(lambda) generalizedMeanPath(lambda, 'y', 0, 'max');
% phi{end + 1} = @(lambda) generalizedMeanPath(lambda, 'x', 0, 'min');
% phi{end + 1} = @(lambda) generalizedMeanPath(lambda, 'y', 0, 'min');
% 
% phi{end + 1} = @(lambda) generalizedMeanPath(lambda, 'x', 1, 'max');
% phi{end + 1} = @(lambda) generalizedMeanPath(lambda, 'y', 1, 'max');
% phi{end + 1} = @(lambda) generalizedMeanPath(lambda, 'x', 1, 'min');
% phi{end + 1} = @(lambda) generalizedMeanPath(lambda, 'y', 1, 'min');
% 
% 
% phi{end + 1} = @(lambda) log(generalizedMeanPath(lambda, 'x', -1, 'mean'));
% phi{end + 1} = @(lambda) log(generalizedMeanPath(lambda, 'y', -1, 'mean'));
% phi{end + 1} = @(lambda) log(generalizedMeanPath(lambda, 'x', -1, 'max'));
% phi{end + 1} = @(lambda) log(generalizedMeanPath(lambda, 'y', -1, 'max'));
% phi{end + 1} = @(lambda) log(generalizedMeanPath(lambda, 'x', -1, 'min'));
% phi{end + 1} = @(lambda) log(generalizedMeanPath(lambda, 'y', -1, 'min'));
% 
% phi{end + 1} = @(lambda) log(generalizedMeanPath(lambda, 'x', 0, 'mean'));
% phi{end + 1} = @(lambda) log(generalizedMeanPath(lambda, 'y', 0, 'mean'));
% phi{end + 1} = @(lambda) log(generalizedMeanPath(lambda, 'x', 0, 'max'));
% phi{end + 1} = @(lambda) log(generalizedMeanPath(lambda, 'y', 0, 'max'));
% phi{end + 1} = @(lambda) log(generalizedMeanPath(lambda, 'x', 0, 'min'));
% phi{end + 1} = @(lambda) log(generalizedMeanPath(lambda, 'y', 0, 'min'));
% 
% phi{end + 1} = @(lambda) log(generalizedMeanPath(lambda, 'x', 1, 'max'));
% phi{end + 1} = @(lambda) log(generalizedMeanPath(lambda, 'y', 1, 'max'));
% phi{end + 1} = @(lambda) log(generalizedMeanPath(lambda, 'x', 1, 'min'));
% phi{end + 1} = @(lambda) log(generalizedMeanPath(lambda, 'y', 1, 'min'));
% 
% % Generalized means along boundaries of coarse element
% phi{end + 1} = @(lambda) generalizedMeanBoundary(lambda, nElc, nElf, -1, 'left');
% phi{end + 1} = @(lambda) generalizedMeanBoundary(lambda, nElc, nElf, -1, 'lower');
% phi{end + 1} = @(lambda) generalizedMeanBoundary(lambda, nElc, nElf, -1, 'right');
% phi{end + 1} = @(lambda) generalizedMeanBoundary(lambda, nElc, nElf, -1, 'upper');
% 
% phi{end + 1} = @(lambda) generalizedMeanBoundary(lambda, nElc, nElf, 0, 'left');
% phi{end + 1} = @(lambda) generalizedMeanBoundary(lambda, nElc, nElf, 0, 'lower');
% phi{end + 1} = @(lambda) generalizedMeanBoundary(lambda, nElc, nElf, 0, 'right');
% phi{end + 1} = @(lambda) generalizedMeanBoundary(lambda, nElc, nElf, 0, 'upper');
% 
% phi{end + 1} = @(lambda) generalizedMeanBoundary(lambda, nElc, nElf, 1, 'left');
% phi{end + 1} = @(lambda) generalizedMeanBoundary(lambda, nElc, nElf, 1, 'lower');
% phi{end + 1} = @(lambda) generalizedMeanBoundary(lambda, nElc, nElf, 1, 'right');
% phi{end + 1} = @(lambda) generalizedMeanBoundary(lambda, nElc, nElf, 1, 'upper');
% 
% 
% %logs
% phi{end + 1} = @(lambda) log(generalizedMeanBoundary(lambda, nElc, nElf, -1, 'left'));
% phi{end + 1} = @(lambda) log(generalizedMeanBoundary(lambda, nElc, nElf, -1, 'lower'));
% phi{end + 1} = @(lambda) log(generalizedMeanBoundary(lambda, nElc, nElf, -1, 'right'));
% phi{end + 1} = @(lambda) log(generalizedMeanBoundary(lambda, nElc, nElf, -1, 'upper'));
% 
% phi{end + 1} = @(lambda) log(generalizedMeanBoundary(lambda, nElc, nElf, 0, 'left'));
% phi{end + 1} = @(lambda) log(generalizedMeanBoundary(lambda, nElc, nElf, 0, 'lower'));
% phi{end + 1} = @(lambda) log(generalizedMeanBoundary(lambda, nElc, nElf, 0, 'right'));
% phi{end + 1} = @(lambda) log(generalizedMeanBoundary(lambda, nElc, nElf, 0, 'upper'));
% 
% phi{end + 1} = @(lambda) log(generalizedMeanBoundary(lambda, nElc, nElf, 1, 'left'));
% phi{end + 1} = @(lambda) log(generalizedMeanBoundary(lambda, nElc, nElf, 1, 'lower'));
% phi{end + 1} = @(lambda) log(generalizedMeanBoundary(lambda, nElc, nElf, 1, 'right'));
% phi{end + 1} = @(lambda) log(generalizedMeanBoundary(lambda, nElc, nElf, 1, 'upper'));


% linCombPixels = true;
% if linCombPixels
%     for i = 1:(prod(nElf)/prod(nElc))
%         phi{end + 1} = @(lambda) log(lambda(i));
%     end
% end

%linear filter with equal weights to start with
% phi{end + 1} = @(lambda) (1/sqrt(romObj.fineScaleDomain.nEl/romObj.coarseScaleDomain.nEl))*sum(log(lambda));
% 

% %pooling. keep in mind to use 'global' mode
% wndw = [64 64];
% stride = wndw;
% % poolfunc = @(lambda) generalizedMean(lambda, -1);
% poolfunc = @(lambda) SCA(lambda(:), conductivities, 'log');
% % %There is much unnecessary overhead here!
% % %Check the number of pooled pixels if you change anything here!
% % for i = 1:((prod(nElf)/prod(nElc))/prod(wndw))
% %     phi{end + 1} = @(lambda) log(poolingFeature(lambda, i, wndw, poolfunc, i, stride));
% % end
% 
% phi{end + 1} = @(lambda) pool2d(lambda, wndw, poolfunc, 0, stride);


% nRandFilt = 10;
% w = 2*rand(romObj.fineScaleDomain.nEl/romObj.coarseScaleDomain.nEl, nRandFilt) - 1;
% for i = 1:nRandFilt
%     w(:, i) = w(:, i)/norm(w(:, i));
%     phi{end + 1} = @(lambda) sum(w(:, i).*lambda);
% end

nBasis = numel(phi);


%print feature function strings to file
delete('./data/basisFunctions.txt');
fid = fopen('./data/basisFunctions.txt','wt');
for i = 1:length(phi)
    phi_str = func2str(phi{i});
    phi_str = strcat(phi_str, '\n');
    fprintf(fid, phi_str);
end
fclose(fid);



