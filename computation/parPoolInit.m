function [] = parPoolInit(N_Threads)
%% Initializes parallel pool

if(nargin == 0 || N_Threads > 16)
    N_Threads = 16;
end

current_pool = gcp('nocreate');
if(~numel(current_pool) || (current_pool.NumWorkers < N_Threads))
    %Create with N_Threads workers
    delete(gcp('nocreate'));
    parpool('local', N_Threads);
end

end

