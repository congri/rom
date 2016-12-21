%performance tests

nRuns = 1e6;
a = 1:1000;
tic
for i = 1:nRuns
    Tf(a);
end
toc

a = uint32(a);
tic
for i = 1:nRuns
    Tf(a);
end
toc
