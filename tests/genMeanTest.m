%test for generalized mean deriviative

x = eye(3);
z = 0;
dz = 1e-5;
[m0, dm_dz0] = generalizedMean(x, z);
[m1, dm_dz1] = generalizedMean(x, z + dz);

dm_dz = .5*(dm_dz0 + dm_dz1)
dm_dzFD = (m1 - m0)/dz
relErr = (dm_dz - dm_dzFD)/dm_dz
m0
m1

