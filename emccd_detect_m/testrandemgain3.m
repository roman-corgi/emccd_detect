close all
clear
clc
format compact

avgCount = 0.4 %#ok<*NOPTS>

Npix = 5

% generate a repeatable Poisson random matrix
rng(1);
NinMtx = poissrnd( avgCount * ones(Npix), Npix, Npix)

EMgain = 100

this is a test





out = randemgain3( NinMtx, EMgain )