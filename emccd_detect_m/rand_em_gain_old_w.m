% Wrapper to make rand_em_gain_old call the same as rand_em_gain

function outMtx = rand_em_gain_old_w(NinMtx, EMgain)
% rand_em_gain wrapper
outMtx = zeros(size(NinMtx));

for i = 1:numel(NinMtx)
    Nin = NinMtx(i);
    outMtx(i) = rand_em_gain_old(Nin, EMgain);
end
end
