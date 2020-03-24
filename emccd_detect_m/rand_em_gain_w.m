% Wrapper to make rand_em_gain call the same as rand_em_gain_new

function outMtx = rand_em_gain_w(NinMtx, EMgain)
% rand_em_gain wrapper
outMtx = zeros(size(NinMtx));

for i = 1:numel(NinMtx)
    Nin = NinMtx(i);
    outMtx(i) = rand_em_gain(Nin, EMgain);
end
end
