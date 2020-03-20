% Wrapper to make rand_em_gain call the same as rand_em_gain_new

function outMtx = rand_em_gain_w(NinMtx, EMgain)
% rand_em_gain wrapper

[nr, nc] = size(NinMtx);
outMtx = zeros(nr, nc);

for ir = 1:nr
    for ic = 1:nc
        Nin = NinMtx(ir, ic);
        outMtx(ir, ic) = rand_em_gain(Nin, EMgain);
    end
end
end
