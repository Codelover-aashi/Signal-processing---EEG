function [wIC,A,W,IC] = Preprocessing(data,varargin)
% Main Fast ICA performing
[IC,A,W] = fastica(data,'g','tanh');

modulus = mod(size(data,2),2^5);
if modulus ~=0
    extra = zeros(1,(2^5)-modulus);
else
    extra = [];
end
disp('Perforing wavelet thresholding');
for s = 1:size(IC,1)
    if ~isempty(extra)
        sig = [IC(s,:),extra]; % pad with zeros
    else
        sig = IC(s,:);
    end
    [thresh,sorh,~] = ddencmp('den','wv',sig);
    
    swc = swt(sig,5,'coif5');
    Y = wthresh(swc,sorh,thresh);
    wIC(s,:) = iswt(Y,'coif5');
    clear y sig thresh sorh swc
end
if ~isempty(extra)
    wIC = wIC(:,1:end-numel(extra));
end
end