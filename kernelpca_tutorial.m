function data_out = kernelpca_tutorial(data_in,num_dim)

if num_dim > size(data_in,1)
    fprintf('\nDimensions of output data has to be lesser than the dimensions of input data\n');
    fprintf('Closing program\n');
    return
end
K = zeros(size(data_in,2),size(data_in,2));
for row = 1:size(data_in,2)
    for col = 1:row
        temp = sum(((data_in(:,row) - data_in(:,col)).^2));
        K(row,col) = exp(-temp); % sigma = 1
    end
end
K = K + K'; 
for row = 1:size(data_in,2)
    K(row,row) = K(row,row)/2;
end
one_mat = ones(size(K));
K_center = K - one_mat*K - K*one_mat + one_mat*K*one_mat;
clear K
opts.issym=1;                          
opts.disp = 0; 
opts.isreal = 1;
neigs = 10;
[eigvec eigval] = eigs(K_center,[],neigs,'lm',opts);
eig_val = eigval ~= 0;
eig_val = eig_val./size(data_in,2);
for col = 1:size(eigvec,2)
    eigvec(:,col) = eigvec(:,col)./(sqrt(eig_val(col,col)));
end
[~, index] = sort(eig_val,'descend');
eigvec = eigvec(:,index);
data_out = zeros(num_dim,size(data_in,2));
for count = 1:num_dim
    data_out(count,:) = eigvec(:,count)'*K_center';
end