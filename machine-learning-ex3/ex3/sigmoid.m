function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = 1.0 ./ (1.0 + exp(-z));


% g = zeros(size(z));
% 
% for j=1:size(z,2)
%     for i=1:size(z,1)
%         g(i,j)=1 / (1 + exp(-z(i,j)));
%     end
% end



end

