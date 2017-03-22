function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

[m,n] = size(R);

for i=1:m
    for j=1:n
        if(R(i,j) == 1)
            % user have rated the moovie
            
            J = J + (Theta(j,:)*X(i,:)' - Y(i,j))^2;
        end
    end
end

J = J / 2 + lambda / 2 *sum(sum(Theta.^2)) + lambda / 2 * sum(sum(X.^2));

for i=1:m    
    for k=1:num_features    
        for j=1:n
            if(R(i,j) == 1)
                % user have rated the moovie
                X_grad(i,k) = X_grad(i,k) + (Theta(j,:)*X(i,:)' - Y(i,j)) * Theta(j,k);             
            end
        end
        
        X_grad(i,k) = X_grad(i,k) + lambda * X(i,k);
    end
end

% for i=size(X_grad,1)
%     for j=size(X_grad,2)
%         X_grad(i,j) = X_grad(i,j) + lambda * X(i,j);
%     end
% end

for j=1:num_users     
    for i=1:m    
        for k=1:num_features
            if(R(i,j) == 1)
                % user have rated the moovie
                Theta_grad(j,k) = Theta_grad(j,k) + (Theta(j,:)*X(i,:)' - Y(i,j)) * X(i,k);             
            end
        end
        Theta_grad(j,i) = Theta_grad(j,i) + lambda * Theta(j,i);
    end
end

% for i=size(Theta_grad,1)
%     for j=size(Theta_grad,2)
%         Theta_grad(i,j) = Theta_grad(i,j) + lambda * Theta(i,j);
%     end
% end        
    

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
