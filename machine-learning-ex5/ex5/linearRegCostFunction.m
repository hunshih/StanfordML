function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hypo = X * theta;
J = sum( (hypo - y) .^ 2 )/(2 * m);
reg = (lambda/(2 * m)) * sum(theta(2:end) .^ 2);
J = J + reg;
grad_sum = grad; % zero initialized
% Caveat: if you sum a single row (sample size 1), you turn a matrix into a scaler
if( size(X, 1) > 1)
	grad_sum = sum( (hypo - y) .* X )/m;
else
	grad_sum = ( (hypo - y) .* X )/m;
endif
grad(1) = grad_sum(1)
grad(2:end) = (grad_sum(2:end))' + (lambda/m)*theta(2:end)





% =========================================================================

grad = grad(:);

end
