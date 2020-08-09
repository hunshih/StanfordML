function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
j = size(theta); % get the number of theta, need as many partial derivatives
for samp = 1:m
	sig0 = -y(samp) * log(sigmoid(X(samp, :) * theta));
	sig1 = (1 - y(samp)) * log( 1 - sigmoid(X(samp, :) * theta));
	J = J + (sig0 - sig1);
endfor % end the iteration over sample
J = J/m;

for feat = 1:(size(theta)(1)) % size returns a 1X2 vec; first element is row #
	theta_tmp = 0;
	for samp = 1:m
		theta_tmp = theta_tmp + (sigmoid(X(samp,:) * theta) - y(samp)) * X(samp, feat);
	endfor % end iteration over sample size
	theta_tmp/m
	grad(feat) = theta_tmp/m;
endfor % end the iteration over features
% =============================================================

end
