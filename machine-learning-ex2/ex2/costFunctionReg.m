function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

j = size(theta); % get the number of theta, need as many partial derivatives
for samp = 1:m
	sig0 = -y(samp) * log(sigmoid(X(samp, :) * theta));
	sig1 = (1 - y(samp)) * log( 1 - sigmoid(X(samp, :) * theta));
	J = J + (sig0 - sig1);
endfor % end the iteration over sample
J = J/m; % only contain the non-regularized up to this point
sub = 0;
if(size(theta)(1) > 1)
	sub = theta(2:end, :);
else
	sub = theta;
endif
J = J + (lambda/(2*m)) * sum(sub .* sub); % add the lambda term

% need to do gradient for the first term separately
if (size(theta) > 0)
	theta_tmp = 0;
	for samp = 1:m
		theta_tmp = theta_tmp + (sigmoid(X(samp,:) * theta) - y(samp)) * X(samp, 1);
	endfor % end iteration over sample size
	theta_tmp/m
	grad(1) = theta_tmp/m;
endif

for feat = 2:(size(theta)(1)) % size returns a 1X2 vec; first element is row #
	theta_tmp = 0;
	for samp = 1:m
		theta_tmp = theta_tmp + (sigmoid(X(samp,:) * theta) - y(samp)) * X(samp, feat);
	endfor % end iteration over sample size
	theta_tmp/m
	grad(feat) = theta_tmp/m + (lambda/m) * theta(feat);
endfor % end the iteration over features

% =============================================================

end
