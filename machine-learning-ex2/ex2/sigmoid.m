function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
z_ones = ones(size(z)); % all ones
z_ones = z_ones .* exp(z* -1); % all now e^(-z)
z_ones = (1 .+ z_ones); % all now 1 + e^z
g = 1 ./ z_ones;



% =============================================================

end
