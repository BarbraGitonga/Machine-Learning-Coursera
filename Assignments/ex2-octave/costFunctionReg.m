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

% Calculating for J
z = (X * theta); % this is in the shape
h = sigmoid(z);
h1 = ((-y') *(log(h)));
h2 = ((1-y') *(log(1 - h)));
J = (h1 - h2);
J = (1/m) .* J;
theta(1) = 0;
reg = ((lambda/(2*m)) * (theta' * theta));
J = J + reg;
% Calculating for Gradient function
g = (h - y);
g = (X' * g);
unreg = (1/m) * g;
reg2 = ((lambda/m) * theta);
grad = unreg .+ reg2;




% =============================================================

end
