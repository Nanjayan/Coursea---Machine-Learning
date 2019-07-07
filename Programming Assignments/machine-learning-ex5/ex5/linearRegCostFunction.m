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


h_x=X*theta;
diff=sum((h_x-y) .* (h_x-y));
first_term = diff/(2*m);
second_term=(lambda/(2*m))*(sum(theta(2:end).*theta(2:end)));

J=first_term + second_term;


reg_first_term=(sum((h_x-y).* (X)))'/m;
reg_second_term=(lambda/m)*theta;
reg_second_term(1)=0;


grad=reg_first_term+reg_second_term;








% =========================================================================

grad = grad(:);

end
