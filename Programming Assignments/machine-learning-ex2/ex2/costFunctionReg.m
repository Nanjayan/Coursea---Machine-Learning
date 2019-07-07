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


for i =1:m
   h_x=1/(1+(exp(-(X(i,:)*theta))));
   J=J-((y(i)*log(h_x))+((1-y(i))*log(1-h_x)));
end
J=J/m;

n=length(X(1,:));
theta_2=0;
for i=2:n
   theta_2=theta_2+(theta(i)*theta(i));
end
theta_2=theta_2*lambda;
regularizer=theta_2/(2*m);
J=J+regularizer;


for i=1:m
        h_x=1/(1+(exp(-(X(i,:)*theta))));
        difference=(h_x-y(i));
        grad=grad+((difference*X(i,:))');
end
grad=grad/m;

regulariser_grad=(lambda/m)*theta;
regulariser_grad(1)=0;

grad=grad+regulariser_grad;






% =============================================================

end
