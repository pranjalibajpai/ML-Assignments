function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
%J=C/2*m;

t=2*m;
H=X*theta-y;
C=H.*H;
C=H.^2;
S=sum(C);
J=S/t;


%theta1=theta(2,1);
%theta0=theta(1,1);
%H=theta0+X.*theta1;
%T=H-y;
%S=T.*T;
%C=S./t;
%J=sum(C,1);
%Z=X.*theta(2,1);
%W=Z+theta(1,1);
%K=(W-y);
%K=K.*K;
%J=K./t;






% =========================================================================

end
