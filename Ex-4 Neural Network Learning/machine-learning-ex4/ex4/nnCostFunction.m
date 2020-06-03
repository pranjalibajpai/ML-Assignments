function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%PART 1
%add ones to x
X=[ones(size(X,1),1),X];
k=num_labels;
for i=1:m,
  tempX=X(i,:);
  a1=tempX';
  z2=Theta1*a1;
  a2=sigmoid(z2);
  a2=[1;a2];
  %a2=[ones(size(a2,1),1),a2];
  z3=Theta2*a2;
  a3=sigmoid(z3);
  h=a3;
  tempy=zeros(k,1);
  tempy(y(i))=1;
  J=J+ (-1/m)* sum( tempy.*log(h) + (1-tempy) .*log(1-h));
endfor
%cost with regularization
t1=Theta1(:,(2:end));
t2=Theta2(:,(2:end));
tempJ= (lambda/(2*m)) *(sum(sum(t1.^2))+ sum(sum(t2.^2)) );
J =J+tempJ;

%Part 2: BACKPROPAGATION
bd1 = 0 ;
bd2 = 0 ;
for i = 1 : m ,
  %feedforward to calculate a1,a2,a3
  tempX = X ( i , : ) ;
  a1 = tempX' ;
  z2 = Theta1 * a1 ;
  a2 = sigmoid ( z2 ) ;
  a2 = [ 1 ; a2 ] ;
  z3 = Theta2 * a2 ;
  a3 = sigmoid ( z3 ) ;
  h = a3 ;
  %backpropagate to compute delta
  tempy = zeros ( k , 1 ) ;
  tempy ( y ( i ) ) = 1 ;
  d3 = a3 - tempy ;
  d2 = t2'*d3.*sigmoidGradient ( z2 ) ;
 % d2 = d2 ( 2 : end ) ;
  
  bd1 = bd1 + d2 * a1';
  bd2 = bd2 + d3 * a2';
endfor
  Theta1_grad = (1/m) * bd1 ;
  Theta2_grad = (1/m) * bd2 ;

  
%gradient with regularized 

%grad1=Theta1_grad(:,(2:end));
%grad2=Theta2_grad(:,(2:end));
%grad1 = (lambda/m).*grad1;
%grad2 = (lambda/m).*grad2;
%add column of 0

%grad1 = [zeros(size(grad1,1),1),grad1];
%grad2 = [zeros(size(grad2,1),1),grad2];

%Theta1_grad =[Theta1_grad(:,1),grad1];
%Theta2_grad =[Theta2_grad(:,1),grad2];;

bias1= Theta1_grad(:,1);
bias2= Theta2_grad(:,1);

reg1= Theta1_grad(:,(2:end));
reg2= Theta2_grad(:,(2:end));

t1=Theta1(:,(2:end));
t2=Theta2(:,(2:end));

reg1= reg1 + (lambda/m).*t1;
reg2= reg2 + (lambda/m).*t2;

Theta1_grad= [bias1,reg1];
Theta2_grad= [bias2,reg2];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
