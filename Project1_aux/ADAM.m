function [w,f,normgrad] = ADAM(fun,gfun,Y,w,bsz,kmax,tol)
B1 = 0.9;
B2 = 0.999;
SSIZE = 0.1;
epsilon = 10e-8;

[n, dim] = size(Y);
I = 1:n;

normgrad = zeros(kmax,1);
f = zeros(kmax + 2,1);
v = zeros(kmax + 2,dim);
m = zeros(kmax + 2,dim);
f(1) = fun(I,w);
k = 1;
for k = 1 : kmax
while k < kmax && normgrad(k) < tol
    k = k + 1
    Ig = randperm(n,bsz);
    m(k, :) = B1*m(k-1, :) + (1-B1)*gfun(Ig, w)';
    v(k, :) = B2*v(k-1, :) + (1-B2)*(gfun(Ig, w).*gfun(Ig, w))';
    mhat = m(k, :)/(1-B1^(k));
    vhat = v(k, :)/(1-B2^(k));
    wn = (SSIZE.*mhat./(sqrt(vhat)+epsilon));
    normgrad(k-1) = norm(wn);
    w = w - wn';
    f(k) = fun(I,w);
end
fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
end
