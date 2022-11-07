function [w,f,normgrad] = Snesterov(fun,gfun,Y,w,bsz,kmax,tol)
[n, dim] = size(Y);
I = 1:n;
f = zeros(kmax + 1,1);
v = zeros(kmax + 1,dim);
f(1) = fun(I,w);
normgrad = zeros(kmax,1);
a = 1;
for k = 1 : kmax
    mu = 1 - 3/(5+k);
    Ig = randperm(n,bsz);
    v(k+1, :) = mu.*v(k, :)' - a*gfun(Ig,w+mu.*v(k, :)');
    normgrad(k) = norm(v(k+1, :));
    w = w + v(k+1, :)';
    f(k + 1) = fun(I,w);
    
    if normgrad(k) < tol
        break;
    end
end
fprintf('k = %d, a = %d, f = %d, ||g|| = %d\n',k,a,f(k+1),normgrad(k));
end
