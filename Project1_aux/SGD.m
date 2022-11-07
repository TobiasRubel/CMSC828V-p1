function [w,f,normgrad] = SGD(fun,gfun,Y,w,bsz,kmax,tol)
    % this one has fixed stepsize
	[n,~] = size(Y);
	I = 1:n;
	f = zeros(kmax + 1,1);
	f(1) = fun(I,w);
	normgrad = zeros(kmax,1);
	a = 1;
	for k = 1 : kmax
		Ig = randperm(n,bsz);
		b = gfun(Ig,w);
		normgrad(k) = norm(b);
		w = w - a*b;
		f(k + 1) = fun(I,w);
	        if normgrad(k) < tol
        		break;
    		end
	end
		fprintf('k = %d, a = %ssize, f = %d, ||g|| = %d\n',k,a,f(k+1),normgrad(k));      
end
