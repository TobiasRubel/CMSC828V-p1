function [w,f,normgrad] = SGD3(fun,gfun,Y,w,bsz,kmax,tol)
    %% this one changes stepsizes 
	[n,~] = size(Y);
	I = 1:n;
	f = zeros(kmax + 1,1);
	f(1) = fun(I,w);
	normgrad = zeros(kmax,1);
	%% set stepsize
    a = 1;
    m = 1000;
    for k = 1 : kmax
        %% update stepsize at every turn
		a = a*m/(m+k);
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
