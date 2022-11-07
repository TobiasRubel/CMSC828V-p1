function [w,f,normgrad] = SGD1(fun,gfun,Y,w,bsz,kmax,tol)
    % this one changes stepsizes 
	[n,~] = size(Y);
	I = 1:n;
	f = zeros(kmax + 1,1);
	f(1) = fun(I,w);
	normgrad = zeros(kmax,1);
	%set stepsize
    a = 1;
    m0 = 8;
	m = m0;
	q = 1;
	for k = 1 : kmax
		%update stepsize
		if mod(k,m) == 0
			a = (2^(-q))*a;
			m = int64(m0*2^q/q);
            q = q+1;
        end
		Ig = randperm(n,bsz);
		b = gfun(Ig,w);
		normgrad(k) = norm(b);
		w = w - a*b;
		f(k + 1) = fun(I,w);
	        if normgrad(k) < tol
        		break;
    		end
	end
		fprintf('k = %d, a = %a, f = %d, ||g|| = %d, m=%d\n',k,a,f(k+1),normgrad(k),m);      
end
