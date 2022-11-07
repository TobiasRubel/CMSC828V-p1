function [w,f,normgrad] = SGD2(fun,gfun,Y,w,bsz,kmax,tol)
    % this one uses linesearch
    jmax = 1000;
    CGimax = 10; % max number of CG iterations
    rho = 0.1;
gam = 0.9;

	[n,~] = size(Y);
	I = 1:n;
	f = zeros(kmax + 1,1);
	f(1) = fun(I,w);
	normgrad = zeros(kmax,1);
	a = 1;
	for k = 1 : kmax
		    IH = randperm(n,bsz);
    		Mvec = @(v)Hvec(IH,w,v);
		Ig = randperm(n,bsz);
		b = gfun(Ig,w);
		normgrad(k) = norm(b);
		s = CG(Mvec,-b,-b,CGimax,rho);
		w = w - a*b;
		f(k + 1) = fun(I,w);
		fprintf('k = %d, s = %ssize, f = %d, ||g|| = %d\n',k,s,f(k+1),normgrad(k));     
        for j = 0 : jmax
			wtry = w + a*s;
			f1 = fun(Ig,wtry);
			if f1 < f0 + a*aux
				% fprintf('Linesearch: j = %d, f1 = %d, f0 = %d, |as| = %d\n',j,f1,f0,norm(a*s));
			    break;
			else
			    a = a*gam;
			end
		    end
		    if j < jmax
			w = wtry;
		    else
			nfail = nfail + 1;
		    end
		    f(k + 1) = fun(I,w);
		    if mod(k,100)==0
			fprintf('k = %d, a = %d, f = %d, ||g|| = %d\n',k,a,f(k+1),normgrad(k));
		    end
		    if nfail > nfailmax
			f(k+2:end) = [];
			normgrad(k+1:end) = [];
			fprintf('stop iteration as linesearch failed more than %d times\n',nfailmax);
			break;
		    end
		
	        if normgrad(k) < tol
        		break;
    		end
	end
end
