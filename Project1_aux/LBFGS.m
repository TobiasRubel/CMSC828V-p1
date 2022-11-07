%%
function [x,f,normgrad] = LBFGS(func,gfun,Hfun,Y,x,bsz,kmax,tol)
a = 5;
dim = size(x,1); % size of variable
n = size(Y,1); % data size
 I = 1:n;
g = gfun(I,x);
bszh = 20;
gam = 0.9; % line search step factor
jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
eta = 0.5; % backtracking stopping criterion factor
M = 20;
m = 5; % the number of steps to keep in memory
% first do steepest decend step
I = 1:n;
Ig = I;
f = zeros(kmax,1); 
normgrad = zeros(kmax,1);
a = linesearch(Ig,x,-g,g,func,eta,gam,jmax);
xnew = x - a*g;
gnew = gfun(I,xnew);
s = zeros(dim,m);
y = zeros(dim,m);
s(:,1) = xnew - x;
y(:,1) = gnew - g;
rho(1) = 1/(s(:,1)'*y(:,1));
x = xnew;
g = gnew;
nor = norm(g);
iter = 1;
k = 1;
while nor > tol && k < kmax
    Ig = randperm(n,bsz); 
    g = gfun(Ig,x);
    if iter < m
        I_m = 1 : iter;
        p = finddirection(g,s(:,I_m),y(:,I_m),rho(I_m));
    else
        p = finddirection(g,s,y,rho);
    end
    [a,j] = linesearch(Ig,x,p,g,func,eta,gam,jmax);
    if j == jmax
        p = -g;
        [a,j] = linesearch(Ig,x,p,g,func,eta,gam,jmax);
    end
    step = a*p;
    xnew = x + step;
    iter = iter + 1
    if mod(k,M) == 0
	    IH = randperm(n,bszh)
    gnew = gfun(IH,xnew);
    s = circshift(s,[0,1]);
    y = circshift(y,[0,1]);
    rho = circshift(rho,[0,1]);
    s(:,1) = step;
    y(:,1) = gnew - g;
    rho(1) = 1/(step'*y(:,1));
    end
    pt = gfun(I,x);
    x = xnew;
    g = gnew;
    f(k) = func(I,x);
    nor = norm(g);
    iter = iter + 1;
    k = k + 1
    normgrad(k) = norm(pt);
end
fprintf('L-BFGS: %d iterations, norm(g) = %d\n',iter,nor);

end
%%
function [a,j] = linesearch(Ig,x,p,g,f,eta,gam,jmax)
a = 1;
f0 = f(Ig,x);
aux = eta*g'*p;
for j = 0 : jmax
    xtry = x + a*p;
    f1 = f(Ig,xtry);
    if f1 < f0 + a*aux
        break;
    else
        a = a*gam;
    end
end
end


function p = finddirection(g,s,y,rho)
% input: g = gradient dim-by-1
% s = matrix dim-by-m, s(:,i) = x_{k-i+1}-x_{k-i}
% y = matrix dim-by-m, y(:,i) = g_{k-i+1}-g_{k-i}
% rho is 1-by-m, rho(i) = 1/(s(:,i)’*y(:,i))
m = size(s,2);
a = zeros(m,1);
for i = 1 : m
    a(i) = rho(i)*s(:,i)'*g;
    g = g - a(i)*y(:,i);
end
gam = s(:,1)'*y(:,1)/(y(:,1)'*y(:,1)); % H0 = gam*eye(dim)
g = g*gam;
for i = m :-1 : 1
    aux = rho(i)*y(:,i)'*g;
    g = g + (a(i) - aux)*s(:,i);
end
p = -g;
end
