load FPeigenval5x2ctrap
load FPerror_u5x2ctrap
load FPlearned_u5x2ctrap
load FPloss5x2ctrap
load FPtrue_u5x2ctrap
load FPerror5x2ctrap

n = length(FPeigenval5x2ctrap);
t = 1:n;

figure(1);
plot(t, FPeigenval5x2ctrap);
title("Eigenvalue \lambda");

figure(2);
semilogy(t, abs(FPerror5x2ctrap))
title("Log of relative error of eigenvalue \lambda");

figure(3);
semilogy(t, FPloss5x2ctrap);
title("Log of loss function");

m = size(FPerror_u5x2ctrap, 1);
s = linspace(-sqrt(5)*pi, sqrt(5)*pi, m);

figure(4);
plot(s, FPerror_u5x2ctrap);
title("Error of Eigenfunction u");

figure(5);
hold on
plot(s, FPlearned_u5x2ctrap);
plot(s, FPtrue_u5x2ctrap);
hold off
legend("Learned u", "True u");
title("Learned vs True Eigenfunction u");