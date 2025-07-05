load FPeigenval25x2ctrap
load FPerror_u25x2ctrap
load FPlearned_u25x2ctrap
load FPloss25x2ctrap
load FPtrue_u25x2ctrap
load FPerror25x2ctrap

n = size(FPeigenval25x2ctrap);
t = 1:n;

figure(1);
plot(t, FPeigenval25x2ctrap);
title("Eigenvalue \lambda");

figure(2);
semilogy(t, abs(FPerror25x2ctrap))
title("Log of relative error of eigenvalue \lambda");

figure(3);
semilogy(t, FPloss25x2ctrap);
title("Log of loss function");

m = size(FPerror_u25x2ctrap, 1);
s = linspace(-pi, pi, m);

figure(4);
plot(s, FPerror_u25x2ctrap);
title("Error of Eigenfunction u from -\pi to \pi");

figure(5);
hold on
plot(s, FPlearned_u25x2ctrap);
plot(s, FPtrue_u25x2ctrap);
hold off
legend("Learned u", "True u");
title("Learned vs True Eigenfunction u from -\pi to \pi");