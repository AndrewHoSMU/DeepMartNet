load FPeigenval200x2ctrap
load FPerror_u200x2ctrap
load FPlearned_u200x2ctrap
load FPloss200x2ctrap
load FPtrue_u200x2ctrap
load FPerror200x2ctrap

n = length(FPeigenval200x2ctrap);
t = 1:n;

figure(1);
plot(t, FPeigenval200x2ctrap);
title("Eigenvalue \lambda");

figure(2);
semilogy(t, abs(FPerror200x2ctrap))
title("Log of relative error of eigenvalue \lambda");

figure(3);
semilogy(t, FPloss200x2ctrap);
title("Log of loss function");

m = length(FPerror_u200x2ctrap);
s = linspace(-5*pi, 5*pi, 106);

figure(4);
plot(s, FPerror_u200x2ctrap(97:m-98));
title("Error of Eigenfunction u");

figure(5);
hold on
plot(s, FPlearned_u200x2ctrap(97:m-98));
plot(s, FPtrue_u200x2ctrap(97:m-98));
hold off
legend("Learned u", "True u");
title("Learned vs True Eigenfunction u");