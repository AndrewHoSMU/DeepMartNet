N = [450, 900, 1350, 1800];
Relative_error_N = [.2061, .1730, .1206, .1197];
Eigenfunction_error_N = [.05028, .04877, .04659, .04825];
log_N = log10(N);
log_RE_N = log10(Relative_error_N);
log_EE_N = log10(Eigenfunction_error_N);
fitresult_RE_N = fit(N', Relative_error_N', 'power1')
fitresult_EE_N = fit(N', Eigenfunction_error_N', 'power1')

figure;
title("Eigenvalue and eigenfunction relative error for various $N$", "Interpreter", "latex")
subtitle("Holding $M$ constant at 10,000", "Interpreter", "latex")
legend("Eigenvalue error", "Eigfunction error")
hold on;
plot(N, Relative_error_N, 'o');
plot(N, Eigenfunction_error_N, 'o');
plot(fitresult_RE_N, '-');
plot(fitresult_EE_N, '-');
hold off;

M = [10000, 15000, 20000, 25000, 30000];
Relative_error_M = [.1197, .0809, .0686, .0441, .0091];
Eigenfunction_error_M = [.04825, .08724, .04786, .04670, .04633];
log_M = log10(M);
log_RE_M = log10(Relative_error_M);
log_EE_M = log10(Eigenfunction_error_M);
fitresult_RE_M = fit(M', Relative_error_M', 'power1');
fitresult_EE_M = fit(M', Eigenfunction_error_M', 'power1');

figure;
title("Eigenvalue and eigenfunction relative error for various $M$", "Interpreter","latex")
subtitle("Holding $N$ constant at 1,800", "Interpreter","latex")
legend("Eigenvalue error", "Eigfunction error")
hold on;
plot(M, Relative_error_M, 'o');
plot(M, Eigenfunction_error_M, 'o');
plot(fitresult_RE_M, '-');
plot(fitresult_EE_M, '-');
hold off;