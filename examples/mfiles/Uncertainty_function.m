
disp(' Economics of Motion Perception');
disp(' ');
disp('Some notes for the start:');
disp('what you would normally write as:');
disp('  f(x) = 2*x + 1,');
disp('can since MATLAB >7 be written as:');
disp('  f = @(x) 2*x + 1,');
disp('the by evaluating at x=7, one gets:');
f = @(x) 2*x + 1
disp(sprintf('f(7) = %f',f(7)));


disp('Let''s start with parameters');

L1 = 0.0055;        %0.0055;
L2 = 0.001;         %0.00096;
L3 = 0.2;           %0.2
L4 = 0.8;           %0.8

disp(sprintf(' L1 = %f\n L2 = %f\n L3 = %f\n L4 = %f\n', L1, L2, L3, L4));
disp('PRESS ANY KEY TO CONTINUE');
pause

disp('The first definition of a anonymous function follows.');
disp('The spatio-temporal uncertainty function is ');
disp('a sum of uncertainty of spatial and temporal distance!');
U_ST = @(T_,S_) (L1./S_ + L3.*S_) + (L2./T_ + L4.*T_);                   % (1)
disp(U_ST);

t = logspace(-2,0);
s = logspace(-2,1);
[T_0,S_0] = meshgrid(t, s)
figure(1);
title('The Spatiotemporal Uncertainty Function.')
surfc(T_0,S_0,U_ST(T_0,S_0));
set(gca,'XScale','log'); xlabel('log(temporal distance)');
set(gca,'YScale','log'); ylabel('log(spatial distance)');

disp('PRESS ANY KEY TO CONTINUE');
pause

disp('the optimum (global minimum)');
O = fminsearch(@(x_) U_ST(x_(1),x_(2)), [0.1,0.1]);
hold on;
plot3(O(1),O(2),U_ST(O(1),O(2)),'ro');
hold off;

disp('PRESS ANY KEY TO CONTINUE');
pause

%                    The Local Optimization
% the hyperbola is the solution of the "Local optimization" configuration,
% the solution for
%                   dUS*v + dUT = 0, S = v*T                             % (2)

% the solutions
S = @(T_, v_) sqrt(L1.*v_ ./ (L3.*v_ + L4 - L2.*T_.^-2));
T = @(S_, v_) sqrt(L2 ./ (L3.*v_ + L4 - L1.*v_.*S_.^-2));
% speedlines
Ss = @(T_,v_) v_.*T_;
% and the intersections of each hyperbola and speedline pair for a specific v
Tinter = @(v_) sqrt( (L2.*v_ + L1)./(v_.*(L3.*v_+L4)) );

figure(1);
gcf;
v = logspace(-2,2,15);
for velocity = v,
    % plot a hyperbola
    temp_t = logspace(log10(T(10,velocity)),0,100);
    loglog(temp_t,S(temp_t,velocity)); hold on;
    % and the corresponding speed line
    plot(t,Ss(t,velocity),'Color',[0.7,0.7,0.7]);
    % mark the intersection
    ti = Tinter(velocity);
    plot(ti,Ss(ti,velocity),'rx','LineWidth',2,'MarkerSize',10);
end
axis([0.01, 1, 0.01, 10]);

% plot the "Local optimization" hyperbola
disp('The solution of the "Local Optimization" configuration can be then shown');
disp('PRESS ANY KEY TO PLOT');
pause

plot(Tinter(v),Ss(Tinter(v),v),'k','LineWidth',2);

disp('PRESS ANY KEY TO CONTINUE');
pause


%--------------------------- "Total integration" -------------------------------
% the solution for dUS*ve + dUT = 0, where ve(v) = Integral(p(v)*v,[0,Inf])

% slow from (Dong, Atick 1995)
%------------------------------------------------------------------------
% parameters of the slow speed asumption distribution:
%   P(v)=1/(v+v0)^n
%
% for Dong, Atick data, n=3.7, v_m (average velocity) = 0.6  
%    ==> v0 = v_m*(n-2) = 0.6*1.7 = 1.02
%------------------------------------------------------------------------
v_m     = 0.6;
n       = 3.7;
v_0     = v_m*(n-2);
probability_dong = @(x_) (v_0)^(n-1) .*(n-1) ./((x_+v_0).^n);

% and similarily with the gaussian function
sigma2 = 2.0;
probability_gauss = @(x_) 1./sqrt(2*pi*sigma2).*exp(-(x_-v_0).^2 ./(2*sigma2));

% ve(v) = Integral(p(v)*v,[0,Inf])
aInf = 1000;
ve_dong = quad( @(x_) probability_dong(x_).*x_, 0, aInf)
ve_gauss = quad( @(x_) probability_gauss(x_).*x_, 0,1) ...
            + quad( @(x_) probability_gauss(x_).*x_, 1,aInf)

% for the solution itself we can reuse the orginal solution for 
% the local optimization model and substitute v = ve
Sti_dong = S(t,ve_dong);
Sti_gauss = S(t,ve_gauss);

disp('For the Dong probability:');
disp('PRESS ANY KEY TO PLOT');
pause
subplot(121);
loglog(Tinter(v),Ss(Tinter(v),v),'k','LineWidth',2); hold on;
%loglog(t,Sti_dong,'Color',[0.4,0.4,0.4],'LineWidth',2);
temp_t = logspace(log10(T(10,ve_dong)),0,100);
loglog(temp_t,S(temp_t,ve_dong),'Color',[0.4,0.4,0.4],'LineWidth',2);
axis([0.01, 1, 0.01, 10]);

disp('For gaussian:')
disp('PRESS ANY KEY TO PLOT');
pause
subplot(122);
loglog(Tinter(v),Ss(Tinter(v),v),'k','LineWidth',2); hold on;
%loglog(t,Sti_gauss,'Color',[0.7,0.7,0.7],'LineWidth',2);
temp_t = logspace(log10(T(10,ve_gauss)),0,100);
loglog(temp_t,S(temp_t,ve_gauss),'Color',[0.4,0.4,0.4],'LineWidth',2);
axis([0.01, 1, 0.01, 10]);

disp('PRESS ANY KEY TO CONTINUE');
pause

% and finally the "Weighted Model"

% the probability distribution stay the same, we use the same parameters,
u_a = 0.1;
C_T = 0.3;    % C_T = [0,sqrt(2)-1]
C_S = 1 - ((C_T + 1)^2)/2;
% the left and the right side of the integration interval

v_a = @(v_) -u_a + v_.*(1-C_S)./(1+C_T);
v_b = @(v_)  u_a + v_.*(1+C_S)./(1-C_T);

% REMOVE !!!!!!!!!!!
% just for now I need something that grow linearly and the left side doesn't
% get negative
% v_a = @(v_) v_.*2;
% v_b = @(v_) 0.1 + v_.*4;

disp('This time the ');

v = logspace(-2,1);

% APPLY is a helper function that applies a function on each element of a matrix
% the implementation is at the bootom of this file
w_dong = @(v_) apply(@(v_) quad(probability_dong, v_a(v_), v_b(v_)),v_);
w_gauss = @(v_) apply(@(v_) quad(probability_gauss, v_a(v_), v_b(v_)),v_);

g_dong = @(v_) w_dong(v_).*v_ + (1 - w_dong(v_)).*ve_dong;
g_gauss = @(v_) w_gauss(v_).*v_ + (1 - w_gauss(v_)).*ve_gauss;

% prepare the velocity vector, we are interested in a curve crossing
% each speedline at a position

disp('For dong:')
disp('PRESS ANY KEY TO PLOT');
pause
Tw = @(v_,g_) sqrt( (L1.*g_(v_) + L2.*v_.^2)./(L3.*g_(v_) + L4) ) ./ v;
Tw_dong = Tw(v,g_dong);
subplot(121); plot(Tw_dong,v.*Tw_dong,'Color',[0.0,0.4,0.4],'LineWidth',2);
disp('For gaussian:')
disp('PRESS ANY KEY TO CONTINUE');
pause
Tw_gauss = Tw(v,g_gauss);
subplot(122); plot(Tw_gauss,v.*Tw_gauss,'Color',[0.0,0.7,0.7],'LineWidth',2);


disp('The equivalence contours are circles')
disp('PRESS ANY KEY TO CONTINUE');
pause
%phi = @(p_,v_,g_) sqrt( (L1.*g_(v_) + L2.*v_.^2)./(v_.^2.*(L3.*g_(v_) + L4 - p_)) );
Tw_gen = @(p_,v_,g_) sqrt( (L1.*g_(v_) + L2.*v_.^2) ...
    ./(v_.^2.*(L3.*g_(v_) + L4 - p_)) );

% just as the paper says, prepare a circle in cartesian coordinates
ang = linspace(0,2*pi,length(v));
Rs = linspace(0.3,5.0,12);
Rs = linspace(0.3,5.0,12);

% subplot(121);
% ps = linspace(-2.0,1.5,20);
% for p = ps,
%     teq = Tw_gen(p,v,g_dong);
%     loglog(teq,v.*teq,'m');
%     drawnow;
% end
% 
% vs = linspace(0.1,200,20);
% teq = Tw_gen(0,vs,g_dong);
% for speed = vs,
%     loglog(teq,Ss(teq,speed),'g');
%     drawnow;
% end

kp = 1.0;
kv = 1.4;

subplot(121);
for R = Rs,
    % centered at p = 0, v = ve
    ps = R.*cos(ang);
    vs = R.*sin(ang);
    % from 1/kv .*log(v./ve) - move the centre to the ve
    vs =  ve_dong.*exp( kv.*vs );
    B_pos = @(v_) L3.*g_dong(v_) + L4;
    ppos = find(ps > 0);
    Bs = ones(1,length(ang));
    Bs(ppos) = B_pos(vs(ppos));
    Ps = Bs .*( 1 - exp(-kp.*ps) );
    teq = Tw_gen(Ps,vs,g_dong);
    loglog(teq,vs.*teq,'r');
    drawnow;
end

subplot(122);
for R = Rs,
    % centered at p = 0, v = ve
    ps = R.*cos(ang);
    vs = R.*sin(ang);
    % from 1/kv .*log(v./ve) - move the centre to the ve
    vs =  ve_gauss.*exp( kv.*vs );
    B_pos = @(v_) L3.*g_gauss(v_) + L4;
    ppos = find(ps > 0);
    Bs = ones(1,length(ang));
    Bs(ppos) = B_pos(vs(ppos));
    Ps = Bs .*( 1 - exp(-kp.*ps) );
    teq = Tw_gen(Ps,vs,g_gauss);
    loglog(teq,vs.*teq,'r');
    drawnow;
end

disp('           GAME OVER!');
disp('Thank you for bearing with us and hope you''ll fly with us the next time.');
