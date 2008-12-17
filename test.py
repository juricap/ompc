
import ompc

########################## Basic Matrix Operations ############################
# from http://www.mathworks.com/products/demos/shipping/matlab/intro.html

a = mcat([1, 2, 3, 4, 6, 4, 3, 4, 5]); print a
b = a + 2; print b
disp('PRESS A KEY TO CONTINUE')
pause()

plot(b)
grid('on')
disp('PRESS A KEY TO CONTINUE')
pause()

bar(b)
xlabel('Sample #')
ylabel('Pounds')
disp('PRESS A KEY TO CONTINUE')
pause()

plot(b,'*')
axis(mcat([0, 10, 0, 10]))
disp('PRESS A KEY TO CONTINUE')
pause()

A = mcat([1, 2, 0, OMPCSEMI, 2, 5, -1, OMPCSEMI, 4, 10, -1]); print A
B = A.cT; print B
C = A * B; print C
C = A *elmul* B; print C
disp('PRESS A KEY TO CONTINUE')
pause()

X = inv(A); print X
I = inv(A) * A; print I
disp('PRESS A KEY TO CONTINUE')
pause()

print eig(A)
print svd(A)
disp('PRESS A KEY TO CONTINUE')
pause()

p = round(poly(A)); print p
print roots(p)
disp('PRESS A KEY TO CONTINUE')
pause()

q = conv(p,p); print q
r = conv(p,q); print r
disp('PRESS A KEY TO CONTINUE')
pause()

plot(r);
disp('PRESS A KEY TO CONTINUE')
pause()

whos()
disp('PRESS A KEY TO CONTINUE')
pause()

print A
disp('PRESS A KEY TO CONTINUE')
pause()

print sqrt(-1)
disp('PRESS A KEY TO CONTINUE')
pause()

############################ Matrix Manipulation ##############################
# from http://www.mathworks.com/products/demos/shipping/matlab/matmanip.html

A = mcat([8, 1, 6, OMPCSEMI, 3, 5, 7, OMPCSEMI, 4, 9, 2]); print A
disp('PRESS A KEY TO CONTINUE')
pause()
print A+2
disp('PRESS A KEY TO CONTINUE')
pause()
B = 2*ones(3)
disp('PRESS A KEY TO CONTINUE')
pause()
print B
print A*B
print A*elmul*B
disp('PRESS A KEY TO CONTINUE')
pause()

print eig(A)
disp('PRESS A KEY TO CONTINUE')
pause()
