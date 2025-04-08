function dy = testodeF_2D(t,y)
global CC F1 F2 w_f1 w_f2 M
f1=F1*sin(2*pi*w_f1*t);
f2=F2*sin(2*pi*w_f2*t);
A00=zeros(2);
FF=[A00;inv(M)]*[f1;f2];
dy=CC*y+FF;