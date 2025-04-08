function [x xd xdd] = BetaNewmark3(M,K,C,time,x0,v0,dt,F)
a=1/6; b=1/2;
x=x0;
xd=v0;
xdd(:,1)=inv(M)*(F(:,1)-C*xd(:,1)-K*x(:,1));

i=1;
for m1=dt:dt:time
    x(:,i+1)=inv((1/(a*dt^2))*M+(b/(a*dt))*C+K)*...
        (F(i+1)+...
        M*((1/(a*dt^2))*x(:,i)+(1/(a*dt))*xd(:,i)+(1/(2*a)-1)*xdd(:,i))...
        +C*((b/(a*dt))*x(:,i)+(b/a-1)*xd(:,i)+(b/a-2)*dt*0.5*xdd(:,i)));
    xdd(:,i+1)=(1/(a*dt^2))*(x(:,i+1)-x(:,i))-(1/(a*dt))*xd(:,i)-(1/(2*a)-1)*xdd(:,i);
    xd(:,i+1)=xd(:,i)+(1-b)*dt*xdd(:,i)+b*dt*xdd(:,i+1);
    Time(1,i+1)=m1;
    i=i+1;
end
