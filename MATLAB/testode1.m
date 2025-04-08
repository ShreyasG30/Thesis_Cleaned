function dy = testode1(t, y)
    global m k  % Declare global variables before using them

    dy = zeros(2,1); % Ensure dy is a column vector of size (2x1)
    
    dy(1) = y(2);           % Velocity equation (dy/dt = velocity)
    dy(2) = (-k * y(1)) / m; % Acceleration equation (Newton’s 2nd law)
end
