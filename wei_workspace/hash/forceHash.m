

str = char("as");


guess = char("abasdfadsfasdfsdafsdafdsaf");

figure(1)

while 1
    % R -> r swap
    B = zeros(200,1);
    for i = 1:2000
        
        %R = ceil(rand() * length(guess));
        % 97 to  122- z
        R = ceil(rand() * 26 + 96);
        r = ceil(rand() * length(guess));
        %z = guess(R);
        %guess(R) = guess(r);
        guess(r) = char(R);
        %guess(r) = z;
        B(i) = diffHash(md5(guess),md5(str));
       % pause(.01);
    end
    disp("done, most matching at beginning:");
    disp(max(B))
    plot(B)
    pause(2)
end

function d = diffHash(a, b)
    N = length(a);
    d = 0;
    for i = 1:N
        if a(i) == b(i), d = d+1; else return; end
    end
end