function d=randnorepeat(m,n)
%生成一列在[1,n]范围内的m个不重复的整数
p=randperm(n);
d=p(1:m);
end

