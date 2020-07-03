function [RM,Pred_RM] = SLCRF(par)
%% set path
addpath(genpath('chuanyang_cvprcode'));
addpath(genpath('minFunc_2009'));
%%
load('data\Indian_Pines\Indian_pines_gt.mat');
gt=indian_pines_gt;
datainfo.N_Cat = max(max(gt));
datainfo.band = 224;%bands num
datainfo.range = zeros(datainfo.N_Cat,2);
num_all=0;
for i=1:datainfo.N_Cat
    te=find(gt==i);
    num=length(te);
    datainfo.range(i,1) = num_all + 1;
    datainfo.range(i,2) = num_all + num;
    num_all=num_all+num;
end
par.train_rate = 0.1;
par.un_rate = 0.5;
datainfo.train_rate = par.train_rate;    %training samples
datainfo.un_rate = par.un_rate;        %labeled samples
%%
%construct unsupervised train and test samples
is_build_dataset = 0;
if is_build_dataset == 0
    unsupervised_data = cell(datainfo.N_Cat,1);
    train_data = cell(datainfo.N_Cat,1);
    for i = 1:datainfo.N_Cat
        category_range = datainfo.range(i,:);
        range1 = category_range(1);
        range2 = category_range(2);
        category_number = range2 - range1 + 1;
        num_training_sample = round(datainfo.train_rate*category_number);
        train_data_ele = range1 +  randnorepeat(num_training_sample,range2 - range1 +1) - 1;
        
        size_train_datai = size(train_data_ele,2);
        num_un_sample = round(datainfo.un_rate);
        rand_data_pos = randnorepeat(num_un_sample,size_train_datai);
        rand_element = train_data_ele(rand_data_pos);
        unsupervised_data{i} = rand_element;
        train_data{i} = train_data_ele;
    end
    name_un = strcat('unsupervised_data',num2str(datainfo.train_rate),'_',num2str(datainfo.un_rate));
    name_train = strcat('train_data_',num2str(datainfo.train_rate),'_',num2str(datainfo.un_rate));
    
    sub_file = './data/Indian_Pines/';
    save([sub_file name_un '.mat'],'unsupervised_data');
    save([sub_file name_train '.mat'],'train_data');
else
    load('.\data\Indian_Pines\unsupervised_data0.1_0.5.mat');
    load('.\data\Indian_Pines\train_data_0.1_0.5.mat');
end
%% learn unsupervised 
%  normalize feature
load('data\Indian_Pines\Indian_pines_corrected.mat');  
load('.\Feature\Indian_Pines\hidden_1.mat');
load('.\Feature\Indian_Pines\Index_indian.mat');
load('.\Feature\Indian_Pines\Feature_CW1.mat');
load('.\Feature\Indian_Pines\Feature_Cb1.mat');
load('.\Feature\Indian_Pines\Feature_W2.mat');
load('.\Feature\Indian_Pines\Feature_b2.mat');

index_un = [];
index_train = [];
for i = 1:size(train_data,1)
    t = unsupervised_data{i,1};
    index_un = [index_un',t]';    
    tt = train_data{i,1};
    index_train = [index_train',tt]';
end

train_X = CAE3D_fea(index_train,:)';
train_loc = CAE3D_loc(index_train,:)';
[n,~] = size(CAE3D_fea);
%% 
[dim_fea,num_train_images] = size(train_X);
%% parameter
para.alpha   = power(10, par.alpha);
para.beta   = power(10,par.beta);
para.gama   = power(10,par.gama);
para.sigama   = power(10,par.sigama);
para.lamda1  = power(10,par.lamda1); 
para.lamda2  = power(10,par.lamda2); 

para.k = par.k;

para.miu = power(10,-2);%Penalty parameter for Lagrange

iteration = 20;

%%

is_initialize_Z = 0;
if is_initialize_Z == 0
    Z_rand = rand(num_train_images,num_train_images)*(1/sqrt(num_train_images));  
    Z_rand = Z_rand ./ repmat(sqrt(sum(Z_rand.^2,2)), 1, size(Z_rand,2));
    Z = Z_rand;
else
    load('.\result\Indian_Pines\Z.mat');
end

%M is the subsitution of W
is_initialize_M = 0;
if is_initialize_M == 0
    M_rand = rand(num_train_images,num_train_images)*(1/sqrt(num_train_images));  
    M_rand = M_rand ./ repmat(sqrt(sum(M_rand.^2,2)), 1, size(M_rand,2));
    M = M_rand;
else
    load('.\result\Indian_Pines\M.mat');
end

T = zeros(num_train_images,num_train_images);

h2 = 1;

S1 = abs(Z+Z')/2;
[S2] = Build_Graph(train_X,train_loc,para);

S = S1 + para.gama*S2;
%% Main body of Algorithm 1
for i = 1:iteration
      Z  = Update_Z(train_X,Z,T,M,h2,S,para);
      M  = Update_M(Z,T,para);
      T = T + para.miu * (Z - M);
      S1 = abs(Z+Z')/2;
      S = double(S1)+para.gama*double(S2);
      h2 = Update_h2(S,h2,para);
      
      H1 = max(W2*train_X+repmat(b2,1,num_train_images),0);
      [W2,b2] = Update_W(train_X,W2,b2,Z,datainfo,para);
      save(['.\result\Indian_pines\W2.mat'],'W2');
      save(['.\result\Indian_pines\b2.mat'],'b2');
      RM = (max(W2*CAE3D_fea'+repmat(b2,1,n),0))'; %relu activation function
      Pred_RM = (CW1 *RM' +repmat(Cb1,1,n))';
      save(['.\result\Indian_pines\RM.mat'],'RM');
      save(['.\result\Indian_pines\Pred_RM.mat'],'Pred_RM');
      
end
end

%%
function [S2] = Build_Graph(image_feature,train_loc,para)
all_feature = image_feature;
num_image = size(all_feature,2);
k = para.k; %neighbor parameter
%distance
loc=double(train_loc');
dist=sqrt(loc.^2*ones(size(loc'))+ones(size(loc))*(loc').^2-2*(loc*loc'));

is_build_edges = 0;
if is_build_edges == 0
    adj_matrix = zeros(num_image,num_image);
    for ii = 1:num_image
        [t1,t2] = sort(dist(ii,:),'ascend');
        for jj = 2:k+1 
            adj_matrix(ii,t2(jj)) = 1;
            adj_matrix(t2(jj),ii) = 1;
        end
    end
    
    edges2=[];
    for i=1:num_image
        indext=[];
        ind=find(adj_matrix(i,:)==1);
        for j=1:length(ind)
            indj=find(adj_matrix(ind(j),:)==1);
            indext=[indext,indj];
        end
        indext=[indext,ind];
        indext=indext((indext>i));
        indext=unique(indext);
        if(~isempty(indext))
            ed=ones(length(indext),2);
            ed(:,2)=i*ed(:,2);
            ed(:,1)=indext;
            edges2=[edges2;ed];
        end
    end
    sub_file = '.\Feature\Indian_Pines\';
    save([sub_file 'edges2.mat'],'edges2');
else
    load('.\Feature\Indian_Pines\edges2.mat');
end
% compute affinity matrix
weights = makegraphweights2(edges2,dist,para);
S2 = adjacency(edges2,weights,num_image);% weighted adjacency matrix
save(['.\Feature\Indian_Pines\' 'S2.mat'],'S2');
end

%%
function weights = makegraphweights2(edges2,dist,para)
num = size(edges2,1);
weights = zeros(num,1);

for ii = 1:num
    dis=dist(edges2(ii,1),edges2(ii,2));
    weights(ii,1)=exp((-1)*dis/para.sigama);   
end
end

%%
function Z  = Update_Z(X,Z,T,M,h2,S,para)
%configure minFunc
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 10;

n = size(Z,1);
Z_initial = Z(:);
[Z, cost] = minFunc( @(Z) compute_Z_cost_grad(X,Z,T,M,h2,S,para), Z_initial, options); 
Z = reshape(Z,n,n);
end

function [cost,grad] = compute_Z_cost_grad(X,Z,T,M,h2,S,para)

n=size(X,2);
Z=reshape(Z,n,n);

%compute the cost
part1 = para.lamda1 * power(norm(X - X*Z,'fro'),2);

part2 = para.lamda2 *para.sigama*h2*sum(S(:));
part3 = 0.5 * para.miu * power(norm(Z - M + T/para.miu,'fro'),2);

cost = part1 + part3 + part2;

%compute the grad
temp1 = para.lamda1 *2*X'*(X*Z-X);
temp2 = 0.5*para.lamda2 *para.sigama*S;
temp3 = para.miu * (Z - M+T/para.miu);

grad = temp1 + temp3 + temp2;
grad = grad(:);
end

%%
function [W1,b1] = Update_W(X,W1,b1,Z,datainfo,para)
%configure minFunc
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 10;

parp=parpool('local', 4);
W_RICA1 = W1;
b_RICA1 = b1;
size_W = size(W_RICA1,1);
dim_X = size(X,1);
W_initial = W_RICA1(:);
b1_initial = b_RICA1(:);
Initial = [W_initial;b1_initial];
W = cat(2,W1,b1);
[W, cost_W] = minFunc( @(W) compute_W_cost_grad(W, X,size_W, dim_X, Z,datainfo,para), Initial, options);
W = reshape(W,size_W,dim_X+1);
b1 = W(:,dim_X+1);
W1 = W(:,1:dim_X);
delete(parp);
end

function [cost,grad] = compute_W_cost_grad(W1, X, size_W,dim_X, Z,datainfo,para)
W1 = reshape(W1,size_W,dim_X+1);

W = W1(:,1:dim_X);
b = W1(:,dim_X+1);
n = size(Z,1);
I = eye(n,n);

H1 = max(W*X+repmat(b,1, n),0);
part3 = para.lamda1 * power(norm(H1 - H1*Z,'fro'),2);

cost = part3;

delta_W1 = delta_relu(W*X+repmat(b,1, n));
delta_b1 = delta_relu(W*X+repmat(b,1, n));
temp3 = 2 * para.lamda1 * (H1 - H1*Z - H1*Z'- H1*Z*Z').*delta_W1*X';
grad_W = temp3;

temp3_b = mean(2 * para.lamda1 * (H1 - H1*Z - H1*Z'- H1*Z*Z').*delta_b1,2);

grad_b = temp3_b;

grad = cat(2,grad_W,grad_b);
grad = grad(:);
end

function output = delta_relu(x)
    output = x;
    output(x>=0) = 1;
    output = max(output,0);
end
%%
function M = Update_M(Z,T,para)
%optimization of M
n = size(Z,1);
[U,Sigma,V] = svd(Z + (T/para.miu));
Omega_Sigma = zeros(n,n);

for i = 1:n
    Omega_Sigma(i,i) = max(Sigma(i,i)-(para.beta/para.miu),0);
end

M= U*Omega_Sigma*V';
end
%%
function h2  = Update_h2(S,h2,para)
%configure minFunc
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 10;

dim = size(h2,1);
h2_initial = h2(:);
[h2, cost] = minFunc( @(h2) compute_h2_cost_grad(S,h2,para), h2_initial, options); 
h2 = reshape(h2,dim,1);
end 
%%
function [cost,grad] = compute_h2_cost_grad(S,h2,para)

%compute the cost
cost=para.lamda2 * para.sigama*h2*sum(S(:));
%compute the grad
grad = para.lamda2*sum(S(:));
grad = grad(:);
end   