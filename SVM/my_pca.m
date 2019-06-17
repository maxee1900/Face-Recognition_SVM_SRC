function [Pro_Matrix,Mean_Image]=my_pca(Train_SET,Eigen_NUM)
%输入：
%Train_SET：训练样本集，每列是一个样本，每行一类特征，Dim*Train_Num
%Eigen_NUM：投影维数

%输出：
%Pro_Matrix：投影矩阵
%Mean_Image：均值图像

[Dim,Train_Num]=size(Train_SET);

 Mean_Image=mean(Train_SET,2); %各行的均值 维度的均值 
 Train_SET=bsxfun(@minus,Train_SET,Mean_Image);  %每一行的元素将去每一行的均值  

 R=Train_SET*Train_SET'/(Train_Num-1); %
    
 [eig_vec,eig_val]=eig(R);  %特征向量和特征值  
 eig_val=diag(eig_val);  % 取矩阵的对角线返回到向量
 [~,ind]=sort(eig_val,'descend');  %按降序排列 ~为排列好的向量 ind为其对应在元向量的索引
 W=eig_vec(:,ind);   %原特征向量的矩阵，重新按列排列
 Pro_Matrix=W(:,1:Eigen_NUM);   %取1：投影维数
    
