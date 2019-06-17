% Face recognition on yale data---稀疏表示方法（SRC）
% by Ma Xin. 2018.6.22

clear all 
clc
close all 

%精度 测试时间 实验次数
Accu = zeros(1,10);    
Time = zeros(1,10);
for k = 1:2   
  
clearvars -except Accu Time k    %清除其他变量  

%% 读取人脸数据、降采样，形成数据集和标签集
yaleData = 'C:\Users\maxee\Desktop\matlab\yaleBExtData ';
dataDir = dir(yaleData);  %dir 列出指定路径下的所有子文件夹和子文件

label = [];
allsamples = [];
for i = 3:40 
  facefile = [yaleData,'\',dataDir(i).name];
  oneperson = dir ([facefile, '\*.pgm']); 
  Asample = [];
  
  for j = 1:length(oneperson)-1
      image = imread([oneperson(j).folder,'\',oneperson(j).name]);
      downsample = image(1:4:192, 1:4:168);
      imagedouble = double(downsample);   %转化为double型
      faceshape = reshape(imagedouble,48*42,1);  %reshape按列顺序转换，不是随机的
      Asample = [Asample,faceshape];  
      allsamples = [allsamples,faceshape];  %所有数据样本：m列样本，n行维度
      label = [label,i-2];   %所有标签集
  end 
  
   Allsample{i-2} = Asample;   %构成1*38的cell数组
end

%% 分出训练集和测试集 （p=7,13,20）
p = 13;
trainsamples = [];
testsamples = [];
trainlabels = [];
testlabels = [];

for i = 1:length(Allsample)
    m = size(Allsample{i},2);
    randse = randperm(m);
    train_one = Allsample{i}(:,randse(1:p));
    test_one = Allsample{i}(:,randse(p+1:m));
    trainlabel_one = i * ones(1,p);
    testlabel_one = i * ones(1,m-p);
    
    trainsamples = [trainsamples,train_one];
    testsamples = [testsamples,test_one];
    trainlabels = [trainlabels,trainlabel_one];
    testlabels = [testlabels,testlabel_one];

end

%% PCA降维至50,100,200
[Pro_Matrix,Mean_Image]=my_pca(trainsamples,100);
    %Pro_Matrix为投影矩阵
    train_project=Pro_Matrix'*trainsamples;
    test_project=Pro_Matrix'*testsamples;
    

%% 单位化
trainNorm = normc(train_project);
testNorm = normc(test_project);

%% 识别并计算准确率
testNum = size(testNorm,2);   %总测试数
trainNum = size(trainNorm,2);   %总训练数 

labelpre = zeros(1,testNum);   %标签预测分配内存 
classnum = length(Allsample);

 h = waitbar(0,'please wait...');   %进度条函数，数值在0~1
  
 for i = 1:testNum
    t1 = clock;
    xp = SolveHomotopy_CBM_std(trainNorm,testNorm(:,i),'lambda',0.01);
    %针对第i个测试样本，用Homotopy方法求优化解，稀疏表示优化算法之一，用到了L1和L2范数
    r = zeros(1, classnum);
    
    for j = 1:classnum
        %将xp中元素处理后归于xn.其中xp中与j类对应的元素保留，其他的为0
        xn = zeros(trainNum,1);                   
        index = (j==trainlabels);   %index为bool变量
        xn(index) = xp(index);   %index与xp中位置自动对齐 
    
        r(j) = norm((testNorm(:,i) - trainNorm * xn));   %误差的2范数
    end
    
    [~,pos] = min(r);    %返回r中误差最小的位置 
    labelpre(i) = pos;
    t2 = clock;
    testtime(i) = etime(t2,t1);
    per = i / testNum;
    waitbar(per,h,sprintf('第%d次实验：%2.0f%%',k,per*100));   %显示每个样本测试进度条    
 end

close(h);
Avtest_time = mean(testtime,2);
accuracy = sum(labelpre == testlabels) / testNum;

fprintf('第%d次实验识别率为：%5.2f%%\n\n',k,accuracy*100);
fprintf('第%d次实验平均每幅图片分类所用时间为：%6.4fs\n',k,Avtest_time);

%% 保存第K次实验精度与测试时间
Accu(k) = accuracy;    
Time(k) = Avtest_time;

end



