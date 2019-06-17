% Face recognition on yale data---SVM方法
% by Ma Xin. 2018.6.22

clear all 
clc
close all

%% 读取人脸数据、降采样

yaleData = 'C:\Users\maxee\Desktop\matlab\yaleBExtData ';
dataDir = dir(yaleData);  %dir 列出指定路径下的所有子文件夹和子文件
 
 
for i = 3:40 
  facefile = [yaleData,'\',dataDir(i).name];
  oneperson = dir ([facefile, '\*.pgm']); 
  Asample = []; 
  for j = 1:length(oneperson)-1
      image = imread([oneperson(j).folder,'\',oneperson(j).name]);
      downsample = image(1:4:192, 1:4:168);
      imagedouble = double(downsample);   %转化为double型
      faceshape = reshape(imagedouble,1,48*42);  %reshape按列顺序转换，不是随机的
      Asample = [Asample;faceshape];   %m行样本，n行维度
  end 
  
   Allsample{i-2} = Asample;   %构成1*38的cell数组
   
end

%% 构造训练集合（p=7,13,20）
p = 13;
TrainData = [];
TestData = [];
TrainLabel = [];
TestLabel = [];

for i = 1:length(Allsample)
    % traindata and testdata for every person 
    [m,~] = size(Allsample{i});
    randse = randperm(m);   %得到随机数列，因为不是所有的文件夹中都有64幅图像可用，故取m
    train_onep = Allsample{i}(randse(1:p),:);   
    test_onep = Allsample{i}(randse(p+1:m),:);
    trainlabel_onep = i * ones(p,1);
    testlabel_onep = i * ones(m-p,1);
    
    % sum to all traindata and testdata 
    TrainData = [TrainData; train_onep];
    TestData = [TestData; test_onep];
    TrainLabel = [TrainLabel; trainlabel_onep];
    TestLabel = [TestLabel; testlabel_onep];
end 


%% PCA降维至50,100,200维（原维度为2016）

trainsamples = TrainData';
testsamples = TestData';
[Pro_Matrix,Mean_Image]=my_pca(trainsamples,1000);
    %Pro_Matrix为投影矩阵
    train_project=Pro_Matrix'*trainsamples;
    test_project=Pro_Matrix'*testsamples;

TrainData = train_project';
TestData = test_project';


%% 归一化

%所有训练集和测试集 
Data = [TrainData;TestData];
Label = [TrainLabel;TestLabel];
%样本顺序打乱形成新的数据
a = size(Data,1);
b = randperm(a);
RandData = Data(b,:);
RandLabel = Label(b);

Guiyi = mapminmax(RandData',-1,1);
NormData = Guiyi';

%% 构造最终的训练集和测试集，训练及测试 

trainNum = size(TrainData,1);
DataNum = size(Data,1);

TrainSample = NormData(1:trainNum,:); 
TrainSample_label = RandLabel(1:trainNum);

TestSample = NormData(trainNum+1:DataNum,:);
TestSample_label = RandLabel(trainNum+1:DataNum);

%训练模型，计算训练所用时间
t1 = clock;
model = svmtrain(TrainSample_label,TrainSample,'-s 0 -t 2 -c 50'); %选用高斯核函数
t2 = clock;
SVMtrainTime = etime(t2,t1);

%测试并计算平均每幅图片分类所用时间
t3 = clock;
[predicted_label, accuracy, decision_values] = svmpredict(TestSample_label,TestSample,model);
t4 = clock;
TestTime = etime(t4,t3);
t = TestTime / size(TestSample,1);

fprintf('模型训练时间为：%3.4fs\n',SVMtrainTime);
fprintf('平均每幅图片分类所用时间为：%3.4fs\n',t);
    





