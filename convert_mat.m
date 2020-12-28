clear; close all; clc;

%% Load .mat file train data
load('../data/train/digitStruct.mat')

%% Number of samples
nsamples = length(digitStruct);
% nsamples = 10;

%% Convert .mat file to csv file
fid = fopen('train_label.csv', 'w');
fprintf(fid, 'name,height,left,top,width,label\n');
for idx = 1:nsamples
    % Get image name
    name = digitStruct(idx).name;
    % Get box position in image
    ndigit = length(digitStruct(idx).bbox);
    for i = 1:ndigit
        height = digitStruct(idx).bbox(i).height;
        left = digitStruct(idx).bbox(i).left;
        top = digitStruct(idx).bbox(i).top;
        width = digitStruct(idx).bbox(i).width;
        label = digitStruct(idx).bbox(i).label;
        fprintf(fid, '%s,%d,%d,%d,%d,%d\n', name, height, left, top, width, label);
    end
end
fclose(fid);

%% Load .mat file test data
load('../data/test/digitStruct.mat')

%% Number of samples
nsamples = length(digitStruct);
% nsamples = 10;

%% Convert .mat file to csv file
fid = fopen('test_label.csv', 'w');
fprintf(fid, 'name,height,left,top,width,label\n');
for idx = 1:nsamples
    % Get image name
    name = digitStruct(idx).name;
    % Get box position in image
    ndigit = length(digitStruct(idx).bbox);
    for i = 1:ndigit
        height = digitStruct(idx).bbox(i).height;
        left = digitStruct(idx).bbox(i).left;
        top = digitStruct(idx).bbox(i).top;
        width = digitStruct(idx).bbox(i).width;
        label = digitStruct(idx).bbox(i).label;
        fprintf(fid, '%s,%d,%d,%d,%d,%d\n', name, height, left, top, width, label);
    end
end
fclose(fid);
