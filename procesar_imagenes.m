clc;

% Se lee todas los archivos jpg de la carpeta input
files = dir("input/*.jpg");

% Se carga la red neuronal
load('DL0007resnet50.mat')

% Se define el input de la primera capa de la red
inputSize = [720 960];

% Se itera sobre cada archivo de imagen
for i = 1:length(files)

    file = files(i);

    % Se lee la imagen
    relative_path = "input/"+ file.name;
    img = imread(relative_path);
    img_info = imfinfo(relative_path);

    % Se guarda los datos del tamaño original de la imagen
    [rows, columns, channels] = size(img);
    
    % Se reescala el tamaño de la imagen
    img_resized = imresize(img, inputSize);

    % Se aplica el modelo de predicción
    C = predict(net, img_resized);
    V = C(:,:,3)>0.7;
    B=uint8(V).*img_resized;

    % Se regresa al tamaño original
    img_output = imresize(B, [rows, columns]);

    % Se guarda el resultado
    % OJO: falta agregar el metadata de la imagen original
    imwrite(img_output, "output/" +file.name);

end
