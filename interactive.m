inputfilename = 'rbptest.txt';
filetext = fileread(inputfilename);
lines = strsplit(filetext, '\n');
for i = 1:8
    temp = strsplit(lines{i}, ':');
    lines{i} = temp{2};
end
numunits = str2num(lines{1});
fullyconnected = str2num(lines{2});
if ~fullyconnected
    connections = str2num(lines{3});
end
numvisible = str2num(lines{4});
dt = str2num(lines{5});
timesteps = str2num(lines{6});
weights = zeros(numunits + 1, numunits);
for j = 1:numunits + 1
    for k = 1:numunits
        if ~fullyconnected 
           if connections(j, k) == 0
               weights(j, k) = 0;
           else
               weights(j, k) = randn();
           end
        else
            connections(j, k) = 1;
            weights(j, k) = randn();
        end
    end
end

trainingpairs = str2num(lines{7});
preciseformat = str2num(lines{8});
inputs = cell(trainingpairs);
targets = cell(trainingpairs);
for i = 1:trainingpairs
    temp = strsplit(lines{7 + 2 * i}, ':');
    temp = str2num(temp{2});
    inputs{i} = zeros(timesteps+1, numvisible);
    inputs{i}(:, :) = nan;
    if preciseformat
        for j = 1:length(temp(:, 1))
            inputs{i}(temp(j, 1), temp(j, 2)) = temp(j, 3);
        end
    else
        inputs{i}(1:length(temp(:, 1)), :) = temp;
    end
    temp = strsplit(lines{8 + 2 * i}, ':');
    targets{i} = str2num(temp{2});
end
for i = 1:2000
    for j = 1:trainingpairs
        weights = weights + trainoninput(timesteps, numunits, numvisible, connections, weights, inputs{j}, targets{j});
    end
end
activations = simulate(timesteps, numunits, numvisible, weights, inputs{1});
activations