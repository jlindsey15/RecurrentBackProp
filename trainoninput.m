function changeinweights = trainoninput( timesteps, numunits, numvisible, connections, weights, inputs, targets )
changeinweights = zeros(numunits + 1, numunits);
learningrate = 0.2;
dt = 0.1;
bias = 1;
activations = simulate(timesteps, numunits, numvisible, weights, inputs);
activations(:, end+1) = bias;
directdeltas = zeros(timesteps+1, numunits);
deltas = zeros(timesteps + 1, numunits);
for i = 1:length(targets(:, 1))
    directdeltas(targets(i, 1), targets(i, 2)) = targets(i, 3) - activations(targets(i, 1), targets(i, 2));
end
deltas(timesteps+1, :) = directdeltas(timesteps+1, :);
for time = timesteps:-1:1
    for unit = 1 : numunits
        newdelta = activations(time, unit) * (1 - activations(time, unit)) * deltas(time+1, :) * weights(unit, :)' + directdeltas(time, unit);
        deltas(time, unit) = dt * newdelta + (1 - dt) * deltas(time+1, unit);
    end
end
for unit1 = 1 : numunits + 1
    for unit2 = 1 : numunits
        changeinweights(unit1, unit2) = learningrate * activations(1:end-1, unit1)' * deltas(2:end, unit2);
    end
end
end