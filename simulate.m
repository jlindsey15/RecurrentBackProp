function trueactivations = simulate( timesteps, numunits, numvisible, weights, inputs )
bias = 1;
dt = 0.1;
activations = zeros(timesteps + 1, numunits + 1);
lastinputs = zeros(1, numunits);
inputmatrix = zeros(timesteps+1, numunits);
for time = 1:timesteps + 1
    for unit = 1:numunits
        if time == 1
            newinput = 0;
        else
            newinput = activations(time - 1, :) * weights(:, unit);
        end
        if unit <= numvisible
            if ~isnan(inputs(time, unit))
                newinput = newinput + inputs(time, unit);
            end
        end
        input = dt * newinput + (1 - dt) * lastinputs(unit);
        lastinputs(unit) = input;
        activations(time, unit) = logistic(input);
    end
    activations(time, numunits + 1) = bias;
end
trueactivations = activations(:, 1:end-1);
end
