%% Self-Organizing Map (SOM) Mehmet Gökalp Köreken

% Load the data and animal names
data = [
    1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0;
    1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0;
    1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0;
    1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0;
    1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0;
    1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0;
    1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0;
    0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0;
    0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0;
    0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0;
    0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0;
    0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0;
    0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0;
    0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0;
    0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0;
    0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0
];
animal_names = {'Dove', 'Hen', 'Duck', 'Goose', 'Owl', 'Hawk', 'Eagle', ...
    'Fox', 'Dog', 'Wolf', 'Cat', 'Tiger', 'Lion', 'Horse', 'Zebra', 'Cow'};

% Parameters
lattice_sizes = [10, 20, 5]; % Different lattice sizes to evaluate
num_epochs = 100; % Number of training epochs
initial_learning_rate = 0.5; % Initial learning rate
initial_radius = 5; % Initial neighborhood radius

data = data'; % Transpose data to match dimensionality (13 x 16)
num_features = size(data, 1);

for lattice_size = lattice_sizes
    % Initialize the SOM weights randomly
    weights = rand(num_features, lattice_size * lattice_size);

    for epoch = 1:num_epochs
        % Decay learning rate and neighborhood radius over time
        learning_rate = initial_learning_rate * exp(-epoch / num_epochs);
        radius = initial_radius * exp(-epoch / num_epochs);

        for sample_idx = 1:size(data, 2)
            sample = data(:, sample_idx);

            % Find the Best Matching Unit (BMU)
            distances = sum((weights - sample).^2, 1);
            [~, bmu_idx] = min(distances);

            % Calculate BMU position in the lattice
            [bmu_row, bmu_col] = ind2sub([lattice_size, lattice_size], bmu_idx);

            % Update the weights of neurons within the neighborhood
            for i = 1:lattice_size
                for j = 1:lattice_size
                    % Compute the Euclidean distance to the BMU
                    distance_to_bmu = sqrt((i - bmu_row)^2 + (j - bmu_col)^2);

                    if distance_to_bmu <= radius
                        % Compute the influence of the BMU
                        influence = exp(-distance_to_bmu^2 / (2 * radius^2));

                        % Update weights
                        neuron_idx = sub2ind([lattice_size, lattice_size], i, j);
                        weights(:, neuron_idx) = weights(:, neuron_idx) + ...
                            learning_rate * influence * (sample - weights(:, neuron_idx));
                    end
                end
            end
        end
    end

    % Map each data point to the trained lattice
    lattice = strings(lattice_size, lattice_size);
    for sample_idx = 1:size(data, 2)
        sample = data(:, sample_idx);
        distances = sum((weights - sample).^2, 1);
        [~, bmu_idx] = min(distances);

        % Map BMU position in the lattice
        [row, col] = ind2sub([lattice_size, lattice_size], bmu_idx);
        if lattice(row, col) == ""
            lattice(row, col) = animal_names{sample_idx};
        else
            lattice(row, col) = lattice(row, col) + ", " + animal_names{sample_idx};
        end
    end

    % Assign the most likely animal for each lattice node
    node_assignments = strings(lattice_size, lattice_size);
    for i = 1:lattice_size
        for j = 1:lattice_size
            neuron_idx = sub2ind([lattice_size, lattice_size], i, j);
            neuron_weights = weights(:, neuron_idx);

            % Find the closest animal to the neuron weights
            distances_to_animals = sum((data - neuron_weights).^2, 1);
            [~, closest_animal_idx] = min(distances_to_animals);
            node_assignments(i, j) = animal_names{closest_animal_idx};
        end
    end

    % Display the results
    fprintf("Lattice size: %dx%d\n", lattice_size, lattice_size);
    disp(lattice);

    % Plot the assignments
    figure;
    imagesc(1:lattice_size, 1:lattice_size, zeros(lattice_size));
    colormap(gray);
    axis equal;
    axis tight;
    title(sprintf('Most Likely Animals for Lattice %dx%d', lattice_size, lattice_size));
    for i = 1:lattice_size
        for j = 1:lattice_size
            text(j, i, node_assignments(i, j), 'HorizontalAlignment', 'center', 'Color', 'red');
        end
    end
end
% Map each data point to the trained lattice
lattice = strings(lattice_size, lattice_size);
labels = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]; % Labels for 3 families
training_data = [];
training_labels = [];

for sample_idx = 1:size(data, 2)
    sample = data(:, sample_idx);
    distances = sum((weights - sample).^2, 1);
    [~, bmu_idx] = min(distances);

    % Map BMU position in the lattice
    [row, col] = ind2sub([lattice_size, lattice_size], bmu_idx);
    lattice(row, col) = animal_names{sample_idx};

    % Create training data for MLP
    training_data = [training_data; row, col];
    training_labels = [training_labels; labels(sample_idx)];
end

% Multi-Layer Perceptron (MLP) Implementation for Animal Family Classification
hidden_dim = 6;
output_dim = 3;
learning_rate = 0.1;
epochs = 500;

% Initialize weights and biases
W1 = randn(hidden_dim, 2) * 0.01;
b1 = zeros(hidden_dim, 1);
W2 = randn(output_dim, hidden_dim) * 0.01;
b2 = zeros(output_dim, 1);

training_data = training_data / lattice_size; % Normalize input
y_one_hot = full(ind2vec(training_labels')); % Convert labels to one-hot encoding

for epoch = 1:epochs
    % Forward propagation
    hidden_z = (W1 * training_data') + b1;
    hidden_output = 1 ./ (1 + exp(-hidden_z));

    output_z = (W2 * hidden_output) + b2;
    output = exp(output_z) ./ sum(exp(output_z), 1);

    % Compute loss (cross-entropy)
    loss = -sum(sum(y_one_hot .* log(output))) / size(training_data, 1);

    % Backpropagation
    output_error = output - y_one_hot;
    hidden_error = (W2' * output_error) .* (hidden_output .* (1 - hidden_output));

    % Update weights and biases
    W2 = W2 - learning_rate * (output_error * hidden_output');
    b2 = b2 - learning_rate * sum(output_error, 2);

    W1 = W1 - learning_rate * (hidden_error * training_data);
    b1 = b1 - learning_rate * sum(hidden_error, 2);

    if mod(epoch, 100) == 0
        fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
    end
end

% Evaluate MLP Performance
[~, predicted_labels] = max(output, [], 1);
confusion_matrix = confusionmat(training_labels, predicted_labels);

% Display Confusion Matrix
fprintf('Confusion Matrix:\n');
disp(confusion_matrix);

% Plot Decision Boundaries
[x1, x2] = meshgrid(0:0.1:1, 0:0.1:1);
grid_points = [x1(:), x2(:)];

hidden_z = (W1 * grid_points') + b1;
hidden_output = 1 ./ (1 + exp(-hidden_z));
output_z = (W2 * hidden_output) + b2;
output = exp(output_z) ./ sum(exp(output_z), 1);
[~, predictions] = max(output, [], 1);

Z = reshape(predictions, size(x1));
figure;
contourf(x1, x2, Z, 'LineColor', 'none');
colorbar;
title('Decision Boundaries for Animal Families');
xlabel('Normalized Row Index');
ylabel('Normalized Column Index');
total_samples = sum(confusion_matrix(:));

% True Positives, False Positives, False Negatives, and True Negatives
true_positives = diag(confusion_matrix); % Diagonal elements
false_positives = sum(confusion_matrix, 1)' - true_positives; % Column sums minus diagonal
false_negatives = sum(confusion_matrix, 2) - true_positives; % Row sums minus diagonal
true_negatives = total_samples - (true_positives + false_positives + false_negatives);

% Accuracy
accuracy = sum(true_positives) / total_samples;

% Precision, Recall, and F1-Score (for each class)
precision = true_positives ./ (true_positives + false_positives);
recall = true_positives ./ (true_positives + false_negatives);
f1_score = 2 * (precision .* recall) ./ (precision + recall);

% Handle NaN values (if any denominator is zero, set metric to 0)
precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;
f1_score(isnan(f1_score)) = 0;

% Display metrics
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
for i = 1:length(true_positives)
    fprintf('Class %d: Precision = %.2f, Recall = %.2f, F1-Score = %.2f\n', ...
        i, precision(i), recall(i), f1_score(i));
end