% BM 533 NEURAL NETWORKS HW - 1 HEBBIAN GATES - MEHMET GOKALP KOREKEN

% Node names
nodeNames = {'w1', 'w2', 'B', 'SUM', 'OUTPUT'};

% Define edges 
s = [1 2 3 4]; 
t = [4 4 4 5]; 

% Create the graph
G = graph(s, t, [], nodeNames);

y = [0.2 0.5 0.8 0.5 0.5];    
x = [0 0 0 0.5 1];            

% Training data for the AND Gate
trainingDataX1_AND = [1,1,-1,-1];
trainingDataX2_AND = [1,-1,1,-1];
targetOutput_AND = [1,-1,-1,-1];
trainingMatrix_AND = {trainingDataX1_AND, trainingDataX2_AND, targetOutput_AND};

% Training data for the OR Gate
trainingDataX1_OR = [1, 1, -1, -1];
trainingDataX2_OR = [1, -1, 1, -1];
targetOutput_OR = [1, 1, 1, -1];
trainingMatrix_OR = {trainingDataX1_OR, trainingDataX2_OR, targetOutput_OR};

% Training data for the XOR Gate
trainingDataX1_XOR = [1, 1, -1, -1];
trainingDataX2_XOR = [1, -1, 1, -1];
targetOutput_XOR = [-1, 1, 1, -1];
trainingMatrix_XOR = {trainingDataX1_XOR, trainingDataX2_XOR, targetOutput_XOR};

% Training Matrix Order is AND OR XOR
trainingMatrix = {trainingMatrix_AND, trainingMatrix_OR, trainingMatrix_XOR };
gateNames = {'AND', 'OR', 'XOR'};

for j = 1 : length(trainingMatrix)
    % Figure 1: Feature Space for Input Data
    figure(1 + (j-1)*2);
    set(gcf, 'WindowState', 'maximized');
    subplot(1,2,1);
    gscatter(trainingMatrix{j}{1}, trainingMatrix{j}{2}, trainingMatrix{j}{3}, 'rb', 'xo');
    title(['Feature Space: ', gateNames{j}, ' Gate']);
    line([0, 0], ylim, 'Color', 'k', 'LineWidth', 1.5, 'LineStyle', '--'); 
    line(xlim, [0, 0], 'Color', 'k', 'LineWidth', 1.5, 'LineStyle', '--'); 
    xlabel('x1 (Input 1)');
    ylabel('x2 (Input 2)');
    grid on;
    axis equal;
    
  
    subplot(1,2,2);
    h = plot(G, 'XData', x, 'YData', y, 'NodeLabel', G.Nodes.Name);
    h.MarkerSize = 8;
    h.NodeColor = 'r';
    h.EdgeColor = 'k';
    h.LineWidth = 1.5;
    title([gateNames{j}, ' Gate Hebbian Structure']);
    axis off;
    
    % initial Weights and bias
    w1 = 0;  
    w2 = 0; 
    B = 0;   
    
    
    % HEBBIAN RULE 
    % x(new) = x(old) + xy
    % b(new) = b(old) + y
    % Decision boundary equation: w1 * x1 + w2 * x2 + B = 0
    % Rearranged: x2 = -(w1/w2) * x1 - (B/w2)
    maxEpochs = 100;  % Maximum number of epochs
    errorHistory = zeros(1, maxEpochs);  % Array to store error values
    tolerance = 1e-3;
    figure(2 + (j-1)*2);
    set(gcf, 'WindowState', 'maximized');

    
    for i = 1 : length(trainingMatrix{j}{1})
        w1 = w1 + trainingMatrix{j}{1}(i) * trainingMatrix{j}{3}(i);  
        w2 = w2 + trainingMatrix{j}{2}(i) * trainingMatrix{j}{3}(i);  
        B = B + trainingMatrix{j}{3}(i);  
    
        x1_values = -1:0.1:1;  
        if w2 ~= 0 % Edge case when w2 is 0
            x2_values = -(w1/w2) * x1_values - (B/w2); 
        else
            x2_values = zeros(size(x1_values));  
        end
        
        % Plot the decision boundary
        subplot(1, length(trainingMatrix{j}{1}), i);
        hold on;
        plot(x1_values, x2_values, 'r-', 'LineWidth', 2);
        hold off;
        
        line([0, 0], ylim, 'Color', 'k', 'LineWidth', 1.5, 'LineStyle', '--'); 
        line(xlim, [0, 0], 'Color', 'k', 'LineWidth', 1.5, 'LineStyle', '--'); 
        title(['Decision Boundary for ', gateNames{j}, ' Gate - Iteration ', num2str(i)]);
        xlabel('x1 (Input 1)');
        ylabel('x2 (Input 2)');
        grid on;
    
        updateWeights(G, h, w1, w2, B);                       
    end
end
