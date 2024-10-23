% BM 533 NEURAL NETWORKS HW - 1 HEBBIAN GATES - MEHMET GOKALP KOREKEN 

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
trainingMatrix = {trainingMatrix_AND, trainingMatrix_OR, trainingMatrix_XOR};
gateNames = {'AND', 'OR', 'XOR'};

% Training parameters
maxEpochs = 100;
learningRate = 0.1;
errorThreshold = 1e-6;

for j = 1:length(trainingMatrix)
    figure(1 + (j-1)*3);
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

    w1 = 0;
    w2 = 0;
    B = 0;
    
    % Arrays for error history and weight history
    errorHistory = zeros(1, maxEpochs);
    w1History = zeros(1, maxEpochs);
    w2History = zeros(1, maxEpochs);
    bHistory = zeros(1, maxEpochs);
    
    figure(2 + (j-1)*3);
    set(gcf, 'WindowState', 'maximized');
    
    % Create figure for error plot
    figure(3 + (j-1)*3);
    set(gcf, 'WindowState', 'maximized');
    
    for epoch = 1:maxEpochs
        totalError = 0;
        
        for i = 1:length(trainingMatrix{j}{1})
            x1 = trainingMatrix{j}{1}(i);
            x2 = trainingMatrix{j}{2}(i);
            target = trainingMatrix{j}{3}(i);
            
            net = w1*x1 + w2*x2 + B;
            
            output = sign(net);
            if output == 0
                output = 1; 
            end
            
            error = target - output;
            totalError = totalError + error^2;
            
            w1 = w1 + learningRate * x1 * target;
            w2 = w2 + learningRate * x2 * target;
            B = B + learningRate * target;
        end
        
        errorHistory(epoch) = totalError/length(trainingMatrix{j}{1});
        w1History(epoch) = w1;
        w2History(epoch) = w2;
        bHistory(epoch) = B;
        
        if mod(epoch, 10) == 0 || epoch == 1 || epoch == maxEpochs
            figure(2 + (j-1)*3);
            subplot(2, 3, min(6, floor(epoch/10) + 1));
            
            gscatter(trainingMatrix{j}{1}, trainingMatrix{j}{2}, trainingMatrix{j}{3}, 'rb', 'xo');
            hold on;
            
            x1_values = -2:0.1:2;
            if w2 ~= 0
                x2_values = -(w1/w2) * x1_values - (B/w2);
            else
                x2_values = zeros(size(x1_values));
            end
            plot(x1_values, x2_values, 'k-', 'LineWidth', 2);
            
            title(sprintf('Epoch %d', epoch));
            xlabel('x1');
            ylabel('x2');
            grid on;
            axis([-2 2 -2 2]);
            hold off;
        end
        
        figure(3 + (j-1)*3);
        plot(1:epoch, errorHistory(1:epoch), 'b-', 'LineWidth', 2);
        title([gateNames{j}, ' Gate - Training Error']);
        xlabel('Epoch');
        ylabel('Mean Squared Error');
        grid on;
        drawnow;
        
        if totalError < errorThreshold
            break;
        end
    end
    
    figure(1 + (j-1)*3);
    subplot(1,2,2);
    updateWeights(G, h, w1, w2, B);
    
end

